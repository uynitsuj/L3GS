from __future__ import annotations
from nerfstudio.data.scene_box import OrientedBox
import dataclasses
import functools
import os
import time
import nerfstudio
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast
import viser.transforms as vtf
import torch
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from cv_bridge import CvBridge  # Needed for converting between ROS Image messages and OpenCV images

from nerfstudio.cameras.camera_utils import get_distortion_params
from nerfstudio.cameras.cameras import Cameras,CameraType
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.process_data.colmap_utils import qvec2rotmat
from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils import profiler, writer
from copy import deepcopy
from nerfstudio.utils.decorators import check_eval_enabled, check_main_thread, check_viewer_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server.viewer_state import ViewerState
from nerfstudio.viewer_beta.viewer import Viewer as ViewerBetaState
from nerfstudio.viewer_beta.viewer_elements import ViewerButton, ViewerCheckbox, ViewerNumber, ViewerText, ViewerDropdown

import nerfstudio.utils.poses as pose_utils
import numpy as np
import scipy.spatial.transform as transform
import rclpy
from rclpy.node import Node
from lifelong_msgs.msg import ImagePose
from l3gos.L3GOS_pipeline import L3GOSPipeline
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped,Pose
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
import trimesh

TORCH_DEVICE = str
TRAIN_ITERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]

def ros_pose_to_nerfstudio(pose_msg: Pose, static_transform=None):
    """
    Takes a ROS Pose message and converts it to the
    3x4 transform format used by nerfstudio.
    """
    quat = np.array(
        [
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
        ],
    )
    posi = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    R = torch.tensor(qvec2rotmat(quat))
    T = torch.cat([R, posi.unsqueeze(-1)], dim=-1)
    T = T.to(dtype=torch.float32)
    if static_transform is not None:
        T = pose_utils.multiply(T, static_transform)
        T2 = torch.zeros(3, 4)
        R1 = transform.Rotation.from_euler("x", 90, degrees=True).as_matrix()
        R2 = transform.Rotation.from_euler("z", 180, degrees=True).as_matrix()
        R3 = transform.Rotation.from_euler("y", 180, degrees=True).as_matrix()
        R = torch.from_numpy(R3 @ R2 @ R1)
        T2[:, :3] = R
        T = pose_utils.multiply(T2, T)


    return T.to(dtype=torch.float32)

@dataclass
class TrainerConfig(ExperimentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: Trainer)
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    load_checkpoint: Optional[Path] = None
    """Path to checkpoint file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients over."""

class TrainerNode(Node):
    def __init__(self,trainer):
        super().__init__('trainer_node')
        self.trainer_ = trainer
        self.subscription_ = self.create_subscription(ImagePose,"/camera/color/imagepose",self.add_img_callback,100)

    def add_img_callback(self,msg):
        print("Appending imagepose to queue",flush=True)
        self.trainer_.image_add_callback_queue.append(msg)


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: L3GOSPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')
        self.train_lock = Lock()
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device: TORCH_DEVICE = config.machine.device_type
        if self.device == "cuda":
            self.device += f":{local_rank}"
        self.mixed_precision: bool = self.config.mixed_precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler
        self.training_state: Literal["training", "paused", "completed"] = "paused"
        self.gradient_accumulation_steps: int = self.config.gradient_accumulation_steps

        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)

        self.base_dir: Path = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir: Path = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

        self.viewer_state = None
        self.image_add_callback_queue = []
        self.image_process_queue = []
        self.query_diff_queue = []
        self.query_diff_size = 5*3
        self.query_diff_msg_queue = []
        self.cvbridge = CvBridge()
        self.clip_out_queue = mp.Queue()
        self.dino_out_queue = mp.Queue()
        self.done_scale_calc = False
        self.calculate_diff = False # whether or not to calculate the image diff
        self.calulate_metrics = False # whether or not to calculate the metrics
        self.num_boxes_added = 0
        self.box_viser_handles = []


    def handle_stage_btn(self, handle: ViewerButton):
        import os.path as osp
        bag_path = osp.join(
            # '/home/lerf/lifelong-lerf/experiment_bags',
            '/home/lerf/lifelong-lerf/bag',
            self.pipeline.lifelong_exp_aname.value,
            f'loop{self.pipeline.lifelong_exp_loop.value + 1}'
        )
        # check if the bag exists
        if not osp.exists(bag_path):
            print("Bag not found at", bag_path)
            return
        self.pipeline.lifelong_exp_aname.set_disabled(True) #dropdown disabling doesn't seem to work?
        
        if self.pipeline.lifelong_exp_loop.value > 0:
            self.pipeline.datamanager.train_dataset.stage.append(len(self.pipeline.datamanager.train_dataset))
            self.calculate_diff = True
            print("Stage set to", self.pipeline.datamanager.train_dataset.stage[-1])
    
        self.query_diff_msg_queue = []
        self.query_diff_queue = []

        # # keeping just in case...
        # if self.pipeline.lifelong_exp_loop.value > 0: # i.e., exp_loop = 2 means that youve already played loop 1 and 2.
        #     stage = self.pipeline.datamanager.train_dataset.stage
        #     mask = self.pipeline.datamanager.train_dataset.mask_tensor[stage[-2]:stage[-1], ...] #Mask is NxHxWx1
        #     mask_sum = torch.sum(mask)
        #     # Calculate the percentage of masked pixels
        #     percentage_masked = 1 - (mask_sum / (mask.shape[0] * mask.shape[1] * mask.shape[2]))
        #     # Save the percentage to a file with name of the stage
        #     with open(f"metrics/metrics_{self.pipeline.lifelong_exp_aname.value}_{self.pipeline.lifelong_exp_loop.value}.txt", "w") as f:
        #         f.write(f"Precentage Masked: {percentage_masked}\n")   
        self.pipeline.stage_button.set_disabled(True)
        self.pipeline.lifelong_exp_loop.value += 1

        self.num_boxes_added = 0 # reset the mask boxes of this scene
        for bbox_viser in self.box_viser_handles:
            bbox_viser.remove()

        # start the bag
        # self.pipeline.lifelong_exp_start.set_disabled(True)
        import subprocess
        proc = subprocess.Popen(['ros2', 'bag', 'play', bag_path, "--rate", "1.0","--topics", "/camera/color/camera_info", "/sim_realsense"])
        print("Started bag at", bag_path)
        proc.communicate()
        proc.terminate()
        print("Terminated bag at", bag_path)
        self.pipeline.stage_button.set_disabled(False)
        
    def handle_calc_metric(self, handle: ViewerButton):
        self.pipeline.calc_metric.disabled = True
        self.calulate_metrics = True

    def handle_percentage_masked(self, handle: ViewerButton):
        # Calculate the sum of masked pixels
        stage = self.pipeline.datamanager.train_dataset.stage
        mask = self.pipeline.datamanager.train_dataset.mask_tensor[stage[-2]:stage[-1], ...] #Mask is NxHxWx1
        mask_sum = torch.sum(mask)
        # Calculate the percentage of masked pixels
        percentage_masked = 1 - (mask_sum / (mask.shape[0] * mask.shape[1] * mask.shape[2]))
        # Save the percentage to a file with name of the stage
        with open(f"metrics/metrics_{self.pipeline.lifelong_exp_aname.value}_{self.pipeline.lifelong_exp_loop.value}.txt", "w") as f:
            f.write(f"Precentage Masked: {percentage_masked}\n")

    def calc_metric(self):
        self.pipeline.model.eval()
        from tqdm import tqdm
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        import pdb; pdb.set_trace()
        cameras = self.pipeline.datamanager.train_dataset.cameras[:len(self.pipeline.datamanager.train_dataset)]
        stage = self.pipeline.datamanager.train_dataset.stage[-1]
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        import matplotlib.pyplot as plt
        for i in tqdm(range(stage, len(self.pipeline.datamanager.train_dataset))):
            cam = cameras[i].to(self.device)
            gt_rgb = self.pipeline.datamanager.train_dataset.image_tensor[i].to(self.device)
            gt_rgb = gt_rgb.permute(2,0,1).unsqueeze(0)
            with torch.no_grad():
                ray_bundle = cam.generate_rays(0)
                ray_bundle.metadata['rgb_only'] = torch.ones((ray_bundle.origins.shape[0],ray_bundle.origins.shape[1], 1), dtype=torch.bool, device=ray_bundle.origins.device)
                predicted_rgb = self.pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)['rgb'].to(self.device)
            predicted_rgb = predicted_rgb.permute(2,0,1).unsqueeze(0)
            psnr = self.psnr(gt_rgb, predicted_rgb)
            ssim = self.ssim(gt_rgb, predicted_rgb)
            lpips = self.lpips(gt_rgb, predicted_rgb)
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(gt_rgb[0].permute(1,2,0).cpu().numpy())
            axes[0].set_title('GT image')
            axes[1].imshow(predicted_rgb[0].permute(1,2,0).cpu().numpy())
            axes[1].set_title(f'Predicted image, PSNR={psnr}')
            plt.show()
            avg_psnr += psnr
            avg_ssim += ssim
            avg_lpips += lpips
        avg_psnr /= (len(cameras) - stage)
        avg_ssim /= (len(cameras) - stage)
        avg_lpips /= (len(cameras) - stage)
        import pdb; pdb.set_trace()

        #Save the metrics to a file with name of the stage
        with open(f"metrics/metrics_{self.pipeline.lifelong_exp_aname.value}_{self.pipeline.lifelong_exp_loop.value}.txt", "w") as f:
            f.write(f"PSNR: {avg_psnr}\n")
            f.write(f"SSIM: {avg_ssim}\n")
            f.write(f"LPIPS: {avg_lpips}\n")
        self.pipeline.calc_metric.disabled = False
        self.calulate_metrics = False
        self.pipeline.model.train()



    # def handle_start_bag(self, handle: ViewerButton):
    #     # find path for the bag
    #     import os.path as osp
    #     bag_path = osp.join(
    #         # '/home/lerf/lifelong-lerf/experiment_bags',
    #         '/home/lerf/lifelong-lerf/bag',
    #         self.pipeline.lifelong_exp_aname.value,
    #         f'loop{self.pipeline.lifelong_exp_loop.value}'
    #     )
    #     # check if the bag exists
    #     if not osp.exists(bag_path):
    #         print("Bag not found at", bag_path)
    #         return
    #     # start the bag
    #     self.pipeline.lifelong_exp_start.set_disabled(True)
    #     import subprocess
    #     with subprocess.Popen(['ros2', 'bag', 'play', bag_path, "--rate", "1.0", "--topics", "/camera/color/camera_info", "/sim_realsense"]) as proc:
    #         print("Started bag at", bag_path)
    #         proc.communicate()
    #     print("Terminated bag at", bag_path)
    #     self.pipeline.stage_button.set_disabled(False)
    #     # print("Started bag at", bag_path)
    #     # proc.communicate()
    #     # proc.terminate()
    #     # print("Terminated bag at", bag_path)
    #     # self.training_state = "training"

    def handle_start_droidslam(self, handle: ViewerButton):
        import subprocess
        with subprocess.Popen(['env', 'CUDA_VISIBLE_DEVICES=0', 'python', 'ros_node.py'], cwd='/home/lerf/DROID-SLAM') as proc:
            print("Started droidslam")
            self.pipeline.droidslam_start.set_disabled(True)
            self.pipeline.stage_button.set_disabled(False)
            proc.communicate()
        print("Terminated droidslam")
        # proc.terminate()


    def add_img_callback(self, msg:ImagePose, decode_only=False):
        '''
        this function queues things to be added
        returns the image, depth, and pose if the dataparser is defined yet, otherwise None
        if decode_only, don't add the image to the clip/dino queue using `add_image`.
        '''
        image: Image = msg.img
        camera_to_worlds = ros_pose_to_nerfstudio(msg.pose)
        # CONSOLE.print("Adding image to dataset")
        # image_data = torch.tensor(image.data, dtype=torch.uint8).view(image.height, image.width, -1).to(torch.float32)/255.
        image_data = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.img,'rgb8'),dtype = torch.float32)/255.
        # By default the D4 VPU provides 16bit depth with a depth unit of 1000um (1mm).
        # --> depth_data is in meters.
        depth_data = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.depth,'16UC1') / 1000. ,dtype = torch.float32)
        fx = torch.tensor([msg.fl_x])
        fy = torch.tensor([msg.fl_y])
        cy = torch.tensor([msg.cy])
        cx = torch.tensor([msg.cx])
        width = torch.tensor([msg.w])
        height = torch.tensor([msg.h])
        distortion_params = get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3)
        camera_type = CameraType.PERSPECTIVE
        c = Cameras(camera_to_worlds, fx, fy, cx, cy, width, height, distortion_params, camera_type)
        retc = deepcopy(c)
        img_out=image_data.clone()
        dep_out=depth_data.clone()
        if not decode_only:
            # with self.train_lock:
            self.pipeline.add_image(img = image_data, pose = c)
        dep_out *= self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        #TODO add the dataparser transform here
        H = self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        row = torch.tensor([[0,0,0,1]],dtype=torch.float32,device=retc.camera_to_worlds.device)
        retc.camera_to_worlds = torch.matmul(torch.cat([H,row]),torch.cat([retc.camera_to_worlds,row]))[:3,:]
        retc.camera_to_worlds[:3,3] *= self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        if not self.done_scale_calc:
            return None,None,None
        return img_out, dep_out, retc

    def process_image(self, msg:ImagePose, clip_dict, dino_data):
        '''
        This function actually adds things to the dataset
        '''
        camera_to_worlds = ros_pose_to_nerfstudio(msg.pose)
        # CONSOLE.print("Adding image to dataset")
        image_data = torch.tensor(self.cvbridge.imgmsg_to_cv2(msg.img,'rgb8'),dtype = torch.float32)/255.
        fx = torch.tensor([msg.fl_x])
        fy = torch.tensor([msg.fl_y])
        cy = torch.tensor([msg.cy])
        cx = torch.tensor([msg.cx])
        width = torch.tensor([msg.w])
        height = torch.tensor([msg.h])
        distortion_params = get_distortion_params(k1=msg.k1,k2=msg.k2,k3=msg.k3)
        camera_type = CameraType.PERSPECTIVE
        c = Cameras(camera_to_worlds, fx, fy, cx, cy, width, height, distortion_params, camera_type)
        with self.train_lock:
            self.pipeline.process_image(img = image_data, pose = c, clip=clip_dict, dino=dino_data)
        print("Done processing image")

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
            clip_out_queue=self.clip_out_queue,
            dino_out_queue=self.dino_out_queue,
        )

        # self.pipeline.lifelong_exp_aname = ViewerText(name="Exp name", default_value="")
        self.pipeline.lifelong_exp_aname = ViewerDropdown(
            name="Exp name", 
            options=[""] + sorted(os.listdir('/home/lerf/lifelong-lerf/bag')), 
            default_value="")
        self.pipeline.lifelong_exp_loop = ViewerNumber(name="Exp loop", default_value=0, disabled=True)
        self.pipeline.droidslam_start = ViewerButton(name="Start Droidslam", cb_hook=self.handle_start_droidslam)

        # self.pipeline.lifelong_exp_start = ViewerButton(name="Start Exp", cb_hook=self.handle_start_bag, disabled=True)
        self.pipeline.stage_button = ViewerButton(name="Update Stage", cb_hook=self.handle_stage_btn, disabled=True)
        self.pipeline.calc_metric = ViewerButton(name="Calculate Metric", cb_hook=self.handle_calc_metric)
        self.pipeline.percentage_masked = ViewerButton(name="Percent Masked", cb_hook=self.handle_percentage_masked)
        self.pipeline.plot_verbose = ViewerCheckbox(name="Plot Verbose", default_value=True)

        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        self.viewer_log_path = viewer_log_path

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
            )
        )
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=self.viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_beta_enabled() and self.local_rank == 0:
            self.viewer_state = ViewerBetaState(
                self.config.viewer,
                log_filename=self.viewer_log_path,
                datapath=self.base_dir,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Viewer Beta at: {self.viewer_state.viewer_url}"]
        self._check_viewer_warnings()
        self._init_viewer_state()

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        camera_optimizer_config = self.config.pipeline.datamanager.camera_optimizer
        if camera_optimizer_config is not None and camera_optimizer_config.mode != "off":
            assert camera_optimizer_config.param_group not in optimizer_config
            optimizer_config[camera_optimizer_config.param_group] = {
                "optimizer": camera_optimizer_config.optimizer,
                "scheduler": None,
            }
        return Optimizers(optimizer_config, param_groups)

    def train(self) -> None:
        print("IM IN")
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        rclpy.init(args=None)
        trainer_node = TrainerNode(self)
        parser_scale_list = []
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            
            while True:
                rclpy.spin_once(trainer_node,timeout_sec=0.00)

                # Check if we have an image to add. We add *one* image per iteration.
                has_image_add = len(self.image_add_callback_queue) > 0
                if has_image_add:
                    #Not sure if we want to loop till the queue is empty or not
                    msg = self.image_add_callback_queue.pop(0)

                    # if we are actively calculating diff for the current scene,
                    # we don't want to add the image to the dataset unless we are sure.
                    if self.calculate_diff:
                        image, depth, pose = self.add_img_callback(msg, decode_only=True)
                        self.query_diff_msg_queue.append(msg)
                        self.query_diff_queue.append([image,depth,pose])
                    else:
                        image, depth, pose = self.add_img_callback(msg)
                        self.image_process_queue.append(msg)

                    if not self.done_scale_calc:
                        parser_scale_list.append(msg.pose)
                        
                # set stage_button to be disabled if we haven't trained sufficient # of steps at the current stage.
                # we want to move on from stage 1 at 1500 steps, stage 2 at 3000 steps, etc.
                if self.pipeline.lifelong_exp_loop.value > 0:
                    if step < self.pipeline.lifelong_exp_loop.value*500:
                        self.pipeline.stage_button.set_disabled(True)
                    else:
                        self.pipeline.stage_button.set_disabled(False)
        
                # If we are paused, then we don't train.
                if self.training_state == "paused":
                    time.sleep(0.01)
                    continue
                # Even if we are supposed to "train", if we don't have enough images we don't train.
                elif not self.done_scale_calc and (len(parser_scale_list)<10):
                    time.sleep(0.01)
                    continue

                #######################################
                # Starting training
                #######################################

                # Create scene scale based on the images collected so far. This is done once.
                if not self.done_scale_calc:
                    self.done_scale_calc = True
                    from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses
                    poses = [np.concatenate([ros_pose_to_nerfstudio(p),np.array([[0,0,0,1]])],axis=0) for p in parser_scale_list]
                    poses = torch.from_numpy(np.stack(poses).astype(np.float32))#TODO THIAS LINE WRONG
                    poses, transform_matrix = auto_orient_and_center_poses(
                        poses,
                        method='up',
                        center_method='poses'
                    )
                    scale_factor = 1.0
                    scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
                    self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform = transform_matrix
                    self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform = transform_matrix
                    self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale = scale_factor
                    self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale = scale_factor

                # Check if we have an image to process, and add *all of them* to the dataset per iteration.
                while not self.clip_out_queue.empty() and not self.dino_out_queue.empty() and self.done_scale_calc:
                    self.process_image(self.image_process_queue.pop(0), self.clip_out_queue.get(), self.dino_out_queue.get())

                step +=1 

                with self.train_lock:
                    #TODO add the image diff stuff here
                    if len(self.query_diff_queue) > self.query_diff_size and self.calculate_diff and image is not None:
                        self.pipeline.eval()
                        heat_map_masks, poses, lerf_depths, depth_distances, lerf_outputs_list = [], [], [], [], []
                        for _ in range(0, self.query_diff_size, 3):
                            image, depth, pose = self.query_diff_queue.pop(0)
                            # convert to z_depth if depth compositing is enabled
                            heat_map, lerf_depth, lerf_outputs = self.pipeline.query_diff(image, pose, vis_verbose=self.pipeline.plot_verbose.value, depth=depth)
                            # heat_map_mask = ((heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())) > 0.7
                            if self.pipeline.use_clip:
                                heat_map_mask = heat_map > -0.85
                            else:
                                heat_map_mask = heat_map
                                
                            pose_ = deepcopy(pose)
                            pose_.rescale_output_resolution(1/self.pipeline.highres_downscale)
                            pose_ = pose_.to(self.device)
                            R = pose_.camera_to_worlds[0:3,0:3].T
                            camera_ray_bundle = pose_.generate_rays(camera_indices=0)
                            pts = (camera_ray_bundle.directions*lerf_outputs['depth'])
                            pts = (R@(pts.view(-1,3).T)).T.view(*camera_ray_bundle.directions.shape)
                            # lerf_depth_z = -pts[...,2:3] #negative z axis is the coordinate convention

                            pose_ = deepcopy(pose)
                            pose_ = pose_.to(self.device)
                            R = pose_.camera_to_worlds[0:3,0:3].T
                            camera_ray_bundle = pose_.generate_rays(camera_indices=0)
                            
                            # depth_distance = depth.to(self.device)/camera_ray_bundle.directions[...,2]
                            depth_distance = depth.to(self.device) / (-camera_ray_bundle.directions @ pose.camera_to_worlds[:3, 2].to(self.device))

                            if self.pipeline.plot_verbose.value:
                                fig, axes = plt.subplots(1, 9)
                                axes[0].imshow(lerf_outputs['rgb'].detach().cpu().numpy())
                                axes[0].set_title("LERF rgb")
                                axes[1].imshow(image.detach().cpu().numpy())
                                axes[1].set_title("Current rgb")
                                axes[2].imshow(lerf_outputs['depth'].detach().cpu().numpy(), vmin=0, vmax=2)
                                axes[2].set_title("LERF depth")
                                axes[3].imshow(depth_distance.detach().cpu().numpy(), vmin=0, vmax=2)
                                axes[3].set_title("Current depth")
                                axes[4].imshow(heat_map.detach().cpu().numpy(), vmin=-1, vmax=0)
                                axes[4].set_title("Heatmap (varying)")
                                axes[5].imshow(heat_map.detach().cpu().numpy())
                                axes[5].set_title("Heatmap (varying), normalized")
                                axes[6].imshow(heat_map_mask.detach().cpu().numpy())
                                axes[6].set_title("Heatmap mask (binary)")

                                min_depth = torch.where(
                                    (lerf_outputs['depth'].squeeze() < depth_distance[::4, ::4]), 
                                    lerf_outputs['depth'].squeeze(), 
                                    depth_distance[::4, ::4]
                                    )
                                axes[7].imshow(min_depth.detach().cpu().numpy(), vmin=0, vmax=2)
                                axes[7].set_title("Min depth")
                                axes[8].imshow((lerf_outputs['depth'].squeeze() - depth_distance[::4, ::4]).abs().detach().cpu().numpy())
                                axes[8].set_title("Min depth - current depth")

                                plt.show()
                                print("lerf image")
                                plt.imshow(lerf_outputs['rgb'].detach().cpu().numpy())
                                plt.show()
                                print("updated image")
                                plt.imshow(image.detach().cpu().numpy())
                                plt.show()

                            heat_map_masks.append(heat_map_mask)
                            poses.append(pose)
                            lerf_depths.append(lerf_depth)
                            depth_distances.append(depth_distance)
                            lerf_outputs_list.append(lerf_outputs)

                        boxes, points_tr = self.pipeline.heatmaps2box(heat_map_masks, poses, lerf_depths, depth_distances, lerf_outputs_list)
                        print("just got the box")
                        print("boxes: ", boxes)
                        if len(boxes) > 0:
                            # Change detected!
                            self.viewer_state.viser_server.add_point_cloud(
                                '/pointcloud',
                                points=points_tr.vertices * 10,
                                colors=points_tr.visual.vertex_colors[:, :3]
                            )
                            for obox_ind, obox in enumerate(boxes):
                                # TODO(cmk) this creates a bunch of "/bbox_{obox_ind}", 
                                # but we probably don't want ti visualize old boxes if we do this for a long run,
                                # this is only ideal if we detect the change *once* in the run.
                                # Adam says that it would be best if mask_volume takes in a list of boxes,
                                # and every time `mask_volume` is called we can remove all previous boxes.
                                bbox_viser = self.pipeline.mask_volume(obox, self.num_boxes_added + obox_ind, self.viewer_state.viser_server)
                                self.box_viser_handles.append(bbox_viser)
                            self.num_boxes_added += len(boxes)
                            self.viewer_state.update_masks = True
                            print("just visualized the box.")
                            # The images for the updated scene now need to be included in the dataset.
                            for _ in range(self.query_diff_size):
                                if len(self.query_diff_msg_queue) != 0:
                                    msg = self.query_diff_msg_queue.pop(0)
                                    image, depth, pose = self.add_img_callback(msg) # --> this is problem
                                    self.image_process_queue.append(msg)
                        else:
                            # No change detected! --> We don't add the current run to the dataset.
                            # Just flush the self.query_diff_msg_queue.
                            # Also, don't check diff for the rest of this stage. 
                            print("no change detected.")
                            self.calculate_diff = False
                            # We're never going to come back to this stage, so we need to flush the queue completely.
                            self.query_diff_msg_queue = []
                            self.query_diff_queue = []
                            # for _ in range(self.query_diff_size):
                            #     if len(self.query_diff_msg_queue) == 0:
                            #         break
                            #     msg = self.query_diff_msg_queue.pop(0)

                        self.pipeline.train()

                    #######################################
                    # Normal training loop
                    #######################################
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )
                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

                #After a training loop is done we check to calc metric
                if self.calulate_metrics:
                    self.calc_metric()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if (
            (self.config.is_viewer_enabled() or self.config.is_viewer_beta_enabled())
            and not self.config.is_tensorboard_enabled()
            and not self.config.is_wandb_enabled()
        ):
            string: str = (
                "[NOTE] Not running eval iterations since only viewer is enabled.\n"
                "Use [yellow]--vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard}[/yellow] to run with eval."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state is not None and self.pipeline.datamanager.train_dataset is not None
        self.viewer_state.init_scene(
            train_dataset=self.pipeline.datamanager.train_dataset,
            train_state="training",
            eval_dataset=self.pipeline.datamanager.eval_dataset,
        )

    @check_viewer_enabled
    def _update_viewer_state(self, step: int) -> None:
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

    @check_viewer_enabled
    def _train_complete_viewer(self) -> None:
        """Let the viewer know that the training is complete"""
        assert self.viewer_state is not None
        self.training_state = "completed"
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")
        CONSOLE.print("Use ctrl+c to quit", justify="center")
        while True:
            time.sleep(0.01)

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int) -> None:
        """Performs update on rays/sec calculation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=self.world_size * train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_ITERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        assert (
            self.gradient_accumulation_steps > 0
        ), f"gradient_accumulation_steps must be > 0, not {self.gradient_accumulation_steps}"

        for _ in range(self.gradient_accumulation_steps):
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                loss = functools.reduce(torch.add, loss_dict.values())
                print(loss_dict)
                if torch.isnan(loss).any():
                    print("NAN LOSS")
                    import pdb;pdb.set_trace()
                loss /= self.gradient_accumulation_steps
            self.grad_scaler.scale(loss).backward()  # type: ignore
        # clip grads
        torch.nn.utils.clip_grad_norm_(self.pipeline.model.parameters(), np.interp(step,[0,200],[10,1000]))
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)


        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
