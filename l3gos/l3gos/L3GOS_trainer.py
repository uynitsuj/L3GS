import torch
import rclpy
import numpy as np
from rclpy.node import Node
from lifelong_msgs.msg import ImagePose
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, cast

from nerfstudio.process_data.colmap_utils import qvec2rotmat
from nerfstudio.configs.experiment_config import ExperimentConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from lerf.lerf_pipeline import LERFPipeline
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

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped,Pose
import lerf.utils.query_diff_utils as query_diff_utils
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
import trimesh


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
