# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from nerfstudio.data.scene_box import OrientedBox
# from copy import deepcopy
# from nerfstudio.cameras.rays import RayBundle
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.transforms.functional as TF

from nerfstudio.cameras.cameras import Cameras
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
# from nerfstudio.models.base_model import Model, ModelConfig
import math
import numpy as np
# from sklearn.neighbors import NearestNeighbors
import viser.transforms as vtf
# from nerfstudio.model_components.losses import scale_gauss_gradients_by_distance_squared
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig, GaussianSplattingModel

# from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from nerfstudio.model_components.losses import depth_ranking_loss
from gsplat.sh import SphericalHarmonics, num_sh_bases
from pytorch_msssim import  SSIM


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def projection_matrix(znear, zfar, fovx, fovy, device:Union[str,torch.device]="cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


@dataclass
class LLGaussianSplattingModelConfig(GaussianSplattingModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: LLGaussianSplattingModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.05
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.9
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 60
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.9
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 4
    """maximum degree of spherical harmonics to use"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="off")
    """camera optimizer config"""
    max_gauss_ratio: float = 5.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """

class LLGaussianSplattingModel(GaussianSplattingModel):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: LLGaussianSplattingModelConfig

    def __init__(self, *args, **kwargs):
        if "seed_points" in kwargs:
            self.seed_pts = kwargs["seed_points"]
        else:
            self.seed_pts = None
        super().__init__(*args, **kwargs)
        self.deprojected_new = []
        self.colors_new = []

    def populate_modules(self):
        if self.seed_pts is not None and not self.config.random_init:
            self.means = torch.nn.Parameter(self.seed_pts[0])  # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((20, 3)) - 0.5) * 10)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if self.seed_pts is not None and not self.config.random_init:
            fused_color = RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().cuda()
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
            self.colors_all = torch.nn.Parameter(shs)
        else:
            colors = torch.nn.Parameter(torch.rand(self.num_points, 1, 3))
            shs_rest = torch.nn.Parameter(torch.zeros((self.num_points, dim_sh - 1, 3)))
            self.colors_all = torch.nn.Parameter(torch.cat([colors, shs_rest], dim=1))

        self.opacities = torch.nn.Parameter(torch.logit(0.01 * torch.ones(self.num_points, 1)))

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self.steps_since_add = 0

        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
    
    def add_new_params_to_optimizer(self, optimizer, new_param_groups):
        """
        Adds new parameters to the optimizer, initializing necessary states.

        Args:
            optimizer (torch.optim.Optimizer): The existing optimizer.
            new_param_groups (dict): A dictionary of new parameters to add, categorized by group.
        """
        # import pdb; pdb.set_trace()
        num_new = new_param_groups[0].shape[0]
      

        param = optimizer.param_groups[0]["params"][0]
        # if 'exp_avg' not in optimizer skip the step
        # print(optimizer.state[param].keys())
        # if 'exp_avg' not in optimizer.state[param].keys():
        #     import pdb; pdb.set_trace()

        param_state = optimizer.state[param]
        # print(param_state)
        # if "exp_avg" not in param_state.keys():
        #     import pdb; pdb.set_trace()
        #     print("Nothing in exp_avg")
        

        repeat_dims = (num_new,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))

        
        param_state["exp_avg"] = torch.cat(
            [param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"][-1]).repeat(*repeat_dims)],
            dim=0,
        )
        param_state["exp_avg_sq"] = torch.cat(
            [
                param_state["exp_avg_sq"],
                torch.zeros_like(param_state["exp_avg_sq"][-1]).repeat(*repeat_dims),
            ],
            dim=0,
        )

        del optimizer.state[param]
        optimizer.state[new_param_groups[0]] = param_state

        optimizer.param_groups[0]["params"] = new_param_groups
        del param

    def add_deprojected_means(self, deprojected, colors, optimizers: Optimizers, step):
        if len(deprojected) > 0:
            with torch.no_grad():
                # import pdb; pdb.set_trace()
                # deprojected = torch.cat([x.float() for x in deprojected], dim=1)
                # colors = torch.cat([x.float() for x in colors], dim=1)
                deprojected = torch.stack(deprojected, dim=0)
                colors = torch.stack(colors, dim=0)
                numpts = len(deprojected)
                print("Adding {} new points".format(numpts))
                distances, _ = self.k_nearest_sklearn(deprojected, 3)
                distances = torch.from_numpy(distances)
                # find the average of the three nearest neighbors for each point and use that as the scale
                avg_dist = distances.mean(dim=-1, keepdim=True)/4
                self.means = torch.nn.Parameter(torch.cat([self.means.detach(), deprojected], dim=0))

                self.scales = torch.nn.Parameter(torch.cat([self.scales.detach(), torch.log(avg_dist.repeat(1, 3)).float().cuda()], dim=0))
                self.quats = torch.nn.Parameter(torch.cat([self.quats.detach(), random_quat_tensor(numpts).float().cuda()]))

                dim_sh = num_sh_bases(self.config.sh_degree)
                if colors.max() > 1.0:
                    colors = colors / 255
                    assert colors.max() <= 1.0
                fused_color = RGB2SH(colors)
                
                # colors = torch.nn.Parameter(torch.rand(numpts, 1, 3))
                # shs_rest = torch.nn.Parameter(torch.zeros((numpts, dim_sh - 1, 3)))
                # self.colors_all = torch.nn.Parameter(torch.cat([self.colors_all.detach(), torch.cat([colors, shs_rest], dim=1).to(self.device)], dim=0))
                
                shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().cuda()
                shs[:, 0, :3] = fused_color
                shs[:, 1:, 3:] = 0.0
                self.colors_all = torch.nn.Parameter(torch.cat([self.colors_all.detach(), shs.to(self.device)], dim=0))
                
                self.opacities = torch.nn.Parameter(torch.cat([self.opacities.detach(), torch.logit(0.25 * torch.ones(numpts, 1)).to(self.device)], dim=0))

                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None
                
                num_new_points = deprojected.shape[0]
                
                # Adding only the new parameters to the optimizer
                # new_gaussian_params = [new_means, new_scales, new_quats, new_colors_all, new_opacities]
                param_groups = self.get_gaussian_param_groups()
                for group, param in param_groups.items():
                    # import pdb; pdb.set_trace()
                    new_param = [param[0][-num_new_points:]]
                    self.add_new_params_to_optimizer(optimizers.optimizers[group], new_param)
            self.deprojected_new.clear()
            self.colors_new.clear()
            self.steps_since_add = 0

    def refinement_after(self, optimizers: Optimizers, step):
        if self.step >= self.config.warmup_length:
            with torch.no_grad():
                # only split/cull if we've seen every image since opacity reset
                reset_interval = self.config.reset_alpha_every * self.config.refine_every
                if (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval > self.num_train_data + self.config.refine_every
                ):
                    # then we densify
                    assert (
                        self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                    )
                    avg_grad_norm = (
                        (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                    )
                    high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                    splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                    if self.step < self.config.stop_screen_size_at:
                        splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                    splits &= high_grads
                    nsamps = self.config.n_split_samples
                    (
                        split_means,
                        split_colors,
                        split_opacities,
                        split_scales,
                        split_quats,
                    ) = self.split_gaussians(splits, nsamps)

                    dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                    dups &= high_grads
                    dup_means, dup_colors, dup_opacities, dup_scales, dup_quats = self.dup_gaussians(dups)
                    self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                    self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))

                    self.opacities = Parameter(
                        torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0)
                    )
                    self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                    self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat(
                        [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                        dim=0,
                    )
                    split_idcs = torch.where(splits)[0]
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.dup_in_optim(optimizers.optimizers[group], split_idcs, param, n=nsamps)
                    dup_idcs = torch.where(dups)[0]

                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.dup_in_optim(optimizers.optimizers[group], dup_idcs, param, 1)

                # Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
                if self.step % reset_interval > self.num_train_data + self.config.refine_every:
                    # then cull
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)

                if self.steps_since_add % reset_interval == self.config.refine_every:
                    # reset_value = self.config.cull_alpha_thresh * 0.95
                    reset_value = 0.15
                    self.opacities.data = torch.full_like(
                        self.opacities.data, torch.logit(torch.tensor(reset_value)).item()
                    )
                    # reset the exp of optimizer
                    optim = optimizers.optimizers["opacity"]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def after_train(self, step: int):
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.grad.detach().norm(dim=-1)  # TODO fill in
            # import pdb; pdb.set_trace()
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                # self.vis_counts = torch.ones_like(visible_mask.float())
                self.xys_grad_norm = grads
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1]))
            )

    def step_cb(self, step):
        self.step = step
        self.steps_since_add += 1

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.add_deprojected_means,
                args=[self.deprojected_new, self.colors_new, training_callback_attributes.optimizers],
            )
        )
        return cbs


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        if self.training:
            # currently relies on the branch vickie/camera-grads
            self.camera_optimizer.apply_to_camera(camera)
        if self.training:
            background = torch.rand(3, device=self.device)
        else:
            background = self.back_color
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": background.repeat(camera.height.item(), camera.width.item(), 1)}
        else:
            crop_ids = None
        # camera_downscale = self._get_downscale_factor()
        # camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z axis to align with gsplat conventions
        R_edit = torch.tensor(vtf.SO3.from_x_radians(np.pi).as_matrix(), device=R.device, dtype=R.dtype)
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            colors_crop = self.colors_all[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            colors_crop = self.colors_all
            scales_crop = self.scales
            quats_crop = self.quats
        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )
        # import pdb; pdb.set_trace()

        if (self.radii).sum() == 0:
            return {"rgb": background.repeat(camera.height.item(), camera.width.item(), 1)}

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()
        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = SphericalHarmonics.apply(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = self.get_colors.squeeze()  # (N, 3)
            rgbs = torch.sigmoid(rgbs)
        rgb = RasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background,
        )
        depth_im = None
        # if not self.training:
        #     depth_im = RasterizeGaussians.apply(
        #         self.xys,
        #         depths,
        #         self.radii,
        #         conics,
        #         num_tiles_hit,
        #         depths[:, None].repeat(1, 3),
        #         torch.sigmoid(opacities_crop),
        #         H,
        #         W,
        #         torch.ones(3, device=self.device) * 10,
        #     )[..., 0:1]
        # # rescale the camera back to original dimensions
        # camera.rescale_output_resolution(camera_downscale)
        return {"rgb": rgb, "depth": depth_im}
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        # d = self._get_downscale_factor()
        # import pdb; pdb.set_trace()

        # if d > 1:
        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
        #     gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
        #     gt_img = batch["image"]
        gt_img = batch["image"]
        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        predicted_rgb = outputs["rgb"]
        assert predicted_rgb.shape == gt_rgb.shape
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict["gaussian_count"] = self.num_points
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # d = self._get_downscale_factor()
        # if d > 1:
        #     newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
        #     gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        # else:
        #     gt_img = batch["image"]
        gt_img = batch["image"]
        Ll1 = torch.abs(gt_img - outputs["rgb"]).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])
        if self.step % 10 == 0:
            # Before, we made split sh and colors onto different optimizer, with shs having a low learning rate
            # This is slow, instead we apply a regularization every few steps
            sh_reg = self.colors_all[:, 1:, :].norm(dim=1).mean()
            scale_exp = torch.exp(self.scales)
            scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(self.config.max_gauss_ratio)) - self.config.max_gauss_ratio
            scale_reg = 0.1 * scale_reg.mean()
        else:
            sh_reg = torch.tensor(0.0).to(self.device)
            scale_reg = torch.tensor(0.0).to(self.device)
        return {"main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss, "sh_reg": sh_reg,'scale_reg':scale_reg}
