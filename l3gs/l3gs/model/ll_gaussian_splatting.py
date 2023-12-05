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
from copy import deepcopy
from nerfstudio.cameras.rays import RayBundle
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.cameras import Cameras
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import viser.transforms as vtf
from nerfstudio.model_components.losses import scale_gauss_gradients_by_distance_squared
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig, GaussianSplattingModel

from torchmetrics.image import StructuralSimilarityIndexMeasure
from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from nerfstudio.model_components.losses import depth_ranking_loss
from gsplat.sh import SphericalHarmonics, num_sh_bases
from pytorch_msssim import  SSIM


def random_quat_tensor(N, **kwargs):
    u = torch.rand(N, **kwargs)
    v = torch.rand(N, **kwargs)
    w = torch.rand(N, **kwargs)
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

def projection_matrix(znear, zfar, fovx, fovy, device="cpu"):
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

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    extra_points: int = 0
    """number of extra points to add to the model in addition to sfm points, randomly distributed"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 4
    """maximum degree of spherical harmonics to use"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="off")
    """camera optimizer config"""


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

    def populate_modules(self):
        if self.seed_pts is not None and not self.config.random_init:
            extra_means = (torch.rand((self.config.extra_points, 3)) - 0.5) * 10
            self.means = torch.nn.Parameter(torch.cat([self.seed_pts[0],extra_means])) # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((500000, 3)) - 0.5) * 15)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)/5))
        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if self.seed_pts is not None and not self.config.random_init:
            fused_color = RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().cuda()
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
            #concat on extra_points amount of random ones at the end
            extra_shs = torch.rand((self.config.extra_points, dim_sh, 3)).float().cuda()
            extra_shs[:,1:,:] = 0.0#zero out the higher freq
            shs = torch.cat([shs, extra_shs])
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
        
        self.crop_box: Optional[OrientedBox] = None
        self.back_color = torch.zeros(3)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
