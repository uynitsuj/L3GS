"""
L3GOS configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig
from l3gos.L3GOS_trainer import TrainerConfig
from l3gos.L3GOS_pipeline import L3GOSPipelineConfig
from l3gos.data.L3GOS_datamanager import L3GOSDataManagerConfig
from l3gos.data.L3GOS_dataparser import L3GOSDataParserConfig


l3gos_method = MethodSpecification(
    config = TrainerConfig(
        method_name="l3gos",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=2000,
        steps_per_eval_all_images=100000, 
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps = {'camera_opt': 100,'color':10,'shs':10},
        pipeline=L3GOSPipelineConfig(
            datamanager=L3GOSDataManagerConfig(
                dataparser=L3GOSDataParserConfig(),
            ),
            model=GaussianSplattingModelConfig(),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "color": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=30000,
                ),
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-3, max_steps=30000)
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer_beta",
    ),
    description="Base config for Lifelong Gaussian Splatting",
)
