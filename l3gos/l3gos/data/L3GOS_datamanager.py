# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Datamanager.
"""

from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union,Generic

import torch
import yaml
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from rich.progress import Console
from l3gos.data.L3GOS_dataset import L3GOSDataset
from l3gos.data.L3GOS_dataloader import L3GOSDataloader

CONSOLE = Console(width=120)

# from lerf.data.utils.dino_dataloader import DinoDataloader
# from lerf.data.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader
from functools import cached_property
# from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig,TDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager, TDataset
import torch.multiprocessing as mp
from l3gos.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder


@dataclass
class L3GOSDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: L3GOSDataManager)
    patch_tile_size_range: Tuple[int, int] = (0.1, 0.5)
    patch_tile_size_res: int = 6
    # patch_tile_size_range: Tuple[int, int] = (0.07, 0.3)
    # patch_tile_size_res: int = 5
    patch_stride_scaler: float = 0.5

class L3GOSDataManager(FullImageDatamanager, Generic[TDataset]):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: L3GOSDataManagerConfig

    def __init__(
        self,
        config: L3GOSDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        network: BaseImageEncoderConfig = BaseImageEncoderConfig(),
        clip_out_queue: Optional[mp.Queue] = None,
        dino_out_queue: Optional[mp.Queue] = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        self.network = network
        # self.clip_out_queue = clip_out_queue
        # self.dino_out_queue = dino_out_queue
        h = self.train_dataparser_outputs.metadata['image_height']
        w = self.train_dataparser_outputs.metadata['image_width']
        # self.dino_dataloader = DinoDataloader(
        #     image_list=[],
        #     device='cuda:1',
        #     cfg={"image_shape": [h,w]},
        #     cache_path=Path("cache"),
        #     out_queue= self.dino_out_queue,
        # )
        # self.dino_dataloader.start()
        # self.dino_dataloader.device= 'cuda:0'
        # self.clip_interpolator = PyramidEmbeddingDataloader(
        #     image_list=[],
        #     device='cuda:1',
        #     network=self.network,
        #     cfg={
        #         "tile_size_range": self.config.patch_tile_size_range,
        #         "tile_size_res": self.config.patch_tile_size_res,
        #         "stride_scaler": 0.5,
        #         "image_shape": [h,w],
        #     },
        #     cache_path= Path("cache"),
        #     out_queue= self.clip_out_queue,
        # )
        # self.clip_interpolator.start()
        # self.clip_interpolator.device='cuda:0'
        # self.clip_interpolator.create(None, self.network.setup())

        torch.cuda.empty_cache()

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        return L3GOSDataset
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = L3GOSDataloader(self.train_dataset)
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        raise NotImplementedError
        # self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        # self.train_camera_optimizer = self.config.camera_optimizer.setup(
        #     num_cameras=self.train_dataset.cameras.size, device=self.device
        # )
        # self.train_ray_generator = RayGenerator(
        #     self.train_dataset.cameras.to(self.device),
        #     self.train_camera_optimizer,
        # )

    def add_image(self, img:torch.tensor, cam: Cameras):
        """
        Adds a new image to the datamanager
        1. add the actual image data
        2. make sure pixel sampling works on that                        (Should work because we override the __getitem__ function in lll_dataset)
        3. add lerf dino+clip features                                   (Justin)
        4. reset camera param for optimization                           (I think we should do this in trainer on the image callback)
        5. make sure we set the mask for the image we just added         (We should handle masks in the pipeline because adding one image requires adding a bunch of masks)
        """
        # ----------------- Handling the lerf features ----------------
        raise NotImplementedError
        # self.clip_interpolator.add_images(img.unsqueeze(0))
        # self.dino_dataloader.add_images(img.unsqueeze(0))


        # ----------------- Handling the IMAGE ----------------
        # self.train_dataset.add_image(img,cam)
        # self.train_ray_generator.cameras = self.train_dataset.cameras.to(self.device)

    def process_image(self, img:torch.tensor, cam: Cameras, clip, dino):
        # ----------------- Handling the IMAGE ----------------
        raise NotImplementedError
        # self.train_dataset.add_image(img,cam)
        # self.train_ray_generator.cameras = self.train_dataset.cameras.to(self.device)
        # dino = dino.to(self.device)
        # for i, tr in enumerate(self.clip_interpolator.tile_sizes):
        #     clip[i] = clip[i].to(self.device)
        #     if self.clip_interpolator.data_dict[i].data is not None:
        #         self.clip_interpolator.data_dict[i].data = torch.cat([self.clip_interpolator.data_dict[i].data, clip[i]])
        #     else:
        #         self.clip_interpolator.data_dict[i].data = clip[i]
        # if self.dino_dataloader.data is None:
        #     self.dino_dataloader.data = dino
        # else:
        #     self.dino_dataloader.data = torch.cat([self.dino_dataloader.data, dino], dim=0)



    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        raise NotImplementedError
        # self.train_count += 1
        # image_batch = next(self.iter_train_image_dataloader)
        # assert self.train_pixel_sampler is not None
        # batch = self.train_pixel_sampler.sample(image_batch)
        # ray_indices = batch["indices"]
        # ray_bundle = self.train_ray_generator(ray_indices)
        # batch["clip"], clip_scale = self.clip_interpolator(ray_indices)
        # batch["dino"] = self.dino_dataloader(ray_indices)
        # ray_bundle.metadata["clip_scales"] = clip_scale
        # # assume all cameras have the same focal length and image width
        # ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        # ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        # ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
        # ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
        # return ray_bundle, batch
