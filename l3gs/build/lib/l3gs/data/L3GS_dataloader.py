# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

"""
Defines the ROSDataloader object that subscribes to pose and images topics,
and populates an image tensor and Cameras object with values from these topics.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""
import time
import warnings
from typing import Union

import numpy as np
import scipy.spatial.transform as transform
from rich.console import Console
import torch
from torch.utils.data.dataloader import DataLoader

from nerfstudio.process_data.colmap_utils import qvec2rotmat
import nerfstudio.utils.poses as pose_utils

from l3gs.data.L3GS_dataset import L3GSDataset
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseArray
from message_filters import TimeSynchronizer, Subscriber


CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")




class L3GSDataloader(DataLoader):
    """
    Creates batches of the dataset return type. In this case of nerfstudio this means
    that we are returning batches of full images, which then are sampled using a
    PixelSampler. For this class the image batches are progressively growing as
    more images are recieved from ROS, and stored in a pytorch tensor.

    Args:
        dataset: Dataset to sample from.
        publish_posearray: publish a PoseArray to a ROS topic that tracks the poses of the
            images that have been added to the training set.
        data_update_freq: Frequency (wall clock) that images are added to the training
            data tensors. If this value is less than the frequency of the topics to which
            this dataloader subscribes (pose and images) then this subsamples the ROS data.
            Otherwise, if the value is larger than the ROS topic rates then every pair of
            messages is added to the training bag.
        device: Device to perform computation.
    """

    dataset: L3GSDataset

    def __init__(
        self,
        dataset: L3GSDataset,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        # This is mostly a parameter placeholder, and manages the cameras
        self.dataset = dataset

        # Image meta data
        self.device = device
        self.num_images = len(self.dataset)
        self.H = self.dataset.image_height
        self.W = self.dataset.image_width
        self.n_channels = 3

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
            # "mask": self.dataset.mask_tensor,
        }

        super().__init__(dataset=dataset, **kwargs)


    # def __getitem__(self, idx):
    #     return self.dataset.__getitem__(idx)

    def _get_updated_batch(self):
        batch = {}
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[: len(self.dataset), ...]
        return batch

    def __iter__(self):
        while True:
            if not hasattr(self,'last_size') or self.last_size != len(self.dataset):
                self.batch = self._get_updated_batch()
                self.last_size = len(self.dataset)

            batch = self.batch
            yield batch