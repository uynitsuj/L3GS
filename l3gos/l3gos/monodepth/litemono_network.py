from __future__ import absolute_import, division, print_function
from dataclasses import dataclass, field
from typing import Tuple, Type

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import l3gos.monodepth.networks as networks
from l3gos.monodepth.layers import disp_to_depth
import cv2
import heapq
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision
import matplotlib.pyplot as plt
from nerfstudio.viewer_beta.viewer_elements import ViewerText


@dataclass
class LiteMonoNetworkConfig():
    _target: Type = field(default_factory=lambda: LiteMonoNetwork)
    # clip_model_type: str = "ViT-B-16"
    # clip_model_pretrained: str = "laion2b_s34b_b88k"
    # clip_n_dims: int = 512
    # negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    dirname = os.path.dirname(__file__)
    encoder_path = os.path.join(dirname, "lite-mono_640x192", "encoder.pth")
    decoder_path = os.path.join(dirname, "lite-mono_640x192", "depth.pth")
    device: str = 'cuda'


class LiteMonoNetwork():

    config: LiteMonoNetworkConfig

    def __init__(self, config:LiteMonoNetworkConfig):
        self.config = config
        self.encoder_dict = torch.load(self.config.encoder_path)
        self.decoder_dict = torch.load(self.config.decoder_path)
        self.feed_height = self.encoder_dict['height']
        self.feed_width = self.encoder_dict['width']
        self.encoder = networks.LiteMono(model="lite-mono",
                                    height=self.feed_height,
                                    width=self.feed_width)
        self.model_dict = self.encoder.state_dict()
        self.encoder.load_state_dict({k: v for k, v in self.encoder_dict.items() if k in self.model_dict})

        self.encoder.to(self.config.device)
        self.encoder.eval()

        self.depth_decoder = networks.DepthDecoder(self.encoder.num_ch_enc, scales=range(3))
        self.depth_model_dict = self.depth_decoder.state_dict()
        self.depth_decoder.load_state_dict({k: v for k, v in self.decoder_dict.items() if k in self.depth_model_dict})

        self.depth_decoder.to(self.config.device)
        self.depth_decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    def get_depth(self, image):
        with torch.no_grad():
        # for idx, image_path in enumerate(paths):

            # if image_path.endswith("_disp.jpg"):
            #     # don't try to predict disparity for a disparity image!
            #     continue

            # Load image and preprocess
            # input_image = pil.open(image_path).convert('RGB')

            original_width, original_height = image.size
            image = image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            image = transforms.ToTensor()(image).unsqueeze(0)

            # PREDICTION
            image = image.to(self.config.device)
            features = self.encoder(image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            # output_name = os.path.splitext(os.path.basename(image_path))[0]
            # output_name = os.path.splitext(image_path)[0].split('/')[-1]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            plt.imshow(colormapped_im)
            plt.show()
            # im = pil.fromarray(colormapped_im)
        return None
