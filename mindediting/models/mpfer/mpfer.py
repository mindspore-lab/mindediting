# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import sys

import mindspore as ms
import mindspore.nn as nn

from mindediting.models.common.space_to_depth import PixelShuffle

from . import geometry_ms


class MPFER(nn.Cell):
    def __init__(self, params: dict = {}):
        super(MPFER, self).__init__()

        self.ref_height: int = params["ref_height"]
        self.ref_width: int = params["ref_width"]
        self.num_inputs: int = params["num_inputs"]
        self.num_targets: int = params["num_targets"]
        self.num_channels: int = params["num_channels"]
        self.in_channels: int = params["in_channels"]
        self.out_channels: int = params["out_channels"]
        self.num_planes: int = params["num_planes_far"] + params["num_planes_near"]
        self.num_filters: int = params["num_filters"]
        self.upfactor: float = params["upfactor"]
        self.split_mpf: bool = params["split_mpf"]
        self.use_input_images: bool = params["use_input_images"]
        self.global_skip: bool = params["global_skip"]

        self.C_in_encoder = self.num_channels * self.num_inputs
        self.C_out_encoder = self.num_channels * self.num_targets if self.split_mpf else self.num_channels
        self.C_in1_renderer = self.num_channels * self.num_planes
        self.C_in2_renderer = self.num_filters + self.in_channels if self.use_input_images else self.num_filters
        self.C_out_renderer = self.out_channels

        self.tail = nn.Conv2d(self.in_channels, self.num_channels, 3, has_bias=True)
        self.encoder = Encoder(self.num_filters, self.C_in_encoder, self.C_out_encoder)
        self.renderer = Renderer(
            self.num_filters,
            self.C_in1_renderer,
            self.C_in2_renderer,
            self.C_out_renderer,
            self.use_input_images,
            self.global_skip,
        )

    def construct(self, batch):

        x = batch["input_imgs"]

        poses = batch["poses"]
        intrinsics = batch["intrinsics"]
        ref_pose = batch["ref_pose"]
        ref_intrinsics = batch["ref_intrinsics"]
        depths = batch["depths"]
        corners = batch["corners"]

        hom_fwd = geometry_ms.ms_get_homographies(
            poses, intrinsics, ref_pose, ref_intrinsics, depths, "Spaces"
        )  # B, D, V, 3, 3
        hom_bwd = ms.ops.MatrixInverse()(hom_fwd)
        hom_fwd = hom_fwd.swapaxes(1, 2)

        # hom_bwd = batch["target_hom_bwd"]              # B, V, D, 3, 3

        # Tail
        B, V, C, h, w = x.shape
        x = x.view(B * V, C, h, w)
        x = self.tail(x)
        x = x.view(B, V, -1, h, w)

        x_ = []
        for d in range(self.num_planes):
            # Forward warp
            coords = geometry_ms.ms_pixel_center_grid(corners, self.upfactor)  # H, W, 2
            coords = geometry_ms.ms_apply_homography(hom_fwd[:, d : d + 1, ...], coords, corners)  # B, 1, V, H, W, 2
            x_d = ms.ops.ExpandDims()(x, 1)  # B, 1, V, C, h, w
            x_d = geometry_ms.ms_sample_image(x_d, coords)  # B, 1, V, C, H, W

            # Encoder
            x_d = self.encoder(x_d)  # B, 1, C, H, W
            x_.append(x_d)
        x = ms.ops.Concat(axis=1)(x_)  # B, D, C, H, W

        B, D, C, H, W = x.shape
        _, V, _, _, _ = hom_bwd.shape

        x_ = []
        for v in range(V):
            if self.split_mpf:  # split mpf
                x = x.view(B, D, V, -1, H, W)  # B, D, V, C, H, W
                x_v = x[:, :, v : v + 1, ...]  # B, D, 1, C, H, W
                x_v = x_v.swapaxes(1, 2)  # B, 1, D, C, H, W
            else:  # duplicate mpf
                x_v = ms.ops.ExpandDims()(x, 1)  # B, 1, D, C, H, W

            # Backward warp
            coords = geometry_ms.ms_pixel_center_grid(corners, 1.0)  # h, w, 2
            coords = geometry_ms.ms_apply_homography(hom_bwd[:, v : v + 1, ...], coords, corners)  # B, 1, D, h, w, 2
            x_v = geometry_ms.ms_sample_image(x_v, coords)  # B, 1, D, C, h, w

            # Renderer
            imgs = batch["input_imgs"][:, v : v + 1, ...]
            x_v = self.renderer(imgs, x_v)  # B, 1, C, h, w
            x_.append(x_v)

        x = ms.ops.Concat(axis=1)(x_)

        return x


class Encoder(nn.Cell):
    def __init__(self, num_filters, C_in, C_out):
        super(Encoder, self).__init__()

        self.tail = nn.SequentialCell(nn.Conv2d(C_in, num_filters, 3, has_bias=True), nn.ReLU())
        self.unet = Unet(num_filters)
        self.head = nn.Conv2d(num_filters, C_out, 3, has_bias=True)

    def construct(self, x):
        B, D, V, C, H, W = x.shape
        x = x.view(B * D, V * C, H, W)

        x = self.tail(x)
        x = self.unet(x)
        x = self.head(x)

        x = x.view(B, D, -1, H, W)

        return x


class Renderer(nn.Cell):
    def __init__(self, num_filters, C_in1, C_in2, C_out, use_input_images, global_skip):
        super(Renderer, self).__init__()

        self.use_input_images = use_input_images
        self.global_skip = global_skip
        self.C_out = C_out

        self.fusion = nn.SequentialCell(
            nn.Conv2d(C_in1, 2 * num_filters, 1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, 3, has_bias=True, group=2 * num_filters),
            nn.ReLU(),
            nn.Conv2d(2 * num_filters, num_filters, 1, has_bias=True),
        )
        self.tail = nn.SequentialCell(nn.Conv2d(C_in2, num_filters, 3, has_bias=True), nn.ReLU())
        self.unet = Unet(num_filters)
        self.head = nn.Conv2d(num_filters, C_out, 3, has_bias=True)

    def construct(self, imgs, x):

        B, V, D, C, h, w = x.shape
        x = x.reshape(B * V, D * C, h, w)

        x = self.fusion(x)

        if self.use_input_images:
            imgs = imgs.reshape(B * V, -1, h, w)
            x = ms.ops.Concat(axis=1)([imgs, x])

        x = self.tail(x)
        x = self.unet(x)
        x = self.head(x)

        if self.global_skip:
            x = x + imgs[:, : self.C_out, ...]

        return x.view(B, V, -1, h, w)


class Unet(nn.Cell):
    def __init__(self, num_filters):
        super(Unet, self).__init__()

        self.Level1 = nn.SequentialCell(
            nn.Conv2d(num_filters, num_filters, 3, has_bias=True),
            nn.ReLU(),
            # nn.Conv2d(num_filters, num_filters, 3, has_bias=True),
            # nn.ReLU(),
        )

        self.Level2 = nn.SequentialCell(
            nn.Conv2d(num_filters, 2 * num_filters, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, 3, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, 3, has_bias=True),
            nn.ReLU(),
        )

        self.Level3 = nn.SequentialCell(
            nn.Conv2d(2 * num_filters, 4 * num_filters, 3, stride=2, pad_mode="pad", padding=1, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(4 * num_filters, 4 * num_filters, 3, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(4 * num_filters, 4 * num_filters, 3, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(4 * num_filters, 8 * num_filters, 3, has_bias=True),
            nn.ReLU(),
        )

        self.Level4 = nn.SequentialCell(
            nn.Conv2d(2 * num_filters, 2 * num_filters, 3, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, 3, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(2 * num_filters, 4 * num_filters, 3, has_bias=True),
            nn.ReLU(),
        )

        self.Level5 = nn.SequentialCell(
            nn.Conv2d(num_filters, num_filters, 3, has_bias=True),
            nn.ReLU(),
            # nn.Conv2d(num_filters, num_filters, 3, has_bias=True),
        )

        # self.PixelShuffle = PixelShuffle(scale=2)
        self.PixelShuffle = PixelShuffle(upscale_factor=2)

    def construct(self, x):

        x = self.Level1(x)
        residual1 = x
        x = self.Level2(x)
        residual2 = x
        x = self.Level3(x)
        x = self.PixelShuffle(x)
        x += residual2
        x = self.Level4(x)
        x = self.PixelShuffle(x)
        x += residual1
        x = self.Level5(x)

        return x
