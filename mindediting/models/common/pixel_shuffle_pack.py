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

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as ops

from ...utils.init_weights import default_init_weights
from .custom_depth_to_space import CusDepthToSpace
from .space_to_depth import DepthToSpace, PixelShuffle


class PixelShufflePack(nn.Cell):
    """Pixel Shuffle upsample layer.
    Args:
        in_channels [int]: Number of input channels.
        out_channels [int]: Number of output channels.
        scale_factor [int]: Upsample ratio.
        upsample_kernel [int]: Kernel size of Conv layer to expand channels.
        blocks_last [bool]: If True consider upsample blocks as last sub-dimension of channels:
                (c_up) -> (c, scale_factor, scale_factor). Otherwise (c_up) -> (scale_factor, scale_factor, c).
        has_bias [bool]: Whether upsample convolution should add bias or not.

    Returns:
        Upsampled feature map.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor,
        upsample_kernel,
        blocks_last=True,
        has_bias=True,
        custom_depth_to_space=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2,
            pad_mode="pad",
            has_bias=has_bias,
        )
        if blocks_last:
            self.pixel_shuffle = PixelShuffle(scale_factor)
        else:
            device = ms.get_context("device_target")
            if device == "Ascend":
                if custom_depth_to_space and scale_factor == 2:
                    print("Using custom DepthToSpace impl for Ascend.")
                    self.pixel_shuffle = CusDepthToSpace()
                else:
                    self.pixel_shuffle = DepthToSpace(scale_factor)
            else:
                self.pixel_shuffle = ops.DepthToSpace(scale_factor)
        self.init_weights()

    def init_weights(self):
        default_init_weights(self.upsample_conv, 1)

    def construct(self, x):
        """Forward function for PixelShufflePack.
        Args:
            x [Tensor]: Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)
        return x
