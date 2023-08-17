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

import math

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

from mindediting.models.common.tunable_conv import TunableConv2d, TunableSequentialCell
from mindediting.models.common.tunable_resblock_norm import InstanceNorm2d, TunableResBlockNorm
from mindediting.utils.utils import is_ascend


class TunableConvBlock(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation=True,
        instance_norm=True,
        pad_mode="same",
        padding=0,
        num_params=1,
        expand_params=1,
        mode="mlp",
    ):
        super(TunableConvBlock, self).__init__()
        self.is_ascend = is_ascend()
        self.conv = TunableConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            has_bias=True,
            pad_mode=pad_mode,
            padding=padding,
            num_params=num_params,
            expand_params=expand_params,
            mode=mode,
        )
        self.norm = None
        if instance_norm:
            if self.is_ascend:
                self.norm = InstanceNorm2d(out_channels, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = None
        if activation:
            self.act = nn.ReLU()

    def construct(self, x, px):
        y = self.conv(x, px)
        if self.norm is not None:
            y = self.norm(y)
        if self.act is not None:
            y = self.act(y)
        return y


class UpsampleNearest(nn.Cell):
    def __init__(self, scale_factor):
        super().__init__()
        assert scale_factor > 1 and type(scale_factor) == int
        self.scale_factor = scale_factor

    def construct(self, x, *args, **kwargs):
        h, w = x.shape[2:]
        s = self.scale_factor
        resize = ops.ResizeNearestNeighbor((s * h, s * w))
        return resize(x)


class TunableStyleNet(nn.Cell):
    def __init__(self, img_channels=3, num_scales=2, num_channels=64, num_blocks=None, num_params=1, mode="mlp"):
        super(TunableStyleNet, self).__init__()
        if num_blocks is None:
            num_blocks = [2, 6, 2]
        self.num_scales: int = num_scales

        nbe, nbl, nbd = num_blocks

        self.conv_in = TunableConvBlock(img_channels, num_channels, kernel_size=9, num_params=num_params, mode=mode)

        layers = []
        for i in range(self.num_scales):
            in_ch, out_ch = num_channels * 2**i, num_channels * 2 ** (i + 1)
            layers.append(
                TunableConvBlock(
                    in_ch, out_ch, stride=2, kernel_size=3, pad_mode="pad", padding=1, num_params=num_params, mode=mode
                )
            )
            for _ in range(1, nbe):
                layers.append(TunableConvBlock(out_ch, out_ch, kernel_size=3, num_params=num_params, mode=mode))

        for _ in range(nbl):
            layers.append(TunableResBlockNorm(out_ch, kernel_size=3, num_params=num_params, mode=mode))

        for i in reversed(range(self.num_scales)):
            in_ch, out_ch = num_channels * 2 ** (i + 1), num_channels * 2**i
            layers.append(UpsampleNearest(scale_factor=2))
            layers.append(TunableConvBlock(in_ch, out_ch, kernel_size=3, num_params=num_params, mode=mode))
            for _ in range(1, nbd):
                layers.append(TunableConvBlock(out_ch, out_ch, kernel_size=3, num_params=num_params, mode=mode))

            self.encoder_decoder = TunableSequentialCell(layers)

        self.conv_out = TunableConvBlock(
            num_channels,
            img_channels,
            kernel_size=9,
            activation=False,
            instance_norm=False,
            num_params=num_params,
            mode=mode,
        )

        assert img_channels == 3
        self.img_mean = ms.Parameter(ms.Tensor([0.485, 0.456, 0.406], ms.float32).view(1, 3, 1, 1), requires_grad=False)
        self.img_std = ms.Parameter(ms.Tensor([0.229, 0.224, 0.225], ms.float32).view(1, 3, 1, 1), requires_grad=False)

    def check_image_size(self, x):
        h, w = x.shape[2:]
        s = 2**self.num_scales
        fn = lambda n: int(math.ceil(n / s) * s - n)
        ph, pw = fn(h), fn(w)
        if ph > 0 or pw > 0:
            pad = nn.Pad(paddings=((0, 0), (0, 0), (0, ph), (0, pw)), mode="REFLECT")
            x = pad(x)
        return x

    def construct(self, x, px):

        h, w = x.shape[2:]
        x = self.check_image_size(x)

        x = (x - self.img_mean) / self.img_std

        out = self.conv_in(x, px)

        out = self.encoder_decoder(out, px)
        out = self.conv_out(out, px)

        out = out * self.img_std + self.img_mean

        return out[:, :, :h, :w]


class TunableStyleNet_post(nn.Cell):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        px = np.expand_dims(params, axis=0).astype(np.float32)
        self.px = ms.Tensor(px)

    def construct(self, x):
        return self.model(x, self.px)
