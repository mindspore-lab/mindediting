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
from mindediting.models.common.tunable_resblock import TunableResBlock


class PixelShuffle(nn.Cell):
    """
    mindspore.ops.DepthToSpace is not consistent with torch.nn.PixelShuffle
    """

    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.scale = scale

    def construct(self, x, *args, **kwargs):
        s = self.scale
        b, c, h, w = x.shape

        y = x.reshape(b, c // s**2, s, s, h, w)
        y = ops.transpose(y, (0, 1, 4, 2, 5, 3))
        y = y.reshape(b, c // s**2, h * s, w * s)
        return y


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x, *args, **kwargs):
        return x


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=1.0, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1, has_bias=True)
        std = ms.Tensor(rgb_std)
        self.weight.set_data(ops.eye(3, 3, ms.float32).view(3, 3, 1, 1) / std.view(3, 1, 1, 1))
        self.bias.set_data(sign * rgb_range * ms.Tensor(rgb_mean) / std)
        for p in self.get_parameters():
            p.requires_grad = False


class TunableEDSR(nn.Cell):
    def __init__(self, img_channels=3, num_blocks=16, num_channels=64, scale=4, num_params=1, mode="mlp"):
        super().__init__()

        self.scale = scale

        self.sub_mean = MeanShift(sign=-1)

        self.head = TunableConv2d(
            img_channels, num_channels, kernel_size=3, has_bias=True, num_params=num_params, mode=mode
        )
        self.act = nn.ReLU()

        body = []
        for _ in range(num_blocks):
            body.append(TunableResBlock(num_channels, kernel_size=3, has_bias=True, num_params=num_params, mode=mode))
        self.body = TunableSequentialCell(body)

        upsample = []
        if self.scale == 3:
            upsample.append(
                TunableConv2d(
                    num_channels, num_channels * 9, kernel_size=3, has_bias=True, num_params=num_params, mode=mode
                )
            )
            upsample.append(PixelShuffle(self.scale))
        elif self.scale > 1 and (self.scale & (self.scale - 1) == 0):
            num_scales = int(math.log2(self.scale))
            for i in range(num_scales):
                upsample.append(
                    TunableConv2d(
                        num_channels, num_channels * 4, kernel_size=3, has_bias=True, num_params=num_params, mode=mode
                    )
                )
                upsample.append(PixelShuffle(2))
        elif self.scale == 1:
            upsample.append(Identity())
        else:
            raise ValueError()
        self.upsample = TunableSequentialCell(upsample)

        self.tail = TunableConv2d(
            num_channels, img_channels, kernel_size=3, has_bias=True, num_params=num_params, mode=mode
        )

        self.add_mean = MeanShift(sign=1)

    def construct(self, x, px):

        x = self.sub_mean(x)

        s = self.act(self.head(x, px))
        y = self.body(s, px)
        y = self.upsample(y + s, px)
        y = self.tail(y, px)

        y = self.add_mean(y)

        return y


class TunableEDSR_post(nn.Cell):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        px = np.expand_dims(params, axis=0).astype(np.float32)
        self.px = ms.Tensor(px)

    def construct(self, x):
        return self.model(x, self.px)


if __name__ == "__main__":
    n = 2
    _b, _c, _h, _w = 2, 3, 24, 24
    _x = ops.normal(shape=(_b, _c, _h, _w), mean=ms.Tensor(0.0), stddev=ms.Tensor(1.0))
    _px = ops.uniform(shape=(_b, n), minval=ms.Tensor(0.0), maxval=ms.Tensor(1.0))

    net = TunableEDSR(img_channels=_c, scale=1, num_params=n)

    _y = net(_x, _px)
    print(_x.shape, _y.shape)

    ms.save_checkpoint(net, "checkpoints/t_edsr_ms.ckpt")
