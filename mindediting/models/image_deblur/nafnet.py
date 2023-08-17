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
import mindspore.ops as ops
from mindspore.common.initializer import initializer

from mindediting.models.common.layer_norm import LayerNorm2dDefault
from mindediting.models.common.space_to_depth import PixelShuffle
from mindediting.utils.utils import is_ascend


class SimpleGate(nn.Cell):
    def __init__(self):
        super().__init__()
        self.split = ops.Split(axis=1, output_num=2)

    def construct(self, x):
        x1, x2 = self.split(x)
        return x1 * x2


class NAFBlock(nn.Cell):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        self.is_ascend = is_ascend()

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, stride=1, group=1, has_bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, stride=1, group=dw_channel, has_bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, stride=1, group=1, has_bias=True)

        # Simplified Channel Attention
        if self.is_ascend:
            self.pool = nn.AvgPool2d(1)
        else:
            self.pool = ops.AdaptiveAvgPool2D(1)
        self.sca = nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, stride=1, group=1, has_bias=True)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, stride=1, group=1, has_bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, stride=1, group=1, has_bias=True)

        self.norm1 = LayerNorm2dDefault(c)
        self.norm2 = LayerNorm2dDefault(c)

        self.beta = ms.Parameter(initializer("zeros", (c, 1, 1)), name="beta", requires_grad=True)
        self.gamma = ms.Parameter(initializer("zeros", (c, 1, 1)), name="gamma", requires_grad=True)

    def construct(self, x):
        input_x = x

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(self.pool(x))
        x = self.conv3(x)

        y = input_x + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        return y + x * self.gamma


class NAFNet(nn.Cell):
    """
    Simple Baselines for Image Restoration

    @article{chen2022simple,
      title={Simple Baselines for Image Restoration},
      author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
      journal={arXiv preprint arXiv:2204.04676},
      year={2022}
    }
    """

    # Params corresponding to default width 32 configuration
    def __init__(
        self,
        img_channels=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
    ):
        super().__init__()

        self.intro = nn.Conv2d(img_channels, width, kernel_size=3, stride=1, group=1, has_bias=True)
        self.ending = nn.Conv2d(width, img_channels, kernel_size=3, stride=1, group=1, has_bias=True)

        encoders = []
        downs = []
        chan = width
        for num in enc_blk_nums:
            encoders.append(nn.SequentialCell([NAFBlock(chan) for _ in range(num)]))
            downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2, has_bias=True))
            chan = chan * 2
        self.encoders = nn.SequentialCell(encoders)
        self.downs = nn.SequentialCell(downs)

        self.middle_blks = nn.SequentialCell([NAFBlock(chan) for _ in range(middle_blk_num)])

        self.shuffle = PixelShuffle(2)
        decoders = []
        ups = []
        for num in dec_blk_nums:
            ups.append(nn.Conv2d(chan, chan * 2, kernel_size=1, stride=1, has_bias=False))
            chan = chan // 2
            decoders.append(nn.SequentialCell([NAFBlock(chan) for _ in range(num)]))
        self.ups = nn.SequentialCell(ups)
        self.decoders = nn.SequentialCell(decoders)

        self.padder_size = 2 ** len(self.encoders)

    def construct(self, inp):
        h, w = inp.shape[-2:]
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders.cell_list, self.downs.cell_list):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders.cell_list, self.ups.cell_list, encs[::-1]):
            x = self.shuffle(up(x))
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x[:, :, :h, :w] + inp
        return x

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = ms.numpy.pad(x, ((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w)))
        return x


if __name__ == "__main__":
    _b, _c, _h, _w = 2, 3, 24, 24
    _x = ops.normal(shape=(_b, _c, _h, _w), mean=ms.Tensor(0.0), stddev=ms.Tensor(1.0))

    net = NAFNet(img_channels=_c)

    _y = net(_x)
    print(_x.shape, _y.shape)
