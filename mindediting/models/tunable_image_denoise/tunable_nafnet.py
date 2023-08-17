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
import numpy as np
from mindspore.common.initializer import initializer

from mindediting.models.common.tunable_conv import TunableConv2d, TunableParameter, TunableSequentialCell
from mindediting.utils.utils import is_ascend


class LayerNorm2d(nn.Cell):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = ms.Parameter(initializer("ones", (1, channels, 1, 1)), name="weight", requires_grad=True)
        self.bias = ms.Parameter(initializer("ones", (1, channels, 1, 1)), name="bias", requires_grad=True)
        self.eps = eps

    def construct(self, x):
        mu = x.mean(axis=1, keep_dims=True)
        var = (x - mu).pow(2).mean(axis=1, keep_dims=True)
        y = (x - mu) / (var + self.eps).sqrt()
        y = self.weight * y + self.bias
        return y


class SimpleGate(nn.Cell):
    def __init__(self):
        super(SimpleGate, self).__init__()
        self.split = ops.Split(axis=1, output_num=2)

    def construct(self, x, *args, **kwargs):
        x1, x2 = self.split(x)
        return x1 * x2


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


class TunableNAFBlock(nn.Cell):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, num_params=1, mode="mlp"):
        super().__init__()
        self.is_ascend = is_ascend()

        dw_channel = c * DW_Expand
        self.conv1 = TunableConv2d(
            c, dw_channel, kernel_size=1, stride=1, group=1, has_bias=True, num_params=num_params, mode=mode
        )
        self.conv2 = TunableConv2d(
            dw_channel,
            dw_channel,
            kernel_size=3,
            stride=1,
            group=dw_channel,
            has_bias=True,
            num_params=num_params,
            mode=mode,
        )
        self.conv3 = TunableConv2d(
            dw_channel // 2, c, kernel_size=1, stride=1, group=1, has_bias=True, num_params=num_params, mode=mode
        )

        # Simplified Channel Attention
        if self.is_ascend:
            self.pool = nn.AvgPool2d(1)
        else:
            self.pool = ops.AdaptiveAvgPool2D(1)

        self.sca = TunableConv2d(
            dw_channel // 2,
            dw_channel // 2,
            kernel_size=1,
            stride=1,
            group=1,
            has_bias=True,
            num_params=num_params,
            mode=mode,
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = TunableConv2d(
            c, ffn_channel, kernel_size=1, stride=1, group=1, has_bias=True, num_params=num_params, mode=mode
        )
        self.conv5 = TunableConv2d(
            ffn_channel // 2, c, kernel_size=1, stride=1, group=1, has_bias=True, num_params=num_params, mode=mode
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else None
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else None

        self.beta = TunableParameter(
            initializer("zeros", (c, 1, 1)), name="beta", requires_grad=True, num_params=num_params, mode=mode
        )
        self.gamma = TunableParameter(
            initializer("zeros", (c, 1, 1)), name="gamma", requires_grad=True, num_params=num_params, mode=mode
        )

    def construct(self, x, px):
        s = x

        x = self.norm1(x)

        x = self.conv1(x, px)
        x = self.conv2(x, px)
        x = self.sg(x)
        x = x * self.sca(self.pool(x), px)
        x = self.conv3(x, px)

        if self.dropout1 is not None:
            x = self.dropout1(x)

        beta = self.beta(px)
        y = s + x * beta

        x = self.conv4(self.norm2(y), px)
        x = self.sg(x)
        x = self.conv5(x, px)

        if self.dropout2 is not None:
            x = self.dropout2(x)

        gamma = self.gamma(px)
        return y + x * gamma


class TunableNAFNet(nn.Cell):
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
        middle_blk_num=12,
        enc_blk_nums=None,
        dec_blk_nums=None,
        num_params=1,
        mode="mlp",
    ):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [2, 2, 4, 8]
        if dec_blk_nums is None:
            dec_blk_nums = [2, 2, 2, 2]
        self.intro = TunableConv2d(
            img_channels, width, kernel_size=3, stride=1, group=1, has_bias=True, num_params=num_params, mode=mode
        )
        self.ending = TunableConv2d(
            width, img_channels, kernel_size=3, stride=1, group=1, has_bias=True, num_params=num_params, mode=mode
        )

        encoders = []
        downs = []
        chan = width
        for num in enc_blk_nums:
            encoders.append(
                TunableSequentialCell([TunableNAFBlock(chan, num_params=num_params, mode=mode) for _ in range(num)])
            )
            downs.append(
                TunableConv2d(chan, 2 * chan, kernel_size=2, stride=2, has_bias=True, num_params=num_params, mode=mode)
            )
            chan = chan * 2

        self.encoders = nn.SequentialCell(encoders)
        self.downs = nn.SequentialCell(downs)

        self.middle_blks = TunableSequentialCell(
            [TunableNAFBlock(chan, num_params=num_params, mode=mode) for _ in range(middle_blk_num)]
        )

        self.shuffle = PixelShuffle(2)
        decoders = []
        ups = []
        for num in dec_blk_nums:
            ups.append(
                TunableConv2d(chan, chan * 2, kernel_size=1, stride=1, has_bias=False, num_params=num_params, mode=mode)
            )
            chan = chan // 2
            decoders.append(
                TunableSequentialCell([TunableNAFBlock(chan, num_params=num_params, mode=mode) for _ in range(num)])
            )

        self.ups = nn.SequentialCell(ups)
        self.decoders = nn.SequentialCell(decoders)

        self.padder_size = 2 ** len(self.encoders)

    def construct(self, x, px):

        h, w = x.shape[-2:]

        fea = self.check_image_size(x)

        fea = self.intro(fea, px)

        encs = []

        for encoder, down in zip(self.encoders.cell_list, self.downs.cell_list):
            fea = encoder(fea, px)
            encs.append(fea)
            fea = down(fea, px)

        fea = self.middle_blks(fea, px)

        for decoder, up, enc_skip in zip(self.decoders.cell_list, self.ups.cell_list, encs[::-1]):
            fea = self.shuffle(up(fea, px))
            fea = fea + enc_skip
            fea = decoder(fea, px)

        y = self.ending(fea, px)
        y = y[:, :, :h, :w] + x

        return y

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        pad = nn.Pad(paddings=((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w)), mode="CONSTANT")
        x = pad(x)
        return x


class TunableNAFNet_post(nn.Cell):
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

    net = TunableNAFNet(img_channels=_c, num_params=n)

    _y = net(_x, _px)
    print(_x.shape, _y.shape)
