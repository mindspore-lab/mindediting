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

from mindediting.models.common.pixel_shuffle_pack import PixelShufflePack


class UNet(nn.Cell):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        cond_dim=32,
        num_block=8,
        scale=4,
        bias=True,
        use_attn=False,
        res=True,
        up_input=False,
    ):
        super().__init__()
        dims = [3] + [dim * d for d in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.scale = scale
        self.bias = bias
        self.use_attn = use_attn
        self.res = res
        self.up_input = up_input
        groups = 0

        self.cond_proj = nn.Conv2dTranspose(
            cond_dim * ((num_block + 1) // 3), dim, scale * 2 - 1, scale, pad_mode="same", has_bias=bias
        )

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.SequentialCell(Dense(dim, dim * 4), Mish(), Dense(dim * 4, dim))

        self.downs = nn.SequentialCell([])
        self.ups = nn.SequentialCell([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            res_blocks = [
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups),
            ]
            if ind < (num_resolutions - 1):
                res_blocks.append(DownSample(dim_out, 2))
            else:
                res_blocks.append(DownSample(dim_out, 1))

            self.downs.append(nn.SequentialCell(res_blocks))
        self.downs_list = []
        for cell in self.downs.cell_list:
            self.downs_list.append([c for c in cell.cell_list])

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            res_blocks = [
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups),
            ]
            if ind < (num_resolutions - 1):
                res_blocks.append(UpSample(dim_in, 2))

            self.ups.append(nn.SequentialCell(res_blocks))

        self.ups_list = []
        for cell in self.ups.cell_list:
            self.ups_list.append([c for c in cell.cell_list])

        self.final_conv = nn.SequentialCell(Block(dim, dim, groups=groups), nn.Conv2d(dim, out_dim, 1, has_bias=bias))

        if res and up_input:
            self.up_proj = nn.Conv2d(3, dim, 3, pad_mode="same", has_bias=bias)

    def construct(self, x, time, cond, img_lr_up):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        cond = self.cond_proj(ms.ops.concat(cond[2::3], axis=1))

        for i, (resnet, resnet2, downsample) in enumerate(self.downs_list):
            x = resnet(x, t)
            x = resnet2(x, t)
            if i == 0:
                x = x + cond
                if self.res and self.up_input:
                    x = x + self.up_proj(img_lr_up)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for i, (resnet, resnet2, upsample) in enumerate(self.ups_list):
            x = ms.ops.concat([x, h[-1 - i]], axis=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)


class Dense(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, pad_mode="same", has_bias=True)

    def construct(self, x):
        b, c = x.shape
        x = self.conv(x.view(b, c, 1, 1))
        return x.view(b, -1)


class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SinusoidalPosEmb(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = ms.ops.log(ms.Tensor(10000, dtype=ms.float32)) / (half_dim - 1)
        self.emb = ms.ops.exp(ms.numpy.arange(half_dim) * -emb)

    def construct(self, x):
        emb = x[:, None] * self.emb[None, :]
        emb = ms.ops.concat((ms.ops.sin(emb), ms.ops.cos(emb)), axis=-1)
        return emb


class Mish(nn.Cell):
    def __init__(self):
        super().__init__()
        self.softplus = ms.ops.Softplus()

    def construct(self, x):
        return x * ms.ops.tanh(self.softplus(x))


class ReZero(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = ms.Parameter(ms.ops.zeros(1))

    def construct(self, x):
        return self.fn(x) * self.g


class Block(nn.Cell):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        if groups == 0:
            self.block = nn.SequentialCell(nn.Conv2d(dim, dim_out, 3, has_bias=True), Mish())
        else:
            self.block = nn.SequentialCell(
                nn.Conv2d(dim, dim_out, 3, has_bias=True), nn.GroupNorm(groups, dim_out), Mish()
            )

    def construct(self, x):
        return self.block(x)


class ResnetBlock(nn.Cell):
    def __init__(self, dim, dim_out, *, time_emb_dim=0, groups=8):
        super().__init__()
        if time_emb_dim > 0:
            self.mlp = nn.SequentialCell(Mish(), nn.Dense(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1, has_bias=True)

    def construct(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        if time_emb is not None:
            h += self.mlp(time_emb)[:, :, None, None]
        if cond is not None:
            h += cond
        h = self.block2(h)
        return h + self.res_conv(x)


class UpSample(nn.Cell):
    def __init__(self, dim, scale):
        super().__init__()
        if scale != 2:
            raise ValueError(f"Supported only scale = 2, but got {scale}")
        self.scale = scale
        self.pixel_shuffle_pack = PixelShufflePack(dim, dim, scale, 3)

    def construct(self, x):
        h, w = x.shape[2:]
        if h % self.scale != 0:
            x = ms.numpy.pad(x, ((0, 0), (0, 0), (0, 1), (0, 0)), constant_values=0)
        if w % self.scale != 0:
            x = ms.numpy.pad(x, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=0)
        y = self.pixel_shuffle_pack(x)
        return y[:, :, : h * self.scale, : w * self.scale]


class DownSample(nn.Cell):
    def __init__(self, dim, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride, has_bias=False)

    def construct(self, x):
        return self.conv(x)
