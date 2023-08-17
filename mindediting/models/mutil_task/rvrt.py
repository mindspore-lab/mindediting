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
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations import nn_ops as NN_OPS

from mindediting.utils.init_weights import HeUniform, Uniform, _calculate_fan_in_and_fan_out, initializer
from mindediting.utils.utils import cast_module

from ..common.grid_sample import GridSample2D
from ..video_super_resolution.ttvsr import FlowWarp
from .vrt import TMSA, TMSAG, SpyNet, Upsample
from .vrt import WindowAttention as WindowAttentionVRT
from .vrt import convert_offsets_from_pt_ms, is_ascend, nn_Transpose


def flatten_0_1(x):
    return x.reshape(-1, *x.shape[2:])


class WindowAttention(WindowAttentionVRT):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        input_resolution=None,
        relative_position_encoding=True,
    ):
        mut_attn = False
        super().__init__(
            dim, window_size, num_heads, qkv_bias, qk_scale, mut_attn, input_resolution, relative_position_encoding
        )

    def construct(self, x, mask=None):
        """Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C), relative_position_encoding=self.relative_position_encoding)

        # projection
        x = self.proj(x_out)

        return x


class STL(TMSA):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=(2, 8, 8),
        shift_size=(0, 0, 0),
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        relative_position_encoding=True,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            False,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            0,
            act_layer,
            norm_layer,
            input_resolution,
        )
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            input_resolution=input_resolution,
            relative_position_encoding=relative_position_encoding,
        )
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)


class STG(TMSAG):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size=[2, 8, 8],
        shift_size=None,
        mlp_ratio=2.0,
        qkv_bias=False,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        relative_position_encoding=True,
    ):

        super().__init__(
            dim,
            depth,
            num_heads,
            window_size,
            shift_size,
            False,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            0,
            norm_layer,
            input_resolution,
        )
        # build blocks
        self.blocks = nn.CellList(
            [
                STL(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    relative_position_encoding=relative_position_encoding,
                )
                for i in range(depth)
            ]
        )


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.SequentialCell(layers)


@ms.ops.constexpr
def _compute_filter_offsets(h, w, K):
    ys, xs = ms.ops.meshgrid(
        (ms.numpy.arange(0, h), ms.numpy.arange(0, w)),
        indexing="ij",
    )

    filter_offset_x = xs[None, None] + ms.numpy.tile(ms.numpy.arange(K), K)[None, :, None, None]
    filter_offset_y = ys[None, None] + ms.numpy.repeat(ms.numpy.arange(K), K)[None, :, None, None]

    return filter_offset_x, filter_offset_y


class DeformableOffsets(nn.Cell):
    def __init__(self, strides=(1, 1, 1, 1), padding=(1, 1, 1, 1), dilations=(1, 1, 1, 1), kernel_size=(3, 3)):
        super(DeformableOffsets, self).__init__()
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.modulated = True
        self.is_ascend = is_ascend()

    def construct(self, x, offsets):
        B = offsets.shape[0]
        Hout, Wout = offsets.shape[-2:]
        offsets = offsets.reshape(B, 2 * self.kernel_size[0] * self.kernel_size[1], Hout, Wout)

        # non-modulate (deform conv2 v1) is not supported in mindspore, that's why have to add masks to offsets.
        masks = ms.numpy.ones((B, 1 * self.kernel_size[0] * self.kernel_size[1], Hout, Wout), dtype=offsets.dtype)

        offsets = ms.ops.concat((offsets, masks), axis=1)

        deformable_offsets = _get_cache_prim(NN_OPS.DeformableOffsets)(
            self.strides, self.padding, self.kernel_size, self.dilations, "NCHW", 1, self.modulated
        )

        x = x.reshape(B, -1, x.shape[-2], x.shape[-1])
        cin = x.shape[1]

        # due to limited support of DeformableOffsets on Ascend, have to add padding to make channels divisible by 8.
        CHANNELS = 8
        if self.is_ascend and cin % CHANNELS != 0:
            x = ms.numpy.pad(x, ((0, 0), (0, CHANNELS - cin % CHANNELS), (0, 0), (0, 0)))

        fm_offset = deformable_offsets(x, offsets)

        if self.is_ascend and cin % CHANNELS != 0:
            fm_offset = fm_offset[:, :cin, :, :]

        return fm_offset


class DeformableOffsetsViaGridSample(nn.Cell):
    def __init__(
        self,
        strides=(1, 1, 1, 1),
        padding=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        kernel_size=(3, 3),
    ):
        super().__init__()
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.modulated = True
        self.grid_sample = GridSample2D(align_corners=True, do_reshape=False)
        self.grid_sample.interpolate.mode = "custom"

    def construct(self, x, offsets):
        kernel_offsets_num = offsets.shape[1]
        assert kernel_offsets_num == 2 * self.kernel_size[0] * self.kernel_size[1]

        _, _, h, w = x.shape
        pad = 1
        grid = self._offset2grid(offsets, pad, h, w)
        x_pad = ms.numpy.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        fm_offset = self.grid_sample(x_pad, grid)

        return fm_offset

    def _offset2grid(self, offset, p, h, w):
        out_h, out_w = offset.shape[-2:]

        filter_offset_x, filter_offset_y = _compute_filter_offsets(out_h, out_w, self.kernel_size[0])

        filter_offset_x = filter_offset_x.astype(offset.dtype)
        filter_offset_y = filter_offset_y.astype(offset.dtype)

        offset_x, offset_y = ms.ops.split(offset, output_num=2, axis=1)
        x_coord = offset_x + filter_offset_x
        y_coord = offset_y + filter_offset_y

        # Normalize coordinates to [-1, 1].
        x_coord = 2 * x_coord / (w + 2 * p - 1) - 1
        y_coord = 2 * y_coord / (h + 2 * p - 1) - 1

        K = self.kernel_size[0]
        assert self.kernel_size[0] == self.kernel_size[1]

        coord = ms.ops.stack((x_coord, y_coord), axis=-1).reshape(-1, K * K, out_h * out_w, 2)
        return coord


class GuidedDeformAttnPack(nn.Cell):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        attention_window=[3, 3],
        deformable_groups=12,
        attention_heads=12,
        clip_size=1,
        deformable_offsets_via_grid_sample=True,
        **kwargs,
    ):
        super(GuidedDeformAttnPack, self).__init__()
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = attention_window[0]
        self.kernel_w = attention_window[1]
        self.attn_size = self.kernel_h * self.kernel_w
        self.deformable_groups = deformable_groups
        self.attention_heads = attention_heads
        self.clip_size = clip_size
        self.stride = 1
        self.padding = self.kernel_h // 2
        self.dilation = 1

        if deformable_offsets_via_grid_sample:
            self.get_columns = self.get_columns_for_gs
            self.deformable_offsets = DeformableOffsetsViaGridSample()
        else:
            self.get_columns = self.get_columns_for_do
            self.deformable_offsets = DeformableOffsets()

        self.conv_offset = nn.SequentialCell(
            [
                nn.Conv3d(
                    has_bias=True,
                    in_channels=self.in_channels * (1 + self.clip_size) + self.clip_size * 2,
                    out_channels=64,
                    kernel_size=(1, 1, 1),
                    pad_mode="pad",
                    padding=(0, 0, 0, 0, 0, 0),
                ),
                nn.LeakyReLU(alpha=0.1),
                nn.Conv3d(
                    has_bias=True,
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(1, 3, 3),
                    pad_mode="pad",
                    padding=(0, 0, 1, 1, 1, 1),
                ),
                nn.LeakyReLU(alpha=0.1),
                nn.Conv3d(
                    has_bias=True,
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(1, 3, 3),
                    pad_mode="pad",
                    padding=(0, 0, 1, 1, 1, 1),
                ),
                nn.LeakyReLU(alpha=0.1),
                nn.Conv3d(
                    has_bias=True,
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(1, 3, 3),
                    pad_mode="pad",
                    padding=(0, 0, 1, 1, 1, 1),
                ),
                nn.LeakyReLU(alpha=0.1),
                nn.Conv3d(
                    has_bias=True,
                    in_channels=64,
                    out_channels=64,
                    kernel_size=(1, 3, 3),
                    pad_mode="pad",
                    padding=(0, 0, 1, 1, 1, 1),
                ),
                nn.LeakyReLU(alpha=0.1),
                nn.Conv3d(
                    has_bias=True,
                    in_channels=64,
                    out_channels=self.clip_size * self.deformable_groups * self.attn_size * 2,
                    kernel_size=(1, 1, 1),
                    pad_mode="pad",
                    padding=(0, 0, 0, 0, 0, 0),
                    weight_init="zeros",
                    bias_init="zeros",
                ),
            ]
        )
        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = int(self.in_channels * 2)
        self.proj_q = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                nn.Dense(self.in_channels, self.proj_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )
        self.proj_k = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                nn.Dense(self.in_channels, self.proj_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )
        self.proj_v = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                nn.Dense(self.in_channels, self.proj_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )
        self.proj = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                nn.Dense(self.proj_channels, self.in_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )
        self.mlp = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                Mlp(self.in_channels, self.in_channels * 2, self.in_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )

    def construct(self, q, k, v, v_prop_warped, flows, return_updateflow):
        offset1, offset2 = ms.ops.split(
            self.max_residue_magnitude
            * ms.ops.tanh(
                self.conv_offset(
                    ms.ops.concat(
                        [q] + v_prop_warped + [flows[0].astype(q.dtype), flows[1].astype(q.dtype)], 2
                    ).swapaxes(1, 2)
                ).swapaxes(1, 2)
            ),
            output_num=2,
            axis=2,
        )
        reversed_flow0 = ms.ops.ReverseV2([2])(flows[0])
        reversed_flow1 = ms.ops.ReverseV2([2])(flows[1])

        shape0 = reversed_flow0.shape
        shape1 = reversed_flow1.shape
        reversed_flow0 = reversed_flow0.reshape(shape0[0] * shape0[1], shape0[2], shape0[3] * shape0[4])
        reversed_flow1 = reversed_flow1.reshape(shape1[0] * shape1[1], shape1[2], shape1[3] * shape1[4])
        tiled_reversed_flow0 = ms.ops.tile(reversed_flow0, (1, offset1.shape[2] // 2, 1))
        tiled_reversed_flow1 = ms.ops.tile(reversed_flow1, (1, offset1.shape[2] // 2, 1))

        offset1 = offset1 + tiled_reversed_flow0.reshape(*shape0[:2], tiled_reversed_flow0.shape[1], *shape0[3:])
        offset2 = offset2 + tiled_reversed_flow1.reshape(*shape1[:2], tiled_reversed_flow0.shape[1], *shape1[3:])

        offset = flatten_0_1(ms.ops.concat([offset1, offset2], axis=2))

        b, t, c, h, w = offset1.shape
        q = self.proj_q(q).reshape(b * t, 1, self.proj_channels, h, w)
        kv = ms.ops.concat([self.proj_k(k), self.proj_v(v)], 2)

        v = self.deform_attn(
            q,
            kv,
            offset,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding,
            self.dilation,
            self.attention_heads,
            self.deformable_groups,
            self.clip_size,
        ).reshape(b, t, self.proj_channels, h, w)

        v = self.proj(v)
        v = v + self.mlp(v)

        if return_updateflow:
            return (
                v,
                ms.ops.ReverseV2([2])(offset1.reshape(b, t, c // 2, 2, h, w).mean(2)),
                ms.ops.ReverseV2([2])(offset2.reshape(b, t, c // 2, 2, h, w).mean(2)),
            )
        else:
            return v

    @staticmethod
    def convert_offsets_from_pt_ms(offset):
        """
        This method does conversion of offsets used in PyTorch to offsets used in MindSpore.
        PyTorch offsets shape: [B, GROUPS x Hf x Wf x 2, Hout, Wout]
        MindSpore offsets shape: [B, 2 x GROUPS x Hf x Wf, Hout, Wout]
        Where the '2' corresponds to coordinates. Moreover, order of offset coordinates in Pytorch is (y, x),
            in MindSpore: it is (x, y).
        """

        offset = offset.reshape(offset.shape[0], -1, 2, offset.shape[-2], offset.shape[-1])
        o1, o2 = ms.ops.split(offset, axis=2, output_num=2)
        offset = ms.ops.concat((o2, o1), axis=1)
        offset = offset.reshape(offset.shape[0], -1, offset.shape[-2], offset.shape[-1])
        return offset

    def get_columns_for_gs(
        self, q, kv, offset, deform_group, b, clip_size, height, width, attn_size, attn_head, attn_dim
    ):

        qb = q[b].reshape(6, 2, -1, 24).swapaxes(1, 2).expand_dims(1)  # [6, 1, spatial, 2, 24]

        columns = []
        attns = []
        for n in range(clip_size):
            cur_x = kv[b // clip_size, (n + b) % clip_size]
            cur_offset = offset[b, n]
            cur_x = cur_x.reshape(deform_group, -1, height, width)
            cur_offset = cur_offset.reshape(deform_group, attn_size * 2, height, width)
            cur_offset = self.convert_offsets_from_pt_ms(cur_offset)
            offsetted = self.deformable_offsets(cur_x, cur_offset)

            offsetted = offsetted.reshape(2, 6, 9, -1, 2, 24)

            attn = (qb * offsetted[0]).sum(-1).expand_dims(-1)  # [6, 9, spatial, 2, 1]

            columns.append(offsetted[1])
            attns.append(attn)

        a = ms.ops.concat(attns, axis=1)
        a = ms.ops.softmax(a, 1)

        b = ms.ops.concat(columns, axis=1)
        output = (a * b).sum(1)

        output = output.reshape(6, -1, 48).swapaxes(1, 2).reshape(attn_head, attn_dim, height, width)

        return output

    def get_columns_for_do(
        self, q, kv, offset, deform_group, b, clip_size, height, width, attn_size, attn_head, attn_dim
    ):
        columns = []

        for n in range(clip_size):
            cur_x = kv[b // clip_size, (n + b) % clip_size]
            cur_offset = offset[b, n]
            cur_x = cur_x.reshape(deform_group, -1, height, width)
            cur_offset = cur_offset.reshape(deform_group, attn_size * 2, height, width)
            cur_offset = convert_offsets_from_pt_ms(cur_offset)
            offsetted = self.deformable_offsets(cur_x, cur_offset)

            columns.append(offsetted)

        columns = ms.ops.stack(columns)
        columns = columns.reshape(-1, 2, attn_head, attn_dim, height, 3, width, 3)
        columns = columns.transpose(1, 2, 4, 6, 3, 0, 5, 7)
        columns = columns.reshape(2, attn_head, height * width, attn_dim, clip_size * attn_size)

        attns = ms.ops.matmul(q[b], columns[0])
        attns = ms.ops.softmax(attns, -1)  # (attn_head x (height*width) x 1 x (clip_size*attn_size))
        output = (
            ms.ops.matmul(attns, columns[1].swapaxes(2, 3)).swapaxes(1, 3).reshape(attn_head, attn_dim, height, width)
        )
        return output

    def deform_attn(
        self, q, kv, offset, kernel_h, kernel_w, stride, padding, dilation, attn_head, deform_group, clip_size
    ):
        assert stride == 1 and padding == 1 and dilation == 1
        batch = q.shape[0]
        kv_channels = kv.shape[2]
        assert kv_channels % 2 == 0
        channels = kv_channels // 2
        height = kv.shape[3]
        width = kv.shape[4]
        area = height * width
        assert channels % attn_head == 0
        attn_dim = channels // attn_head
        attn_size = kernel_h * kernel_w
        attn_scale = attn_dim ** (-0.5)

        q = (
            q.reshape(batch, 1, attn_head, attn_dim, area).transpose(0, 2, 4, 1, 3) * attn_scale
        )  # batch x attn_head x (height*width) x 1 x attn_dim
        # ms.ops.Print()(q.shape)
        offset = offset.reshape(
            batch, clip_size, offset.shape[1] // clip_size, area
        )  # batch x clip_size x (deform_groupxattn_sizex2) x (heightxwidht)

        output = []

        for b in range(batch):
            output.append(
                self.get_columns(
                    q, kv, offset, deform_group, b, clip_size, height, width, attn_size, attn_head, attn_dim
                )
            )

        output = ms.ops.stack(output)

        output = output.reshape(batch, channels, height, width)

        return output


class Mlp(nn.Cell):
    """Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)

    def construct(self, x):
        return self.fc2(self.act(self.fc1(x)))


class RSTB(nn.Cell):
    """Residual Swin Transformer Block (RSTB).

    Args:
        kwargs: Args for RSTB.
    """

    def __init__(self, **kwargs):
        super(RSTB, self).__init__()

        self.residual_group = STG(**kwargs)
        self.linear = nn.Dense(kwargs["dim"], kwargs["dim"])

    def construct(self, x):
        return x + self.linear(self.residual_group(x).swapaxes(1, 4)).swapaxes(1, 4)


class RSTBWithInputConv(nn.Cell):
    """RSTB with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        kernel_size (int): Size of kernel of the first conv.
        stride (int): Stride of the first conv.
        group (int): Group of the first conv.
        num_blocks (int): Number of residual blocks. Default: 2.
         **kwarg: Args for RSTB.
    """

    def __init__(self, in_channels=3, kernel_size=(1, 3, 3), stride=1, groups=1, num_blocks=2, **kwargs):
        super().__init__()

        main = []
        main += [
            nn_Transpose("n d c h w -> n c d h w"),
            nn.Conv3d(
                has_bias=True,
                in_channels=in_channels,
                out_channels=kwargs["dim"],
                kernel_size=kernel_size,
                stride=stride,
                pad_mode="pad",
                padding=(
                    kernel_size[0] // 2,
                    kernel_size[0] // 2,
                    kernel_size[1] // 2,
                    kernel_size[1] // 2,
                    kernel_size[2] // 2,
                    kernel_size[2] // 2,
                ),
                group=groups,
            ),
            nn_Transpose("n c d h w -> n d h w c"),
            nn.LayerNorm([kwargs["dim"]]),
            nn_Transpose("n d h w c -> n c d h w"),
        ]

        # RSTB blocks
        main.append(make_layer(RSTB, num_blocks, **kwargs))

        main += [
            nn_Transpose("n c d h w -> n d h w c"),
            nn.LayerNorm([kwargs["dim"]]),
            nn_Transpose("n d h w c -> n d c h w"),
        ]

        self.main = nn.SequentialCell(main)

    def construct(self, x):
        """
        Forward function for RSTBWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, t, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, t, out_channels, h, w)
        """
        return self.main(x)


class CellDict(nn.Cell):
    def __init__(self, d) -> None:
        super().__init__()
        for k, v in d.items():
            setattr(self, k, v)


class RVRT(nn.Cell):
    """Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT).
        A PyTorch impl of : `Recurrent Video Restoration Transformer with Guided Deformable Attention`  -
          https://arxiv.org/pdf/2205.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        clip_size (int): Size of clip in recurrent restoration transformer.
        img_size (int | tuple(int)): Size of input video. Default: [2, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (2,8,8).
        num_blocks (list[int]): Number of RSTB blocks in each stage.
        depths (list[int]): Depths of each RSTB.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        inputconv_groups (int): Group of the first convolution layer in RSTBWithInputConv. Default: [1,1,1,1,1,1]
        spynet_path (str): Pretrained SpyNet model path.
        deformable_groups (int): Number of deformable groups in deformable attention. Default: 12.
        attention_heads (int): Number of attention heads in deformable attention. Default: 12.
        attention_window (list[int]): Attention window size in aeformable attention. Default: [3, 3].
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
    """

    def __init__(
        self,
        upscale=4,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[144, 144, 144],
        num_heads=[6, 6, 6],
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        inputconv_groups=[1, 1, 1, 1, 1, 1],
        spynet_path=None,
        max_residue_magnitude=10,
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        nonblind_denoising=False,
        to_float16=False,
        deformable_offsets_via_grid_sample=True,
        relative_position_encoding=True,
    ):
        super().__init__()
        self.upscale = upscale
        self.clip_size = clip_size
        self.nonblind_denoising = nonblind_denoising

        # optical flow
        self.spynet = SpyNet(spynet_path, fast_grid_sample=deformable_offsets_via_grid_sample)
        blocks_class = RSTBWithInputConv

        # shallow feature extraction
        if self.upscale == 4:
            # video sr
            self.feat_extract = blocks_class(
                in_channels=3,
                kernel_size=(1, 3, 3),
                groups=inputconv_groups[0],
                num_blocks=num_blocks[0],
                dim=embed_dims[0],
                input_resolution=img_size,
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=[1, window_size[1], window_size[2]],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                relative_position_encoding=relative_position_encoding,
            )
        else:
            # video deblurring/denoising
            self.feat_extract = nn.SequentialCell(
                [
                    nn_Transpose("n d c h w -> n c d h w"),
                    nn.Conv3d(
                        has_bias=True,
                        in_channels=4 if self.nonblind_denoising else 3,
                        out_channels=embed_dims[0],
                        kernel_size=(1, 3, 3),
                        stride=(1, 2, 2),
                        pad_mode="pad",
                        padding=(0, 0, 1, 1, 1, 1),
                    ),
                    nn.LeakyReLU(alpha=0.1),
                    nn.Conv3d(
                        has_bias=True,
                        in_channels=embed_dims[0],
                        out_channels=embed_dims[0],
                        kernel_size=(1, 3, 3),
                        stride=(1, 2, 2),
                        pad_mode="pad",
                        padding=(0, 0, 1, 1, 1, 1),
                    ),
                    nn.LeakyReLU(alpha=0.1),
                    nn_Transpose("n c d h w -> n d c h w"),
                    blocks_class(
                        in_channels=embed_dims[0],
                        kernel_size=(1, 3, 3),
                        groups=inputconv_groups[0],
                        num_blocks=num_blocks[0],
                        dim=embed_dims[0],
                        input_resolution=img_size,
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=[1, window_size[1], window_size[2]],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        norm_layer=norm_layer,
                        relative_position_encoding=relative_position_encoding,
                    ),
                ]
            )

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        # recurrent feature refinement
        self.backbone = dict()
        self.deform_align = dict()
        self.modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(self.modules):
            # deformable attention
            self.deform_align[module] = GuidedDeformAttnPack(
                embed_dims[1],
                embed_dims[1],
                attention_window=attention_window,
                attention_heads=attention_heads,
                deformable_groups=deformable_groups,
                clip_size=clip_size,
                max_residue_magnitude=max_residue_magnitude,
                input_resolution=[img_size[1], img_size[2]],
                deformable_offsets_via_grid_sample=deformable_offsets_via_grid_sample,
            )

            # feature propagation
            self.backbone[module] = blocks_class(
                in_channels=(2 + i) * embed_dims[0],
                kernel_size=(1, 3, 3),
                groups=inputconv_groups[i + 1],
                num_blocks=num_blocks[1],
                dim=embed_dims[1],
                input_resolution=[self.clip_size, img_size[1], img_size[2]],
                depth=depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                relative_position_encoding=relative_position_encoding,
            )

        # reconstruction
        self.reconstruction = blocks_class(
            in_channels=5 * embed_dims[0],
            kernel_size=(1, 3, 3),
            groups=inputconv_groups[5],
            num_blocks=num_blocks[2],
            dim=embed_dims[2],
            input_resolution=[1, img_size[1], img_size[2]],
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=[1, window_size[1], window_size[2]],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
            relative_position_encoding=relative_position_encoding,
        )
        self.conv_before_upsampler = nn.SequentialCell(
            [
                nn.Conv3d(
                    has_bias=True,
                    in_channels=embed_dims[-1],
                    out_channels=64,
                    kernel_size=(1, 1, 1),
                    pad_mode="pad",
                    padding=(0, 0, 0, 0, 0, 0),
                ),
                nn.LeakyReLU(alpha=0.1),
            ]
        )
        self.backbone = CellDict(self.backbone)
        self.deform_align = CellDict(self.deform_align)
        self.backbone_dict = {k: getattr(self.backbone, k) for k in self.modules}
        self.deform_align_dict = {k: getattr(self.deform_align, k) for k in self.modules}
        self.upsampler = Upsample(4, 64)
        self.conv_last = nn.Conv3d(
            has_bias=True,
            in_channels=64,
            out_channels=3,
            kernel_size=(1, 3, 3),
            pad_mode="pad",
            padding=(0, 0, 1, 1, 1, 1),
        )

        self.flow_warp_bilinear = FlowWarp(
            padding_mode="zeros",
            interpolation="bilinear",
            align_corners=True,
            fast_grid_sample=deformable_offsets_via_grid_sample,
        )

        self._init_weights()

        if to_float16:
            cast_module("float16", self)

    def _init_weights(self):
        for name, cell in self.cells_and_names():
            if "spynet" not in name.lower():
                if isinstance(
                    cell,
                    (
                        nn.Conv2d,
                        nn.Conv3d,
                        nn.Dense,
                    ),
                ):
                    if hasattr(cell, "weight") and cell.weight is not None:
                        kaiming_init = HeUniform(
                            negative_slope=math.sqrt(5.0), mode="fan_in", nonlinearity="leaky_relu"
                        )
                        cell.weight = initializer(kaiming_init, cell.weight.shape, cell.weight.dtype)
                    if hasattr(cell, "bias") and cell.bias is not None:
                        fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                        bound = 1.0 / math.sqrt(fan_in)
                        cell.bias = initializer(Uniform(bound), cell.bias.shape, cell.bias.dtype)

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip([1])'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.shape
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).reshape(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip([1])
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).reshape(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.shape[1] % 2 == 0:
            lqs_1, lqs_2 = ms.ops.split(lqs, output_num=2, axis=1)
            if ms.ops.norm(lqs_1 - ms.ops.ReverseV2([1])(lqs_2), list(range(len(lqs_1.shape)))) == 0:
                self.is_mirror_extended = True

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        feats["shallow"] = ms.ops.concat(feats["shallow"], 1)
        feats["backward_1"] = ms.ops.concat(feats["backward_1"], 1)
        feats["forward_1"] = ms.ops.concat(feats["forward_1"], 1)
        feats["backward_2"] = ms.ops.concat(feats["backward_2"], 1)
        feats["forward_2"] = ms.ops.concat(feats["forward_2"], 1)

        hr = ms.ops.concat([feats[k] for k in feats], axis=2)
        hr = self.reconstruction(hr)
        hr = self.conv_last(self.upsampler(self.conv_before_upsampler(hr.swapaxes(1, 2)))).swapaxes(1, 2)

        dim0, dim1, dim2, dim3, dim4 = lqs.shape
        lqs = lqs.reshape(-1, dim2, dim3, dim4)
        resized = ms.ops.interpolate(
            lqs, sizes=(dim3 * 4, dim4 * 4), coordinate_transformation_mode="half_pixel", mode="bilinear"
        )
        resized = resized.reshape(dim0, dim1, dim2, resized.shape[-2], resized.shape[-1])

        return hr + resized

    def backward1(self, feats, flows_backward, updated_flows):
        flows = flows_backward
        direction = "backward"
        module_name = f"{direction}_1"
        feats[module_name] = []
        n, t, _, h, w = flows.shape
        flow_idx_backward = range(t, -1, -1)
        clip_idx_backward = range((t + 1) // self.clip_size - 1, -1, -1)

        feat_prop = ms.ops.Zeros()(feats["shallow"][0].shape, feats["shallow"][0].dtype)

        last_key = list(feats)[-2]

        idx_c = clip_idx_backward[0]
        feat = [feats[k][idx_c] for k in feats if k not in [module_name]]
        feat = [ms.ops.ReverseV2([1])(k) for k in feat] + [feat_prop]
        feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
        feats[module_name].append(feat_prop)

        for i in range(1, len(clip_idx_backward)):
            idx_c = clip_idx_backward[i]
            flow_n01 = flows[:, flow_idx_backward[self.clip_size * i - 1], :, :, :]
            flow_n12 = flows[:, flow_idx_backward[self.clip_size * i], :, :, :]
            flow_n23 = flows[:, flow_idx_backward[self.clip_size * i + 1], :, :, :]
            flow_n02 = flow_n12 + self.flow_warp_bilinear(flow_n01, flow_n12.transpose(0, 2, 3, 1))
            flow_n13 = flow_n23 + self.flow_warp_bilinear(flow_n12, flow_n23.transpose(0, 2, 3, 1))
            flow_n03 = flow_n23 + self.flow_warp_bilinear(flow_n02, flow_n23.transpose(0, 2, 3, 1))
            flow_n1 = ms.ops.stack([flow_n02, flow_n13], 1)
            flow_n2 = ms.ops.stack([flow_n12, flow_n03], 1)

            feat_q = feats[last_key][idx_c]
            feat_k = feats[last_key][clip_idx_backward[i - 1]]
            feat_q = ms.ops.ReverseV2([1])(feat_q)
            feat_k = ms.ops.ReverseV2([1])(feat_k)

            feat_prop_warped1 = self.flow_warp_bilinear(
                flatten_0_1(feat_prop), flatten_0_1(flow_n1.transpose(0, 1, 3, 4, 2))
            ).reshape(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
            feat_prop_warped2 = self.flow_warp_bilinear(
                flatten_0_1(ms.ops.ReverseV2([1])(feat_prop)),
                flatten_0_1(flow_n2.transpose(0, 1, 3, 4, 2)),
            ).reshape(n, feat_prop.shape[1], feat_prop.shape[2], h, w)

            feat_prop, flow_n1, flow_n2 = self.deform_align_dict[module_name](
                feat_q, feat_k, feat_prop, [feat_prop_warped1, feat_prop_warped2], [flow_n1, flow_n2], True
            )
            updated_flows[f"{direction}_n1"].append(flow_n1)
            updated_flows[f"{direction}_n2"].append(flow_n2)

            feat = [feats[k][idx_c] for k in feats if k not in [module_name]]
            feat = [ms.ops.ReverseV2([1])(k) for k in feat] + [feat_prop]
            feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
            feats[module_name].append(feat_prop)

        fi = len(feats[module_name]) - 1
        feats[module_name] = [
            ms.ops.ReverseV2([1])(feats[module_name][fi - i]) for i, _ in enumerate(feats[module_name])
        ]

        return feats, updated_flows

    def forward1(self, feats, flows_forward, flows_backward, updated_flows):
        direction = "forward"
        module_name = f"{direction}_1"
        flows = flows_forward if flows_forward is not None else ms.ops.ReverseV2([1])(flows_backward)
        feats[module_name] = []
        n, t, _, h, w = flows.shape
        flow_idx_forward = range(-1, t)
        clip_idx_forward = range(0, (t + 1) // self.clip_size)

        feat_prop = ms.ops.Zeros()(feats["shallow"][0].shape, feats["shallow"][0].dtype)

        last_key = list(feats)[-2]

        idx_c = clip_idx_forward[0]
        feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
        feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
        feats[module_name].append(feat_prop)

        for i in range(1, len(clip_idx_forward)):
            idx_c = clip_idx_forward[i]
            flow_n01 = flows[:, flow_idx_forward[self.clip_size * i - 1], :, :, :]
            flow_n12 = flows[:, flow_idx_forward[self.clip_size * i], :, :, :]
            flow_n23 = flows[:, flow_idx_forward[self.clip_size * i + 1], :, :, :]
            flow_n02 = flow_n12 + self.flow_warp_bilinear(flow_n01, flow_n12.transpose(0, 2, 3, 1))
            flow_n13 = flow_n23 + self.flow_warp_bilinear(flow_n12, flow_n23.transpose(0, 2, 3, 1))
            flow_n03 = flow_n23 + self.flow_warp_bilinear(flow_n02, flow_n23.transpose(0, 2, 3, 1))
            flow_n1 = ms.ops.stack([flow_n02, flow_n13], 1)
            flow_n2 = ms.ops.stack([flow_n12, flow_n03], 1)

            feat_q = feats[last_key][idx_c]
            feat_k = feats[last_key][clip_idx_forward[i - 1]]

            feat_prop_warped1 = self.flow_warp_bilinear(
                flatten_0_1(feat_prop), flatten_0_1(flow_n1.transpose(0, 1, 3, 4, 2))
            ).reshape(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
            feat_prop_warped2 = self.flow_warp_bilinear(
                flatten_0_1(ms.ops.ReverseV2([1])(feat_prop)),
                flatten_0_1(flow_n2.transpose(0, 1, 3, 4, 2)),
            ).reshape(n, feat_prop.shape[1], feat_prop.shape[2], h, w)

            feat_prop, flow_n1, flow_n2 = self.deform_align_dict[module_name](
                feat_q, feat_k, feat_prop, [feat_prop_warped1, feat_prop_warped2], [flow_n1, flow_n2], True
            )
            updated_flows[f"{direction}_n1"].append(flow_n1)
            updated_flows[f"{direction}_n2"].append(flow_n2)

            feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
            feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
            feats[module_name].append(feat_prop)

        return feats, updated_flows

    def backward2(self, feats, flows_backward, updated_flows):
        direction = "backward"
        module_name = f"{direction}_2"
        flows = flows_backward
        feats[module_name] = []
        n, t, _, h, w = flows.shape
        clip_idx_backward = range((t + 1) // self.clip_size - 1, -1, -1)

        feat_prop = ms.ops.Zeros()(feats["shallow"][0].shape, feats["shallow"][0].dtype)

        last_key = list(feats)[-2]

        idx_c = clip_idx_backward[0]
        feat = [feats[k][idx_c] for k in feats if k not in [module_name]]
        feat = [ms.ops.ReverseV2([1])(k) for k in feat] + [feat_prop]
        feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
        feats[module_name].append(feat_prop)

        for i in range(1, len(clip_idx_backward)):
            idx_c = clip_idx_backward[i]
            feat_q = feats[last_key][idx_c]
            feat_k = feats[last_key][clip_idx_backward[i - 1]]
            feat_q = ms.ops.ReverseV2([1])(feat_q)
            feat_k = ms.ops.ReverseV2([1])(feat_k)

            flow_n1 = updated_flows[f"{direction}_n1"][i - 1]
            flow_n2 = updated_flows[f"{direction}_n2"][i - 1]

            feat_prop_warped1 = self.flow_warp_bilinear(
                flatten_0_1(feat_prop), flatten_0_1(flow_n1.transpose(0, 1, 3, 4, 2))
            ).reshape(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
            feat_prop_warped2 = self.flow_warp_bilinear(
                flatten_0_1(ms.ops.ReverseV2([1])(feat_prop)),
                flatten_0_1(flow_n2.transpose(0, 1, 3, 4, 2)),
            ).reshape(n, feat_prop.shape[1], feat_prop.shape[2], h, w)

            feat_prop, flow_n1, flow_n2 = self.deform_align_dict[module_name](
                feat_q, feat_k, feat_prop, [feat_prop_warped1, feat_prop_warped2], [flow_n1, flow_n2], True
            )

            feat = [feats[k][idx_c] for k in feats if k not in [module_name]]
            feat = [ms.ops.ReverseV2([1])(k) for k in feat] + [feat_prop]
            feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
            feats[module_name].append(feat_prop)

        fi = len(feats[module_name]) - 1
        feats[module_name] = [
            ms.ops.ReverseV2([1])(feats[module_name][fi - i]) for i, _ in enumerate(feats[module_name])
        ]

        return feats

    def forward2(self, feats, flows_forward, flows_backward, updated_flows):
        direction = "forward"
        module_name = f"{direction}_{2}"
        flows = flows_forward if flows_forward is not None else ms.ops.ReverseV2([1])(flows_backward)
        feats[module_name] = []
        n, t, _, h, w = flows.shape
        clip_idx_forward = range(0, (t + 1) // self.clip_size)

        feat_prop = ms.ops.Zeros()(feats["shallow"][0].shape, feats["shallow"][0].dtype)

        last_key = list(feats)[-2]

        idx_c = clip_idx_forward[0]
        feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
        feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
        feats[module_name].append(feat_prop)

        for i in range(1, len(clip_idx_forward)):
            idx_c = clip_idx_forward[i]
            feat_q = feats[last_key][idx_c]
            feat_k = feats[last_key][clip_idx_forward[i - 1]]

            flow_n1 = updated_flows[f"{direction}_n1"][i - 1]
            flow_n2 = updated_flows[f"{direction}_n2"][i - 1]

            feat_prop_warped1 = self.flow_warp_bilinear(
                flatten_0_1(feat_prop), flatten_0_1(flow_n1.transpose(0, 1, 3, 4, 2))
            ).reshape(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
            feat_prop_warped2 = self.flow_warp_bilinear(
                flatten_0_1(ms.ops.ReverseV2([1])(feat_prop)),
                flatten_0_1(flow_n2.transpose(0, 1, 3, 4, 2)),
            ).reshape(n, feat_prop.shape[1], feat_prop.shape[2], h, w)

            feat_prop, flow_n1, flow_n2 = self.deform_align_dict[module_name](
                feat_q, feat_k, feat_prop, [feat_prop_warped1, feat_prop_warped2], [flow_n1, flow_n2], True
            )

            feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
            feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
            feats[module_name].append(feat_prop)

        return feats

    def construct(self, lqs):
        """Forward function for RVRT.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        t = lqs.shape[1]

        if self.upscale == 4:
            lqs_downsample = lqs.copy()
        else:
            raise NotImplementedError()
            # lqs_downsample = F.interpolate(lqs[:, :, :3, :, :].reshape(-1, 3, h, w), scale_factor=0.25, mode='bicubic')\
            #     .reshape(n, t, 3, h // 4, w // 4)

        # check whether the input is an extended sequence
        # self.check_if_mirror_extended(lqs)

        # shallow feature extractions
        feats = {}

        feats["shallow"] = list(ms.ops.split(self.feat_extract(lqs), output_num=t // self.clip_size, axis=1))
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # recurrent feature refinement

        updated_flows = {
            "backward_n1": [],
            "backward_n2": [],
            "forward_n1": [],
            "forward_n2": [],
        }

        feats, updated_flows = self.backward1(feats, flows_backward, updated_flows)
        feats, updated_flows = self.forward1(feats, flows_forward, flows_backward, updated_flows)
        feats = self.backward2(feats, flows_backward, updated_flows)
        feats = self.forward2(feats, flows_forward, flows_backward, updated_flows)

        # reconstruction
        return self.upsample(lqs[:, :, :3, :, :], feats)

    def relative_position_bias_to_table(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, WindowAttention):
                cell.relative_position_bias_to_table()

    def relative_position_table_to_bias(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, WindowAttention):
                cell.relative_position_table_to_bias()
