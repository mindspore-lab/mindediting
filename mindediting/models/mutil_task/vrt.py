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
import os
import warnings
from functools import reduce
from operator import mul

import mindspore as ms
import numpy as np
import scipy
from mindspore import nn
from mindspore.ops import grid_sample
from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations import nn_ops as NN_OPS

from mindediting.models.common.grid_sample import grid_sample_2d as grid_sample
from mindediting.utils.utils import is_ascend

from ..video_super_resolution.ttvsr import FlowWarp


def _pair(x):
    return x, x


def get_permute(input):
    if any(x in input for x in {"(", ")"}):
        raise NotImplementedError()
    order_before, order_after = [x.strip().split(" ") for x in input.split("->")]
    if len(order_before) != len(order_after):
        raise RuntimeError()
    permute = [order_before.index(x) for x in order_after]

    return permute


class nn_Transpose(nn.Cell):
    def __init__(self, perm) -> None:
        super(nn_Transpose, self).__init__()
        self.perm = perm if isinstance(perm, (list, tuple)) else get_permute(perm)

    def construct(self, x):
        return x.transpose(self.perm)


def _offset2grid(offset, K, p, h, w):
    n, khkw2, out_h, out_w = offset.shape
    khkw = int(khkw2 * 0.5)

    ys, xs = ms.ops.meshgrid(
        (ms.numpy.arange(0, out_h, dtype=offset.dtype), ms.numpy.arange(0, out_w, dtype=offset.dtype)),
        indexing="ij",
    )

    filter_offset_x = ms.numpy.tile(ms.numpy.arange(K, dtype=offset.dtype), K)
    filter_offset_y = ms.numpy.repeat(ms.numpy.arange(K, dtype=offset.dtype), K)

    x_coord = offset[:, :khkw] + xs[None, None] + filter_offset_x[None, :, None, None]
    y_coord = offset[:, khkw:] + ys[None, None] + filter_offset_y[None, :, None, None]

    x_coord = 2 * x_coord / (w + 2 * p - 1) - 1
    y_coord = 2 * y_coord / (h + 2 * p - 1) - 1

    x_coord = x_coord.reshape(-1, K * K, out_h * out_w)
    y_coord = y_coord.reshape(-1, K * K, out_h * out_w)

    coord = ms.ops.stack((x_coord, y_coord), axis=-1)
    return coord


def deformable_offsets2(x, offset, stride=1, pad=1, K=3):
    offset, mask = offset[:, : 2 * K * K, :, :], offset[:, 2 * K * K :, :, :]

    _, _, h, w = x.shape
    _, khkw2, _, _ = offset.shape

    if khkw2 != 2 * K * K:
        raise ValueError("The shape of the offset does not match the kernel size")

    grid = _offset2grid(offset, K, pad, h, w)
    x_pad = ms.nn.Pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))(x)
    x_st = grid_sample(x_pad, grid, align_corners=True)
    x_st = x_st * mask.reshape(x_st.shape[0], 1, x_st.shape[2], x_st.shape[3])

    return x_st


class AltDeformableConv2d(ms.nn.Cell):
    def __init__(
        self,
        deformable_groups=1,
        strides=(1, 1, 1, 1),
        padding=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        kernel_size=(3, 3),
        modulated=True,
    ):
        super(AltDeformableConv2d, self).__init__()

        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.modulated = modulated
        self.deformable_groups = deformable_groups
        self.is_ascend = is_ascend()

    def construct(self, x, offsets, weight, bias):
        groups = x.shape[1] // weight.shape[1]
        bias_add_ = _get_cache_prim(P.BiasAdd)()

        assert self.kernel_size[0] == self.kernel_size[1]
        K = self.kernel_size[0]
        K2 = K * K
        B, _, Hout, Wout = offsets.shape
        HW = Hout * Wout
        G = groups
        assert offsets.shape[1] == 3 * K2 * self.deformable_groups

        offsets = offsets.reshape(B, 3, self.deformable_groups, K2, Hout, Wout)
        offsets = offsets.transpose(0, 2, 1, 3, 4, 5)
        offsets = offsets.reshape(B * self.deformable_groups, 3 * K2, Hout, Wout)

        cin = x.shape[1]
        x = x.reshape(B * self.deformable_groups, cin // self.deformable_groups, x.shape[-2], x.shape[-1])
        fm_offset = deformable_offsets2(x, offsets)
        fm_offset = fm_offset.reshape(B, cin, K2, HW)

        fm_offset2 = fm_offset.reshape(B, G, cin // G * K2, HW)
        fm_offset2 = fm_offset2.transpose(1, 0, 3, 2).reshape(G, B * HW, cin // G * K2)

        cout = weight.shape[0]
        weight2 = weight.reshape(G, cout // G, cin // G, K2)
        weight2 = weight2.transpose(0, 2, 3, 1).reshape(G, cin // G * K2, cout // G)

        y = ms.ops.matmul(fm_offset2, weight2)  # G, B * HW, cout // G
        y = y.reshape(G, B, HW, cout // G)
        y = y.transpose(1, 0, 3, 2)
        output = y.reshape(B, cout, Hout, Wout)

        if bias is not None:
            output = bias_add_(output, bias)

        return output


class DeformableConv2d(ms.nn.Cell):
    def __init__(
        self,
        deformable_groups=1,
        strides=(1, 1, 1, 1),
        padding=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        kernel_size=(3, 3),
        modulated=True,
    ):
        super(DeformableConv2d, self).__init__()

        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.modulated = modulated
        self.deformable_groups = deformable_groups
        self.is_ascend = is_ascend()

    def construct(self, x, offsets, weight, bias):
        weight_shape = weight.shape
        out_channel = weight_shape[0]
        strides_conv = (self.kernel_size[0], self.kernel_size[1])
        groups = x.shape[1] // weight.shape[1]
        bias_add_ = _get_cache_prim(P.BiasAdd)()

        CHANNELS = 8
        if self.is_ascend and (self.deformable_groups > 1 or x.shape[1] % CHANNELS):
            deformable_offsets = _get_cache_prim(NN_OPS.DeformableOffsets)(
                self.strides, self.padding, self.kernel_size, self.dilations, "NCHW", 1, self.modulated
            )
            B, _, Hout, Wout = offsets.shape
            offsets = offsets.reshape(
                B, 3, self.deformable_groups, self.kernel_size[0] * self.kernel_size[1], Hout, Wout
            )
            fm_offset = []
            split_offsets = ms.ops.split(offsets, axis=2, output_num=self.deformable_groups)
            split_x = ms.ops.split(x, axis=1, output_num=self.deformable_groups)
            for i in range(self.deformable_groups):
                cur_offsets = split_offsets[i]
                cur_offsets = cur_offsets.reshape(B, -1, Hout, Wout)
                cur_x = split_x[i]
                cin = cur_x.shape[1]

                if cin % CHANNELS != 0:
                    cur_x = ms.nn.Pad(((0, 0), (0, CHANNELS - cin % CHANNELS), (0, 0), (0, 0)))(cur_x)

                cur_fm_offset = deformable_offsets(cur_x, cur_offsets)

                if cin % CHANNELS != 0:
                    cur_fm_offset = cur_fm_offset[:, :cin, :, :]

                fm_offset.append(cur_fm_offset)
            fm_offset = ms.ops.concat(fm_offset, axis=1)
        else:
            deformable_offsets = _get_cache_prim(NN_OPS.DeformableOffsets)(
                self.strides,
                self.padding,
                self.kernel_size,
                self.dilations,
                "NCHW",
                self.deformable_groups,
                self.modulated,
            )
            fm_offset = deformable_offsets(x, offsets)

        conv = _get_cache_prim(P.Conv2D)(out_channel, self.kernel_size, 1, "valid", 0, strides_conv, 1, groups)
        output = conv(fm_offset, weight)

        if bias is not None:
            output = bias_add_(output, bias)
        return output


class ModulatedDeformConv(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
    ):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.transposed = False
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.kernel_size = _pair(kernel_size)
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
            stdv = 1.0 / math.sqrt(n)
        self.weight = ms.Parameter(
            ms.ops.uniform(
                (out_channels, in_channels // groups, *self.kernel_size),
                ms.Tensor(-stdv, ms.float32),
                ms.Tensor(stdv, ms.float32),
                dtype=ms.float32,
            )
        )
        if bias:
            self.bias = ms.Parameter(ms.Tensor(np.zeros([out_channels]), ms.float32))
        else:
            raise NotImplementedError
            self.register_parameter("bias", None)


class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            pad_mode="pad",
            padding=self.padding,
            dilation=_pair(self.dilation),
            has_bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) * 0.5

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    low = norm_cdf((a - mean) / std)
    up = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [low, up], then translate to
    # [2l-1, 2u-1].
    tensor = np.random.uniform(2 * low - 1, 2 * up - 1, size=tensor.shape).astype(np.float32)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    # FIXME(ikrylov): no CPU implementation for Erfinv
    tensor = scipy.special.erfinv(tensor)

    tensor = ms.Tensor(tensor)

    # Transform to proper mean, std
    tensor = ms.ops.stop_gradient(tensor * std * math.sqrt(2.0))
    tensor = ms.ops.stop_gradient(tensor + mean)

    # Clamp to ensure it's in the proper range
    tensor = ms.ops.stop_gradient(ms.ops.clip_by_value(tensor, ms.Tensor(a, ms.float32), ms.Tensor(b, ms.float32)))

    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


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
    offset = ms.ops.concat((o2, o1), axis=2)
    offset = offset.transpose(0, 2, 1, 3, 4)
    offset = offset.reshape(offset.shape[0], -1, offset.shape[-2], offset.shape[-1])
    return offset


class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.

    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)
        self.pa_frames = kwargs.pop("pa_frames", 2)

        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        self.conv_offset = nn.SequentialCell(
            nn.Conv2d(
                (1 + self.pa_frames // 2) * self.in_channels + self.pa_frames,
                self.out_channels,
                3,
                1,
                padding=1,
                pad_mode="pad",
                has_bias=True,
            ),
            nn.LeakyReLU(alpha=0.1),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, padding=1, pad_mode="pad", has_bias=True),
            nn.LeakyReLU(alpha=0.1),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, padding=1, pad_mode="pad", has_bias=True),
            nn.LeakyReLU(alpha=0.1),
            nn.Conv2d(
                self.out_channels,
                3 * 9 * self.deformable_groups,
                3,
                1,
                padding=1,
                pad_mode="pad",
                weight_init="zeros",
                bias_init="zeros",
                has_bias=True,
            ),
        )

        if os.getenv("USE_ALT_DEFORM_CONV", "False").lower() in {"true", "on", "yes", "1"}:
            self.deformable_conv = AltDeformableConv2d(self.deformable_groups)
        else:
            self.deformable_conv = DeformableConv2d(self.deformable_groups)

    def construct(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset(ms.ops.concat(x_flow_warpeds + [x_current] + flows, axis=1))
        o1, o2, mask = ms.ops.split(out, output_num=3, axis=1)

        # offset
        offset = self.max_residue_magnitude * ms.ops.tanh(ms.ops.concat((o1, o2), axis=1))

        if self.pa_frames == 2:
            offset = offset + ms.numpy.tile(ms.ops.ReverseV2([1])(flows[0]), (1, offset.shape[1] // 2, 1, 1))
        elif self.pa_frames == 4:
            offset1, offset2 = ms.ops.split(offset, output_num=2, axis=1)
            offset1 = offset1 + ms.numpy.tile(ms.ops.ReverseV2([1])(flows[0]), (1, offset1.shape[1] // 2, 1, 1))
            offset2 = offset2 + ms.numpy.tile(ms.ops.ReverseV2([1])(flows[1]), (1, offset2.shape[1] // 2, 1, 1))
            offset = ms.ops.concat([offset1, offset2], axis=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * ms.ops.tanh(ms.ops.concat((o1, o2), axis=1))
            offset1, offset2, offset3 = ms.ops.split(offset, output_num=3, axis=1)
            offset1 = offset1 + ms.numpy.tile(ms.ops.ReverseV2([1])(flows[0]), (1, offset1.shape[1] // 2, 1, 1))
            offset2 = offset2 + ms.numpy.tile(ms.ops.ReverseV2([1])(flows[1]), (1, offset2.shape[1] // 2, 1, 1))
            offset3 = offset3 + ms.numpy.tile(ms.ops.ReverseV2([1])(flows[2]), (1, offset3.shape[1] // 2, 1, 1))
            offset = ms.ops.concat([offset1, offset2, offset3], axis=1)

        # mask
        mask = ms.ops.Sigmoid()(mask)

        offset = convert_offsets_from_pt_ms(offset)

        offset_and_mask = ms.ops.concat((offset, mask), axis=1)

        return self.deformable_conv(x, offset_and_mask, self.weight, self.bias)


class BasicModule(nn.Cell):
    """Basic Module for SpyNet."""

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.SequentialCell(
            nn.Conv2d(
                in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, pad_mode="pad", has_bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, pad_mode="pad", has_bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, pad_mode="pad", has_bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, pad_mode="pad", has_bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3, pad_mode="pad", has_bias=True
            ),
        )

    def construct(self, tensor_input):
        return self.basic_module(tensor_input)


def drop_path_(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + ms.ops.UniformReal()(shape)
    random_tensor = ms.ops.Floor()(random_tensor)  # binarize
    output = ms.ops.Div()(x, keep_prob) * random_tensor
    return output


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        return drop_path_(x, self.drop_prob, self.training)


def flow_warp_bilinear(x, flow, padding_mode="zeros", align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.shape
    # create mesh grid grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_x, grid_y = ms.ops.meshgrid((ms.numpy.arange(0, w), ms.numpy.arange(0, h)))
    grid = ms.ops.stack((grid_x, grid_y), 2)  # W(x), H(y), 2

    grid = ms.ops.stop_gradient(grid)

    vgrid = grid + flow

    vgrid_x, vgrid_y = ms.ops.split(vgrid, axis=3, output_num=2)

    vgrid_x = 2.0 * vgrid_x / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid_y / max(h - 1, 1) - 1.0

    vgrid_scaled = ms.ops.concat((vgrid_x, vgrid_y), axis=3)

    output = grid_sample(
        x,
        vgrid_scaled.astype(x.dtype),
        interpolation_mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
        dtype=x.dtype,
    )

    return output


def flow_warp_nearest4(x, flow, padding_mode="zeros", align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.shape
    # create mesh grid grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_x, grid_y = ms.ops.meshgrid((ms.numpy.arange(0, w), ms.numpy.arange(0, h)))
    grid = ms.ops.stack((grid_x, grid_y), 2)  # W(x), H(y), 2

    grid = ms.ops.stop_gradient(grid)

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x_floor = 2.0 * ms.ops.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
    vgrid_x_ceil = 2.0 * ms.ops.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
    vgrid_y_floor = 2.0 * ms.ops.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
    vgrid_y_ceil = 2.0 * ms.ops.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

    output00 = grid_sample(
        x,
        ms.ops.stack((vgrid_x_floor, vgrid_y_floor), axis=3),
        interpolation_mode="nearest",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    output01 = grid_sample(
        x,
        ms.ops.stack((vgrid_x_floor, vgrid_y_ceil), axis=3),
        interpolation_mode="nearest",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    output10 = grid_sample(
        x,
        ms.ops.stack((vgrid_x_ceil, vgrid_y_floor), axis=3),
        interpolation_mode="nearest",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    output11 = grid_sample(
        x,
        ms.ops.stack((vgrid_x_ceil, vgrid_y_ceil), axis=3),
        interpolation_mode="nearest",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    output = ms.ops.concat([output00, output01, output10, output11], 1)

    return output


class SpyNet(nn.Cell):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
        return_levels (list[int]): return flows of different levels. Default: [5].
    """

    def __init__(self, load_path=None, return_levels=[5], fast_grid_sample=True):
        super(SpyNet, self).__init__()
        if return_levels is None:
            return_levels = [5]
        self.return_levels = return_levels
        self.basic_module = nn.CellList([BasicModule() for _ in range(6)])
        if load_path:
            if load_path.endswith(".pth"):
                raise NotImplementedError
            elif load_path.endswith(".ckpt"):
                param_dict = ms.load_checkpoint(load_path)
                param_not_load = ms.load_param_into_net(self, param_dict)
                assert not param_not_load
            else:
                raise NotImplementedError

        self.mean = ms.Parameter(ms.Tensor([0.485, 0.456, 0.406], ms.float32).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = ms.Parameter(ms.Tensor([0.229, 0.224, 0.225], ms.float32).reshape(1, 3, 1, 1), requires_grad=False)

        self.max_level = 5

        self.flow_warp_bilinear = nn.CellList(
            [
                FlowWarp(
                    padding_mode="border",
                    interpolation="bilinear",
                    align_corners=True,
                    fast_grid_sample=fast_grid_sample,
                )
                for i in range(self.max_level, -1, -1)
            ]
        )

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp, w, h, w_floor, h_floor):
        flow_list = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(self.max_level):
            ref.insert(0, ms.ops.AvgPool(kernel_size=2, strides=2)(ref[0]))
            supp.insert(0, ms.ops.AvgPool(kernel_size=2, strides=2)(supp[0]))

        ref0_s0, ref0_s2, ref0_s3 = ref[0].shape[0], ref[0].shape[2], ref[0].shape[3]
        flow = ms.ops.Zeros()((ref0_s0, 2, (ref0_s2 // 2), (ref0_s3 // 2)), ref[0].dtype)

        for level in range(len(ref)):
            upsampled_flow = (
                ms.ops.interpolate(
                    flow,
                    sizes=(flow.shape[-2] * 2, flow.shape[-1] * 2),
                    coordinate_transformation_mode="align_corners",
                    mode="bilinear",
                )
                * 2.0
            )

            if upsampled_flow.shape[2] != ref[level].shape[2]:
                upsampled_flow = ms.nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 1)), mode="SYMMETRIC")(upsampled_flow)

            if upsampled_flow.shape[3] != ref[level].shape[3]:
                upsampled_flow = ms.nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 0)), mode="SYMMETRIC")(upsampled_flow)

            flow = (
                self.basic_module[level](
                    ms.ops.concat(
                        [
                            ref[level],
                            self.flow_warp_bilinear[level](
                                supp[level],
                                upsampled_flow.transpose(0, 2, 3, 1),
                                # padding_mode="border",
                            ),
                            upsampled_flow,
                        ],
                        1,
                    )
                )
                + upsampled_flow
            )

            if level in self.return_levels:
                # level=5 (scale=1), level=4 (scale=2), level=3 (scale=4), level=2 (scale=8)
                scale = 2 ** (self.max_level - level)
                # half_pixel does not work on CPU
                flow_out = ms.ops.interpolate(
                    flow,
                    sizes=(h // scale, w // scale),
                    coordinate_transformation_mode="half_pixel",
                    mode="bilinear",
                )
                flow_out *= ms.Tensor(
                    (float(w // scale) / (w_floor // scale), float(h // scale) / (h_floor // scale))
                ).reshape(2, 1, 1)
                flow_list.insert(0, flow_out.astype(ref[0].dtype))

        return flow_list

    @staticmethod
    def round32(x):
        return (x + 31) // 32 * 32

    def construct(self, ref, supp):
        h, w = ref.shape[2], ref.shape[3]
        w_floor = self.round32(w)
        h_floor = self.round32(h)

        # half_pixel does not work on CPU
        ref = ms.ops.interpolate(
            ref, sizes=(h_floor, w_floor), coordinate_transformation_mode="half_pixel", mode="bilinear"
        )
        supp = ms.ops.interpolate(
            supp, sizes=(h_floor, w_floor), coordinate_transformation_mode="half_pixel", mode="bilinear"
        )

        flow_list = self.process(ref, supp, w, h, w_floor, h_floor)

        return flow_list[0] if len(flow_list) == 1 else flow_list


def window_partition(x, window_size):
    """Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.reshape(
        B * D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = x.transpose(0, 2, 4, 1, 3, 5, 6).reshape(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.reshape(
        B * D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.transpose(0, 3, 1, 4, 2, 5, 6).reshape(B, D, H, W, -1)

    return x


def get_window_size2(x_size, window_size, use_shift_size=None):
    """Get the window size and the shift size"""

    use_window_size = list(window_size)
    if use_shift_size is not None:
        use_shift_size = list(use_shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if use_shift_size is not None:
                use_shift_size[i] = 0

    if use_shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


# @lru_cache()
def compute_mask(D, H, W, window_size, shift_size):
    """Compute attention mask for input of size (D, H, W). @lru_cache caches each stage results."""

    img_mask = ms.numpy.zeros((1, D, H, W, 1))  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.expand_dims(1) - mask_windows.expand_dims(2)
    attn_mask[attn_mask != 0] = -100.0
    return attn_mask


class Upsample(nn.SequentialCell):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        class Identity(nn.Cell):
            def __init__(self):
                super(Identity, self).__init__()

            def construct(self, x):
                return x

        class SomePixelShuffle(nn.Cell):
            def __init__(self, upscale_factor):
                super(SomePixelShuffle, self).__init__()
                self.upscale_factor = upscale_factor

            def construct(self, x):
                b, c, t, h, w = x.shape
                out_channel = int(c / (self.upscale_factor**2))
                out_h = int(h * self.upscale_factor)
                out_w = int(w * self.upscale_factor)

                x = x.reshape(b, out_channel, self.upscale_factor, self.upscale_factor, t, h, w)
                x = x.transpose(0, 1, 4, 5, 2, 6, 3).reshape(b, out_channel, t, out_h, out_w)
                return x

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(
                    nn.Conv3d(
                        num_feat,
                        4 * num_feat,
                        kernel_size=(1, 3, 3),
                        padding=(0, 0, 1, 1, 1, 1),
                        pad_mode="pad",
                        has_bias=True,
                    )
                )
                m.append(Identity())  # for compatibility with PyTorch weights
                m.append(SomePixelShuffle(2))
                m.append(Identity())  # for compatibility with PyTorch weights
                m.append(nn.LeakyReLU(alpha=0.1))
            m.append(
                nn.Conv3d(
                    num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 0, 1, 1, 1, 1), pad_mode="pad", has_bias=True
                )
            )
        elif scale == 3:
            m.append(
                nn.Conv3d(
                    num_feat,
                    9 * num_feat,
                    kernel_size=(1, 3, 3),
                    padding=(0, 0, 1, 1, 1, 1),
                    pad_mode="pad",
                    has_bias=True,
                )
            )
            m.append(Identity())  # for compatibility with PyTorch weights
            m.append(SomePixelShuffle(3))
            m.append(Identity())  # for compatibility with PyTorch weights
            m.append(nn.LeakyReLU(alpha=0.1))
            m.append(
                nn.Conv3d(
                    num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 0, 1, 1, 1, 1), pad_mode="pad", has_bias=True
                )
            )
        else:
            raise ValueError(f"scale {scale} is not supported. " "Supported scales: 2^n and 3.")
        super(Upsample, self).__init__(*m)


class Mlp_GEGLU(nn.Cell):
    """Multilayer perceptron with gated linear unit (GEGLU). Ref. "GLU Variants Improve Transformer".

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super(Mlp_GEGLU, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc11 = nn.Dense(in_features, hidden_features)
        self.fc12 = nn.Dense(in_features, hidden_features)

        if act_layer is nn.GELU:
            # for better compatibility with PyTorch
            self.act = act_layer(approximate=False)
        else:
            self.act = act_layer()

        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(1.0 - drop)

    def construct(self, x):
        x = self.act(self.fc11(x)) * self.fc12(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class WindowAttention(nn.Cell):
    """Window based multi-head mutual attention and self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        mut_attn (bool): If True, add mutual attention to the module. Default: True
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        mut_attn=True,
        input_resolution=None,
        relative_position_encoding=True,
    ):
        super(WindowAttention, self).__init__()

        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mut_attn = mut_attn
        self.is_ascend = is_ascend()
        # self attention with relative position bias
        self.relative_position_encoding = relative_position_encoding
        if self.relative_position_encoding:
            self.relative_position_bias_table = ms.Parameter(
                ms.ops.Zeros()(
                    ((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads),
                    ms.float32,
                ),  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH,
            )
            self.relative_position_index = ms.Parameter(self.get_position_index(window_size), requires_grad=False)

            self.relative_position_bias_table.set_data(trunc_normal_(self.relative_position_bias_table, std=0.02))

            assert input_resolution is not None
            self.input_resolution = input_resolution
            self.N = reduce(mul, get_window_size2(self.input_resolution, self.window_size))

            self.relative_position_bias = ms.Parameter(
                self.relative_position_bias_table[self.relative_position_index[: self.N, : self.N].reshape(-1)].reshape(
                    self.N, self.N, -1
                )
            )

        self.qkv_self = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.proj = nn.Dense(dim, dim)

        # mutual attention with sine position encoding
        if self.mut_attn:
            self.position_bias = ms.Parameter(
                self.get_sine_position_encoding(window_size[1:], dim // 2, normalize=True), requires_grad=False
            )
            self.qkv_mut = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
            self.proj = nn.Dense(2 * dim, dim)

        self.softmax = nn.Softmax(axis=-1)

    def relative_position_bias_to_table(self):
        if self.relative_position_encoding:
            N = reduce(mul, get_window_size2(self.input_resolution, self.window_size))
            assert N == self.N, f"{N} vs {self.N}"
            new_data = ms.ops.stop_gradient(ms.Tensor(self.relative_position_bias_table))
            new_data[self.relative_position_index[: self.N, : self.N].reshape(-1)] = ms.ops.stop_gradient(
                self.relative_position_bias.reshape(-1, new_data.shape[-1])
            )
            self.relative_position_bias_table.set_data(new_data)

    def relative_position_table_to_bias(self):
        if self.relative_position_encoding:
            N = reduce(mul, get_window_size2(self.input_resolution, self.window_size))
            assert N == self.N, f"{N} vs {self.N}"
            new_data = ms.ops.stop_gradient(
                self.relative_position_bias_table[self.relative_position_index[: self.N, : self.N].reshape(-1)].reshape(
                    self.N, self.N, -1
                )
            )
            self.relative_position_bias.set_data(new_data)

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

        # mutual attention
        if self.mut_attn:
            qkv = (
                self.qkv_mut(x + ms.numpy.tile(self.position_bias, (1, 2, 1)))
                .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                .transpose(2, 0, 3, 1, 4)
            )
            (q1, q2), (k1, k2), (v1, v2) = (
                ms.ops.Split(output_num=2, axis=2)(qkv[0]),
                ms.ops.Split(output_num=2, axis=2)(qkv[1]),
                ms.ops.Split(output_num=2, axis=2)(qkv[2]),
            )  # B_, nH, N/2, C
            x1_aligned = self.attention(q2, k1, v1, mask, (B_, N // 2, C), relative_position_encoding=False)
            x2_aligned = self.attention(q1, k2, v2, mask, (B_, N // 2, C), relative_position_encoding=False)
            x_out = ms.ops.Concat(axis=2)((ms.ops.Concat(axis=1)((x1_aligned, x2_aligned)), x_out))

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        B_, N, C = x_shape

        attn = ms.ops.matmul((q * self.scale), k.swapaxes(-2, -1))

        if relative_position_encoding:
            if self.is_ascend and self.training:
                relative_position_bias = self.relative_position_bias
            else:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index[:N, :N].reshape(-1)
                ].reshape(
                    N, N, -1
                )  # Wd*Wh*Ww, Wd*Wh*Ww,nH
            attn = attn + (relative_position_bias.transpose(2, 0, 1)).expand_dims(axis=0)  # B_, nH, N, N

        if mask is not None:
            nW = mask[0].shape[0]
            mask_idx = 0 if len(mask) == 1 or mask[0].shape[1] == N else 1
            assert mask[mask_idx].shape[1] == N, f"{mask[mask_idx].shape} vs {N}"
            assert mask[mask_idx].shape[2] == N, f"{mask[mask_idx].shape} vs {N}"
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + mask[mask_idx].expand_dims(1).expand_dims(0)
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = ms.ops.matmul(attn, v).swapaxes(1, 2).reshape(B_, N, C)

        return x

    def get_position_index(self, window_size):
        """Get pair-wise relative position index for each token inside the window."""

        coords_d = ms.numpy.arange(window_size[0])
        coords_h = ms.numpy.arange(window_size[1])
        coords_w = ms.numpy.arange(window_size[2])
        coords = ms.ops.stack(ms.ops.meshgrid((coords_d, coords_h, coords_w), indexing="ij"))  # 3, Wd, Wh, Ww
        coords_flatten = coords.reshape(3, -1)  # 3, Wd*Wh*Ww
        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        if self.is_ascend:
            relative_coords = relative_coords.astype(ms.float32)

        relative_coords += ms.Tensor((window_size[0] - 1, window_size[1] - 1, window_size[2] - 1))
        relative_coords *= ms.Tensor(((2 * window_size[1] - 1) * (2 * window_size[2] - 1), 2 * window_size[2] - 1, 1))

        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        if self.is_ascend:
            relative_position_index = ms.ops.Cast()(ms.ops.round(relative_position_index), ms.int64)

        return relative_position_index

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """Get sine position encoding"""

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        not_mask = ms.ops.ones((1, HW[0], HW[1]), ms.float32)
        y_embed = not_mask.cumsum(1, dtype=ms.float32)
        x_embed = not_mask.cumsum(2, dtype=ms.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = ms.numpy.arange(num_pos_feats, dtype=ms.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = ms.ops.stack((ms.ops.sin(pos_x[:, :, :, 0::2]), ms.ops.cos(pos_x[:, :, :, 1::2])), axis=4)
        pos_x = pos_x.reshape(pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], -1)
        pos_y = ms.ops.stack((ms.ops.sin(pos_y[:, :, :, 0::2]), ms.ops.cos(pos_y[:, :, :, 1::2])), axis=4)
        pos_y = pos_y.reshape(pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], -1)
        pos_embed = (ms.ops.concat((pos_y, pos_x), axis=3)).transpose(0, 3, 1, 2)

        pos_embed = pos_embed.reshape(pos_embed.shape[0], pos_embed.shape[1], -1)
        return pos_embed.transpose(0, 2, 1)


class TMSA(nn.Cell):
    """Temporal Mutual Self Attention (TMSA).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(6, 8, 8),
        shift_size=(0, 0, 0),
        mut_attn=True,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        input_resolution=None,
    ):
        super(TMSA, self).__init__()

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim if isinstance(dim, (list, tuple)) else [dim])
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mut_attn=mut_attn,
            input_resolution=input_resolution,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
        self.norm2 = norm_layer(dim if isinstance(dim, (list, tuple)) else [dim])
        self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size_cache = {}
        self.shift_size_cache = {}

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size2((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = x.transpose(0, 4, 1, 2, 3)
        x = x.reshape(-1, D, H, W)
        x = ms.numpy.pad(x, ((0, 0), (pad_d0, pad_d1), (pad_t, pad_b), (pad_l, pad_r)))
        x = x.reshape(B, C, D + pad_d0 + pad_d1, H + pad_t + pad_b, W + pad_l + pad_r)
        x = x.transpose(0, 2, 3, 4, 1)

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = ms.numpy.roll(x, shift=(-shift_size[0], -shift_size[1], -shift_size[2]), axis=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.reshape(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = ms.numpy.roll(shifted_x, shift=(shift_size[0], shift_size[1], shift_size[2]), axis=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        if self.drop_path is not None:
            x = self.drop_path(x)

        return x

    def forward_part2(self, x):
        x = self.mlp(self.norm2(x))
        if self.drop_path is not None:
            return self.drop_path(x)
        return x

    def construct(self, x, mask_matrix):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        x = x + self.forward_part1(x, mask_matrix)
        x = x + self.forward_part2(x)

        return x


@ms.ops.constexpr
def compute_attn_mask(D, H, W, window_size, shift_size):
    window_size, shift_size = get_window_size2((D, H, W), window_size, shift_size)
    Dp = int(np.ceil(D / window_size[0])) * window_size[0]
    Hp = int(np.ceil(H / window_size[1])) * window_size[1]
    Wp = int(np.ceil(W / window_size[2])) * window_size[2]
    return compute_mask(Dp, Hp, Wp, window_size, shift_size)


class TMSAG(nn.Cell):
    """Temporal Mutual Self Attention Group (TMSAG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=None,
        shift_size=None,
        mut_attn=True,
        mlp_ratio=2.0,
        qkv_bias=False,
        qk_scale=None,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        input_resolution=None,
    ):
        super(TMSAG, self).__init__()
        if window_size is None:
            window_size = [6, 8, 8]
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size
        self.mut_attn = mut_attn

        # build blocks
        self.blocks = nn.CellList(
            [
                TMSA(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                    mut_attn=mut_attn,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    input_resolution=input_resolution,
                )
                for i in range(depth)
            ]
        )

    def construct(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """

        B, _, D, H, W = x.shape
        x = x.transpose(0, 2, 3, 4, 1)

        attn_mask = compute_attn_mask(D, H, W, self.window_size, self.shift_size)
        if self.mut_attn:
            attn_mask = (attn_mask, attn_mask[:, : (attn_mask.shape[1] // 2), : (attn_mask.shape[2] // 2)])
        else:
            attn_mask = (attn_mask,)
        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.reshape(B, D, H, W, -1)
        x = x.transpose(0, 4, 1, 2, 3)

        return x


class RTMSA(nn.Cell):
    """Residual Temporal Mutual Self Attention (RTMSA). Only used in stage 8.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        input_resolution=None,
    ):
        super(RTMSA, self).__init__()

        self.residual_group = TMSAG(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mut_attn=False,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            input_resolution=input_resolution,
        )

        self.linear = nn.Dense(dim, dim)

    def construct(self, x):
        return x + self.linear(self.residual_group(x).swapaxes(1, 4)).swapaxes(1, 4)


class nn_RearrangeDown(nn.Cell):
    def __init__(self) -> None:
        super(nn_RearrangeDown, self).__init__()

    def construct(self, x):
        #   0 1 2  3  4     5   6
        # ("n c d (h neih) (w neiw) -> n d h w (neiw neih c)", neih=2, neiw=2)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2, 2, x.shape[4] // 2, 2)
        x = x.transpose(0, 2, 3, 5, 6, 4, 1)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], -1)
        return x


class nn_RearrangeUp(nn.Cell):
    def __init__(self) -> None:
        super(nn_RearrangeUp, self).__init__()

    def construct(self, x):
        #  0   1    2   3  4 5 6
        # n (neiw neih c) d h w -> n d (h neih) (w neiw) c

        x = x.reshape(x.shape[0], 2, 2, -1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = x.transpose(0, 4, 5, 2, 6, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4] * x.shape[5], x.shape[6])

        return x


class Stage(nn.Cell):
    """Residual Temporal Mutual Self Attention Group and Parallel Warping.

    Args:
        in_dim (int): Number of input channels.
        dim (int): Number of channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        reshape (str): Downscale (down), upscale (up) or keep the size (none).
        max_residue_magnitude (float): Maximum magnitude of the residual of optical flow.
    """

    def __init__(
        self,
        in_dim,
        dim,
        depth,
        num_heads,
        window_size,
        mul_attn_ratio=0.75,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pa_frames=2,
        deformable_groups=16,
        reshape=None,
        max_residue_magnitude=10,
        input_resolution=None,
    ):
        super(Stage, self).__init__()
        self.pa_frames = pa_frames

        # reshape the tensor
        if reshape == "none":
            self.reshape = nn.SequentialCell(
                nn_Transpose((0, 2, 3, 4, 1)),
                nn.LayerNorm([dim]),
                nn_Transpose((0, 4, 1, 2, 3)),
            )
        elif reshape == "down":
            self.reshape = nn.SequentialCell(
                nn_RearrangeDown(),
                nn.LayerNorm([4 * in_dim]),
                nn.Dense(4 * in_dim, dim),
                nn_Transpose((0, 4, 1, 2, 3)),
            )
        elif reshape == "up":
            self.reshape = nn.SequentialCell(
                nn_RearrangeUp(),
                nn.LayerNorm([in_dim // 4]),
                nn.Dense(in_dim // 4, dim),
                nn_Transpose((0, 4, 1, 2, 3)),
            )

        # mutual and self attention
        self.residual_group1 = TMSAG(
            dim=dim,
            depth=int(depth * mul_attn_ratio),
            num_heads=num_heads,
            window_size=(2, window_size[1], window_size[2]),
            mut_attn=True,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            input_resolution=input_resolution,
        )
        self.linear1 = nn.Dense(dim, dim)

        # only self attention
        self.residual_group2 = TMSAG(
            dim=dim,
            depth=depth - int(depth * mul_attn_ratio),
            num_heads=num_heads,
            window_size=window_size,
            mut_attn=False,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_path=drop_path,
            norm_layer=norm_layer,
            input_resolution=input_resolution,
        )
        self.linear2 = nn.Dense(dim, dim)

        # parallel warping
        if self.pa_frames:
            self.pa_deform = DCNv2PackFlowGuided(
                dim,
                dim,
                3,
                padding=1,
                deformable_groups=deformable_groups,
                max_residue_magnitude=max_residue_magnitude,
                pa_frames=pa_frames,
            )
            self.pa_fuse = Mlp_GEGLU(dim * (1 + 2), dim * (1 + 2), dim)

    def construct(self, x, flows_backward, flows_forward):
        x = self.reshape(x)
        x = self.linear1(self.residual_group1(x).swapaxes(1, 4)).swapaxes(1, 4) + x
        x = self.linear2(self.residual_group2(x).swapaxes(1, 4)).swapaxes(1, 4) + x

        if self.pa_frames:
            x = x.swapaxes(1, 2)
            if self.pa_frames == 2:
                x_backward, x_forward = self.get_aligned_feature_2frames(x, flows_backward, flows_forward)
            elif self.pa_frames == 4:
                x_backward, x_forward = self.get_aligned_feature_4frames(x, flows_backward, flows_forward)
            elif self.pa_frames == 6:
                x_backward, x_forward = self.get_aligned_feature_6frames(x, flows_backward, flows_forward)
            else:
                raise NotImplementedError
            x = self.pa_fuse(ms.ops.concat([x, x_backward, x_forward], 2).transpose(0, 1, 3, 4, 2)).transpose(
                0, 4, 1, 2, 3
            )

        return x

    def get_aligned_feature_2frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 2 frames."""

        # backward
        n = x.shape[1]
        x_backward = [ms.ops.ZerosLike()(x[:, -1, ...])]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[0][:, i - 1, ...]
            x_i_warped = flow_warp_bilinear(x_i, flow.transpose(0, 2, 3, 1))  # frame i+1 aligned towards i
            x_backward.insert(0, self.pa_deform(x_i, [x_i_warped], x[:, i - 1, ...], [flow]))

        # forward
        x_forward = [ms.ops.ZerosLike()(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[0][:, i, ...]
            x_i_warped = flow_warp_bilinear(x_i, flow.transpose(0, 2, 3, 1))  # frame i-1 aligned towards i
            x_forward.append(self.pa_deform(x_i, [x_i_warped], x[:, i + 1, ...], [flow]))

        return [ms.ops.stack(x_backward, 1), ms.ops.stack(x_forward, 1)]

    def get_aligned_feature_4frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 4 frames."""

        # backward
        n = x.shape[1]
        x_backward = [ms.ops.ZerosLike()(x[:, -1, ...])]
        for i in range(n, 1, -1):
            x_i = x[:, i - 1, ...]
            flow1 = flows_backward[0][:, i - 2, ...]
            if i == n:
                x_ii = ms.ops.ZerosLike()(x[:, n - 2, ...])
                flow2 = ms.ops.ZerosLike()(flows_backward[1][:, n - 3, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_backward[1][:, i - 2, ...]

            x_i_warped = flow_warp_bilinear(x_i, flow1.transpose(0, 2, 3, 1))  # frame i+1 aligned towards i
            x_ii_warped = flow_warp_bilinear(x_ii, flow2.transpose(0, 2, 3, 1))  # frame i+2 aligned towards i
            x_backward.insert(
                0,
                self.pa_deform(
                    ms.ops.concat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i - 2, ...], [flow1, flow2]
                ),
            )

        # forward
        x_forward = [ms.ops.ZerosLike()(x[:, 0, ...])]
        for i in range(-1, n - 2):
            x_i = x[:, i + 1, ...]
            flow1 = flows_forward[0][:, i + 1, ...]
            if i == -1:
                x_ii = ms.ops.ZerosLike()(x[:, 1, ...])
                flow2 = ms.ops.ZerosLike()(flows_forward[1][:, 0, ...])
            else:
                x_ii = x[:, i, ...]
                flow2 = flows_forward[1][:, i, ...]

            x_i_warped = flow_warp_bilinear(x_i, flow1.transpose(0, 2, 3, 1))  # frame i-1 aligned towards i
            x_ii_warped = flow_warp_bilinear(x_ii, flow2.transpose(0, 2, 3, 1))  # frame i-2 aligned towards i
            x_forward.append(
                self.pa_deform(
                    ms.ops.concat([x_i, x_ii], 1), [x_i_warped, x_ii_warped], x[:, i + 2, ...], [flow1, flow2]
                )
            )

        return [ms.ops.stack(x_backward, 1), ms.ops.stack(x_forward, 1)]

    def get_aligned_feature_6frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 6 frames."""

        # backward
        n = x.shape[1]
        x_backward = [ms.ops.ZerosLike()(x[:, -1, ...])]
        for i in range(n + 1, 2, -1):
            x_i = x[:, i - 2, ...]
            flow1 = flows_backward[0][:, i - 3, ...]
            if i == n + 1:
                x_ii = ms.ops.ZerosLike()(x[:, -1, ...])
                flow2 = ms.ops.ZerosLike()(flows_backward[1][:, -1, ...])
                x_iii = ms.ops.ZerosLike()(x[:, -1, ...])
                flow3 = ms.ops.ZerosLike()(flows_backward[2][:, -1, ...])
            elif i == n:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = ms.ops.ZerosLike()(x[:, -1, ...])
                flow3 = ms.ops.ZerosLike()(flows_backward[2][:, -1, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_backward[1][:, i - 3, ...]
                x_iii = x[:, i, ...]
                flow3 = flows_backward[2][:, i - 3, ...]

            x_i_warped = flow_warp_bilinear(x_i, flow1.transpose(0, 2, 3, 1))  # frame i+1 aligned towards i
            x_ii_warped = flow_warp_bilinear(x_ii, flow2.transpose(0, 2, 3, 1))  # frame i+2 aligned towards i
            x_iii_warped = flow_warp_bilinear(x_iii, flow3.transpose(0, 2, 3, 1))  # frame i+3 aligned towards i
            x_backward.insert(
                0,
                self.pa_deform(
                    ms.ops.concat([x_i, x_ii, x_iii], 1),
                    [x_i_warped, x_ii_warped, x_iii_warped],
                    x[:, i - 3, ...],
                    [flow1, flow2, flow3],
                ),
            )

        # forward
        x_forward = [ms.ops.ZerosLike()(x[:, 0, ...])]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow1 = flows_forward[0][:, i, ...]
            if i == 0:
                x_ii = ms.ops.ZerosLike()(x[:, 0, ...])
                flow2 = ms.ops.ZerosLike()(flows_forward[1][:, 0, ...])
                x_iii = ms.ops.ZerosLike()(x[:, 0, ...])
                flow3 = ms.ops.ZerosLike()(flows_forward[2][:, 0, ...])
            elif i == 1:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = ms.ops.ZerosLike()(x[:, 0, ...])
                flow3 = ms.ops.ZerosLike()(flows_forward[2][:, 0, ...])
            else:
                x_ii = x[:, i - 1, ...]
                flow2 = flows_forward[1][:, i - 1, ...]
                x_iii = x[:, i - 2, ...]
                flow3 = flows_forward[2][:, i - 2, ...]

            x_i_warped = flow_warp_bilinear(x_i, flow1.transpose(0, 2, 3, 1))  # frame i-1 aligned towards i
            x_ii_warped = flow_warp_bilinear(x_ii, flow2.transpose(0, 2, 3, 1))  # frame i-2 aligned towards i
            x_iii_warped = flow_warp_bilinear(x_iii, flow3.transpose(0, 2, 3, 1))  # frame i-3 aligned towards i
            x_forward.append(
                self.pa_deform(
                    ms.ops.concat([x_i, x_ii, x_iii], 1),
                    [x_i_warped, x_ii_warped, x_iii_warped],
                    x[:, i + 1, ...],
                    [flow1, flow2, flow3],
                )
            )

        return [ms.ops.stack(x_backward, 1), ms.ops.stack(x_forward, 1)]


class VRT(nn.Cell):
    """Video Restoration Transformer (VRT).
        `VRT: A Video Restoration Transformer`

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        out_chans (int): Number of output image channels. Default: 3.
        img_size (int | tuple(int)): Size of input image. Default: [6, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (6,8,8).
        depths (list[int]): Depths of each Transformer stage.
        indep_reconsts (list[int]): Layers that extract features of different frames independently.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mul_attn_ratio (float): Ratio of mutual attention layers. Default: 0.75.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        spynet_path (str): Pretrained SpyNet model path.
        pa_frames (float): Number of warpped frames. Default: 2.
        deformable_groups (float): Number of deformable groups. Default: 16.
        recal_all_flows (bool): If True, derive (t,t+2) and (t,t+3) flows from (t,t+1). Default: False.
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(
        self,
        upscale=4,
        in_chans=3,
        out_chans=3,
        img_size=None,
        window_size=None,
        depths=None,
        indep_reconsts=None,
        embed_dims=None,
        num_heads=None,
        mul_attn_ratio=0.75,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        spynet_path=None,
        pa_frames=2,
        deformable_groups=16,
        recal_all_flows=False,
        nonblind_denoising=False,
    ):
        super(VRT, self).__init__()
        if img_size is None:
            img_size = [6, 64, 64]
        if window_size is None:
            window_size = [6, 8, 8]
        if depths is None:
            depths = [8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4]
        if indep_reconsts is None:
            indep_reconsts = [11, 12]
        if embed_dims is None:
            embed_dims = [120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180]
        if num_heads is None:
            num_heads = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.pa_frames = pa_frames
        self.recal_all_flows = recal_all_flows
        self.nonblind_denoising = nonblind_denoising

        # conv_first
        if self.pa_frames:
            if self.nonblind_denoising:
                conv_first_in_chans = in_chans * (1 + 2 * 4) + 1
            else:
                conv_first_in_chans = in_chans * (1 + 2 * 4)
        else:
            conv_first_in_chans = in_chans
        self.conv_first = nn.Conv3d(
            conv_first_in_chans,
            embed_dims[0],
            kernel_size=(1, 3, 3),
            padding=(0, 0, 1, 1, 1, 1),
            pad_mode="pad",
            has_bias=True,
        )

        # main body
        if self.pa_frames:
            self.spynet = SpyNet(spynet_path, [2, 3, 4, 5])
        # stochastic depth decay rule
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depths))]
        reshapes = ["none", "down", "down", "down", "up", "up", "up"]
        self.scales = [1, 2, 4, 8, 4, 2, 1]

        # stage 1- 7
        for i in range(7):
            setattr(
                self,
                f"stage{i + 1}",
                Stage(
                    in_dim=embed_dims[i - 1],
                    dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    mul_attn_ratio=mul_attn_ratio,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                    pa_frames=pa_frames,
                    deformable_groups=deformable_groups,
                    reshape=reshapes[i],
                    max_residue_magnitude=10 / self.scales[i],
                    input_resolution=(img_size[0], img_size[1] // self.scales[i], img_size[2] // self.scales[i]),
                ),
            )

        # stage 8
        self.stage8 = nn.CellList(
            [
                nn.SequentialCell(
                    nn_Transpose((0, 2, 3, 4, 1)),
                    nn.LayerNorm([embed_dims[6]]),
                    nn.Dense(embed_dims[6], embed_dims[7]),
                    nn_Transpose((0, 4, 1, 2, 3)),
                )
            ]
        )
        for i in range(7, len(depths)):
            self.stage8.append(
                RTMSA(
                    dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=[1, window_size[1], window_size[2]] if i in indep_reconsts else window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                    input_resolution=img_size,
                )
            )

        self.norm = norm_layer([embed_dims[-1]])

        self.conv_after_body = nn.Dense(embed_dims[-1], embed_dims[0])

        # reconstruction
        if self.pa_frames:
            if self.upscale == 1:
                # for video deblurring, etc.
                self.conv_last = nn.Conv3d(
                    embed_dims[0],
                    out_chans,
                    kernel_size=(1, 3, 3),
                    padding=(0, 0, 1, 1, 1, 1),
                    pad_mode="pad",
                    has_bias=True,
                )
            else:
                # for video sr
                num_feat = 64
                self.conv_before_upsample = nn.SequentialCell(
                    nn.Conv3d(
                        embed_dims[0],
                        num_feat,
                        kernel_size=(1, 3, 3),
                        padding=(0, 0, 1, 1, 1, 1),
                        pad_mode="pad",
                        has_bias=True,
                    ),
                    nn.LeakyReLU(alpha=0.01),
                )
                self.upsample = Upsample(upscale, num_feat)
                self.conv_last = nn.Conv3d(
                    num_feat,
                    out_chans,
                    kernel_size=(1, 3, 3),
                    padding=(0, 0, 1, 1, 1, 1),
                    pad_mode="pad",
                    has_bias=True,
                )
        else:
            raise NotImplementedError
            num_feat = 64
            self.linear_fuse = nn.Conv2d(
                embed_dims[0] * img_size[0],
                num_feat,
                kernel_size=1,
                stride=1,
                padding=(0, 0, 0, 0),
                pad_mode="pad",
                has_bias=True,
            )
            self.conv_last = nn.Conv2d(
                num_feat, out_chans, kernel_size=7, stride=1, padding=(0, 0, 0, 0), pad_mode="pad", has_bias=True
            )

        self.is_ascend = is_ascend()
        if self.is_ascend:
            self.to_float(ms.float16)

    def reflection_pad2d(self, x, pad=1):
        """Reflection padding for any dtypes (torch.bfloat16.

        Args:
            x: (tensor): BxCxHxW
            pad: (int): Default: 1.
        """

        x = ms.ops.concat(
            [ms.ops.ReverseV2([2])(x[:, :, 1 : pad + 1, :]), x, ms.ops.ReverseV2([2])(x[:, :, -pad - 1 : -1, :])], 2
        )
        x = ms.ops.concat(
            [ms.ops.ReverseV2([3])(x[:, :, :, 1 : pad + 1]), x, ms.ops.ReverseV2([3])(x[:, :, :, -pad - 1 : -1])], 3
        )
        return x

    def construct(self, x):
        # main network
        if self.pa_frames:
            # obtain noise level map
            if self.nonblind_denoising:
                raise NotImplementedError

            x_lq = x.copy()

            # calculate flows
            flows_backward, flows_forward = self.get_flows(x)

            # warp input
            x_backward, x_forward = self.get_aligned_image_2frames(x, flows_backward[0], flows_forward[0])
            x = ms.ops.concat([x, x_backward, x_forward], 2)

            # concatenate noise level map
            if self.nonblind_denoising:
                raise NotImplementedError

            if self.upscale == 1:
                # video deblurring, etc.
                x = self.conv_first(x.swapaxes(1, 2))
                x = x + self.conv_after_body(
                    self.forward_features(x, flows_backward, flows_forward).swapaxes(1, 4)
                ).swapaxes(1, 4)
                x = self.conv_last(x).swapaxes(1, 2)
                return x + x_lq
            else:
                # video sr
                x = self.conv_first(x.swapaxes(1, 2))
                x = x + self.conv_after_body(
                    self.forward_features(x, flows_backward, flows_forward).swapaxes(1, 4)
                ).swapaxes(1, 4)
                x = self.conv_last(self.upsample(self.conv_before_upsample(x))).swapaxes(1, 2)
                H, W = x.shape[-2:]
                dim0, dim1, dim2, dim3, dim4 = x_lq.shape
                x_lq = x_lq.reshape(-1, dim2, dim3, dim4)
                # half pixel does not work on cpu
                resized = ms.ops.interpolate(
                    x_lq, sizes=(H, W), coordinate_transformation_mode="half_pixel", mode="bilinear"
                )
                resized = resized.reshape(dim0, dim1, dim2, resized.shape[-2], resized.shape[-1])
                return x + resized
        else:
            # video fi
            x_mean = x.mean([1, 3, 4], keep_dims=True)
            x = x - x_mean

            x = self.conv_first(x.swapaxes(1, 2))
            x = x + self.conv_after_body(self.forward_features(x, [], []).swapaxes(1, 4)).swapaxes(1, 4)

            x = ms.ops.concat(ms.ops.Unstack(2)(x), 1)
            x = self.conv_last(self.reflection_pad2d(ms.nn.LeakyReLU(alpha=0.2)(self.linear_fuse(x)), pad=3))
            x = ms.ops.stack(ms.ops.split(x, axis=1, output_num=3), 1)

            return x + x_mean

    def get_flows(self, x):
        """Get flows for 2 frames, 4 frames or 6 frames."""

        flows_backward = None
        flows_forward = None
        if self.pa_frames == 2:
            flows_backward, flows_forward = self.get_flow_2frames(x)
        elif self.pa_frames == 4:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(
                flows_forward_2frames, flows_backward_2frames
            )
            flows_backward = flows_backward_2frames + flows_backward_4frames
            flows_forward = flows_forward_2frames + flows_forward_4frames
        elif self.pa_frames == 6:
            flows_backward_2frames, flows_forward_2frames = self.get_flow_2frames(x)
            flows_backward_4frames, flows_forward_4frames = self.get_flow_4frames(
                flows_forward_2frames, flows_backward_2frames
            )
            flows_backward_6frames, flows_forward_6frames = self.get_flow_6frames(
                flows_forward_2frames, flows_backward_2frames, flows_forward_4frames, flows_backward_4frames
            )
            flows_backward = flows_backward_2frames + flows_backward_4frames + flows_backward_6frames
            flows_forward = flows_forward_2frames + flows_forward_4frames + flows_forward_6frames

        return flows_backward, flows_forward

    def get_flow_2frames(self, x):
        """Get flow between frames t and t+1 from x."""

        b, n, c, h, w = x.shape
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        # backward
        flows_backward = self.spynet(x_1, x_2)
        flows_backward = [
            flow.reshape(b, n - 1, 2, h // (2**i), w // (2**i)) for flow, i in zip(flows_backward, range(4))
        ]

        # forward
        flows_forward = self.spynet(x_2, x_1)
        flows_forward = [
            flow.reshape(b, n - 1, 2, h // (2**i), w // (2**i)) for flow, i in zip(flows_forward, range(4))
        ]

        return flows_backward, flows_forward

    def get_flow_4frames(self, flows_forward, flows_backward):
        """Get flow between t and t+2 from (t,t+1) and (t+1,t+2)."""

        # backward
        d = flows_forward[0].shape[1]
        print("flows_forward[0].shape---", flows_forward[0].shape, "d", d)
        flows_backward2 = []
        for flows in flows_backward:
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows[:, i - 1, :, :, :]  # flow from i+1 to i
                flow_n2 = flows[:, i, :, :, :]  # flow from i+2 to i+1
                flow_list.insert(
                    0, flow_n1 + flow_warp_bilinear(flow_n2, flow_n1.transpose(0, 2, 3, 1))
                )  # flow from i+2 to i
            flows_backward2.append(ms.ops.stack(flow_list, 1))

        # forward
        flows_forward2 = []
        for flows in flows_forward:
            flow_list = []
            for i in range(1, d):
                flow_n1 = flows[:, i, :, :, :]  # flow from i-1 to i
                flow_n2 = flows[:, i - 1, :, :, :]  # flow from i-2 to i-1
                flow_list.append(
                    flow_n1 + flow_warp_bilinear(flow_n2, flow_n1.transpose(0, 2, 3, 1))
                )  # flow from i-2 to i
            flows_forward2.append(ms.ops.stack(flow_list, 1))

        return flows_backward2, flows_forward2

    def get_flow_6frames(self, flows_forward, flows_backward, flows_forward2, flows_backward2):
        """Get flow between t and t+3 from (t,t+2) and (t+2,t+3)."""

        # backward
        d = flows_forward2[0].shape[1]
        print("flows_forward[0].shape--++++-", flows_forward[0].shape, "d", d)
        flows_backward3 = []
        for flows, flows2 in zip(flows_backward, flows_backward2):
            flow_list = []
            for i in range(d - 1, 0, -1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i+2 to i
                flow_n2 = flows[:, i + 1, :, :, :]  # flow from i+3 to i+2
                flow_list.insert(
                    0, flow_n1 + flow_warp_bilinear(flow_n2, flow_n1.transpose(0, 2, 3, 1))
                )  # flow from i+3 to i
            flows_backward3.append(ms.ops.stack(flow_list, 1))

        # forward
        flows_forward3 = []
        for flows, flows2 in zip(flows_forward, flows_forward2):
            flow_list = []
            for i in range(2, d + 1):
                flow_n1 = flows2[:, i - 1, :, :, :]  # flow from i-2 to i
                flow_n2 = flows[:, i - 2, :, :, :]  # flow from i-3 to i-2
                flow_list.append(
                    flow_n1 + flow_warp_bilinear(flow_n2, flow_n1.transpose(0, 2, 3, 1))
                )  # flow from i-3 to i
            flows_forward3.append(ms.ops.stack(flow_list, 1))

        return flows_backward3, flows_forward3

    def get_aligned_image_2frames(self, x, flows_backward, flows_forward):
        """Parallel feature warping for 2 frames."""

        # backward
        n = x.shape[1]
        x_backward = [ms.numpy.tile(ms.ops.ZerosLike()(x[:, -1, ...]), (1, 4, 1, 1))]
        for i in range(n - 1, 0, -1):
            x_i = x[:, i, ...]
            flow = flows_backward[:, i - 1, ...]
            x_backward.insert(0, flow_warp_nearest4(x_i, flow.transpose(0, 2, 3, 1)))  # frame i+1 aligned towards i

        # forward
        x_forward = [ms.numpy.tile(ms.ops.ZerosLike()(x[:, 0, ...]), (1, 4, 1, 1))]
        for i in range(0, n - 1):
            x_i = x[:, i, ...]
            flow = flows_forward[:, i, ...]
            x_forward.append(flow_warp_nearest4(x_i, flow.transpose(0, 2, 3, 1)))  # frame i-1 aligned towards i

        return [ms.ops.stack(x_backward, 1), ms.ops.stack(x_forward, 1)]

    def forward_features(self, x, flows_backward, flows_forward):
        """Main network for feature extraction."""

        x1 = self.stage1(x, flows_backward[0::4], flows_forward[0::4])
        x2 = self.stage2(x1, flows_backward[1::4], flows_forward[1::4])
        x3 = self.stage3(x2, flows_backward[2::4], flows_forward[2::4])
        x4 = self.stage4(x3, flows_backward[3::4], flows_forward[3::4])
        x = self.stage5(x4, flows_backward[2::4], flows_forward[2::4])
        x = self.stage6(x + x3, flows_backward[1::4], flows_forward[1::4])
        x = self.stage7(x + x2, flows_backward[0::4], flows_forward[0::4])
        x = x + x1

        for layer in self.stage8:
            x = layer(x)

        x = x.transpose(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.transpose(0, 4, 1, 2, 3)

        return x

    def relative_position_bias_to_table(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, WindowAttention):
                cell.relative_position_bias_to_table()

    def relative_position_table_to_bias(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, WindowAttention):
                cell.relative_position_table_to_bias()
