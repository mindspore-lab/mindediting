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
from mindspore import nn, ops
from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations import nn_ops as NN_OPS

from mindediting.models.common.unnorm_grid_sample import UnnormalizedGridSample2D
from mindediting.models.mutil_task.vrt import ModulatedDeformConvPack
from mindediting.utils.utils import is_ascend


class DeformableConv2dGPU(nn.Cell):
    def __init__(
        self,
        deformable_groups=1,
        strides=(1, 1, 1, 1),
        padding=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        kernel_size=(3, 3),
        modulated=True,
    ):
        super().__init__()

        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.modulated = modulated
        self.deformable_groups = deformable_groups
        self.is_ascend = is_ascend()

    def construct(self, x, offsets_x, offsets_y, magnitudes, weight, bias):
        out_channel = weight.shape[0]
        strides_conv = (self.kernel_size[0], self.kernel_size[1])
        groups = x.shape[1] // weight.shape[1]

        deformable_offsets = _get_cache_prim(NN_OPS.DeformableOffsets)(
            self.strides, self.padding, self.kernel_size, self.dilations, "NCHW", self.deformable_groups, self.modulated
        )
        offsets = ops.concat([offsets_x, offsets_y, magnitudes], 1)
        fm_offset = deformable_offsets(x, offsets)

        conv = _get_cache_prim(P.Conv2D)(out_channel, self.kernel_size, 1, "valid", 0, strides_conv, 1, groups)
        output = conv(fm_offset, weight)

        if bias is not None:
            output = _get_cache_prim(P.BiasAdd)()(output, bias)

        return output


@ops.constexpr
def _make_grid(kernel, h, w, dtype):
    yy = ms.numpy.arange(0, h, dtype=dtype)
    xx = ms.numpy.arange(0, w, dtype=dtype)

    ys, xs = P.Meshgrid(indexing="ij")((yy, xx))

    filter_offset_x = ms.numpy.tile(ms.numpy.arange(kernel, dtype=dtype) - kernel // 2, kernel)
    filter_offset_y = ms.numpy.repeat(ms.numpy.arange(kernel, dtype=dtype) - kernel // 2, kernel)

    grid_x = xs[None, None] + filter_offset_x[None, :, None, None]
    grid_y = ys[None, None] + filter_offset_y[None, :, None, None]

    return grid_x, grid_y


class DeformableConv2dAscend(nn.Cell):
    def __init__(
        self,
        deformable_groups=1,
        strides=(1, 1, 1, 1),
        padding=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        kernel_size=(3, 3),
        modulated=True,
    ):
        super().__init__()

        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.modulated = modulated
        self.deformable_groups = deformable_groups

        self.bias_add = P.BiasAdd()
        self.grid_sample = UnnormalizedGridSample2D(interpolation_mode="bilinear", padding_mode="zeros")
        self.cast = ops.Cast()

    def _deformable_offsets(self, x, offsets_x, offsets_y, magnitudes, kernel=3):
        _, c, h, w = x.shape

        grid_x, grid_y = _make_grid(kernel, h, w, offsets_x.dtype)
        grid_x = (self.cast(offsets_x, ms.float32) + grid_x).reshape(-1, kernel * kernel, h * w, 1)
        grid_y = (self.cast(offsets_y, ms.float32) + grid_y).reshape(-1, kernel * kernel, h * w, 1)

        out = self.grid_sample(x, grid_x, grid_y)
        out = out * magnitudes.reshape(-1, 1, kernel * kernel, h * w)
        out = out.reshape(-1, c, kernel, kernel, h, w)
        out = out.transpose(0, 1, 4, 2, 5, 3)

        return out

    def construct(self, x, offsets_x, offsets_y, magnitudes, weight, bias):
        groups = x.shape[1] // weight.shape[1]
        assert groups == 1

        assert self.kernel_size[0] == self.kernel_size[1]
        kernel = self.kernel_size[0]
        strides_conv = (self.kernel_size[0], self.kernel_size[1])
        b, _, out_h, out_w = offsets_x.shape
        assert offsets_x.shape[1] == kernel * kernel * self.deformable_groups
        assert offsets_y.shape[1] == kernel * kernel * self.deformable_groups
        assert magnitudes.shape[1] == kernel * kernel * self.deformable_groups

        channels = x.shape[1]
        deform_group_size = channels // self.deformable_groups

        x = x.reshape(b * self.deformable_groups, deform_group_size, x.shape[-2], x.shape[-1])
        offsets_x = offsets_x.reshape(b * self.deformable_groups, kernel * kernel, out_h, out_w)
        offsets_y = offsets_y.reshape(b * self.deformable_groups, kernel * kernel, out_h, out_w)
        magnitudes = magnitudes.reshape(b * self.deformable_groups, kernel * kernel, out_h, out_w)

        fm_offset = self._deformable_offsets(x, offsets_x, offsets_y, magnitudes, kernel)
        fm_offset = fm_offset.reshape(b, channels, kernel * out_h, kernel * out_w)

        conv = _get_cache_prim(P.Conv2D)(channels, self.kernel_size, 1, "valid", 0, strides_conv, 1, 1)
        y = conv(fm_offset, weight)

        if bias is not None:
            y = self.bias_add(y, bias)

        return y


def build_deform_conv(
    deformable_groups=1,
    strides=(1, 1, 1, 1),
    padding=(1, 1, 1, 1),
    dilations=(1, 1, 1, 1),
    kernel_size=(3, 3),
    modulated=True,
):
    if is_ascend():
        return DeformableConv2dAscend(deformable_groups, strides, padding, dilations, kernel_size, modulated)
    else:
        return DeformableConv2dGPU(deformable_groups, strides, padding, dilations, kernel_size, modulated)


class ModulatedDeformConvPack2D(ModulatedDeformConvPack):
    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack2D, self).__init__(*args, **kwargs)

        self.offset_size = 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]

        self.deformable_conv = build_deform_conv(
            deformable_groups=self.deformable_groups,
            strides=(1, self.stride, self.stride, 1),
            padding=(1, self.padding, self.padding, 1),
            dilations=(1, self.dilation, self.dilation, 1),
            kernel_size=self.kernel_size,
            modulated=True,
        )

    @staticmethod
    def _convert_offsets_from_pt_ms(offsets):
        """
        This method does conversion of offsets used in PyTorch to offsets used in MindSpore.
        PyTorch offsets shape: [B, GROUPS x Hf x Wf x 2, Hout, Wout]
        MindSpore offsets shape: [B, 2 x GROUPS x Hf x Wf, Hout, Wout]
        Where the '2' corresponds to coordinates. Moreover, order of offset coordinates in Pytorch is (y, x),
            in MindSpore: it is (x, y).
        """

        b, _, h, w = offsets.shape

        offsets = offsets.reshape(b, -1, 2, h, w)
        offsets_x = offsets[:, :, 1]
        offsets_y = offsets[:, :, 0]

        return offsets_x, offsets_y

    def construct(self, x):
        out = self.conv_offset(x)

        offset = out[:, : self.offset_size]
        mask = ops.sigmoid(out[:, self.offset_size :])

        offset_x, offset_y = self._convert_offsets_from_pt_ms(offset)

        out = self.deformable_conv(x, offset_x, offset_y, mask, self.weight, self.bias)

        return out
