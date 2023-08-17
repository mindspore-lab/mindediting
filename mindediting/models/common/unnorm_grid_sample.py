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
import mindspore.ops as ops
import mindspore.ops.operations.nn_ops as nn_ops

from mindediting.utils.mixed_precision import NotSetPrecisionCell
from mindediting.utils.utils import is_ascend


@ops.constexpr(get_instance=False)
def _zero_pad_limits(H, W):
    return -2, -2, W + 1, H + 1


@ops.constexpr(get_instance=False)
def _border_pad_limits(H, W):
    return 0, 0, W - 1, H - 1


class _NearestInterpolation(NotSetPrecisionCell):
    def __init__(self, pad):
        super().__init__()

        self.pad = pad
        self.round = ops.Round()

    def construct(self, input_pad, grid_x, grid_y, N, C, W):
        ix_nearest = self.round(grid_x[:, :, :, 0]).astype(ms.int32)
        iy_nearest = self.round(grid_y[:, :, :, 1]).astype(ms.int32)

        grid_nearest = (iy_nearest + self.pad) * (W + 2 * self.pad) + ix_nearest + self.pad
        grid_nearest = grid_nearest.reshape(N, 1, -1)
        grid_nearest = grid_nearest.broadcast_to((-1, C, -1))

        out = ops.gather_elements(input_pad, 2, grid_nearest)

        return out


class _BilinearInterpolation(NotSetPrecisionCell):
    def __init__(self, pad):
        super().__init__()

        self.pad = pad
        self.floor = ops.Floor()
        self.cast = ops.Cast()

    @staticmethod
    def _index_processing(x0, y0, num_cols, pad, input_pad_shape):
        x0_int32 = x0.astype(ms.int32)
        y0_int32 = y0.astype(ms.int32)

        n, c, h, w = input_pad_shape
        assert w == num_cols + 2 * pad
        batch_stride = h * w
        batch_offsets = ms.numpy.arange(0, n * batch_stride, batch_stride, dtype=ms.int32).reshape(-1, 1, 1, 1)

        grid_y0_x0 = (y0_int32 + pad) * w + x0_int32 + pad + batch_offsets

        return grid_y0_x0

    def construct(self, input_pad, grid_x, grid_y, N, C, W):
        x0 = self.floor(grid_x)
        y0 = self.floor(grid_y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0

        w_a = self.cast((x1 - grid_x) * (y1 - grid_y), input_pad.dtype)
        w_b = self.cast((x1 - grid_x) * (grid_y - y0), input_pad.dtype)
        w_c = self.cast((grid_x - x0) * (y1 - grid_y), input_pad.dtype)
        w_d = self.cast((grid_x - x0) * (grid_y - y0), input_pad.dtype)
        w = ops.concat([w_a, w_b, w_c, w_d], 3)
        w = w.reshape(-1, 4, 1)  # [N * Hgrid * Wgrid, 4, 1]

        grids = self._index_processing(x0, y0, W, self.pad, input_pad.shape)
        grids = grids.reshape(-1)

        # input_pad [N, C, Hpad, Wpad]
        input_pad = ops.transpose(input_pad, (0, 2, 3, 1))  # [N, Hpad, Wpad, C]
        input_pad_pad = ms.numpy.pad(input_pad, ((0, 0), (0, 1), (0, 1), (0, 0)))
        input_pad = ops.concat(
            [
                input_pad,
                input_pad_pad[:, 1:, :-1],
                input_pad_pad[:, :-1, 1:],
                input_pad_pad[:, 1:, 1:],
            ],
            axis=-1,
        )  # [N, Hpad, Wpad, 4 * C]
        input_pad = input_pad.reshape(-1, 4 * C)  # [N * Hpad * Wpad, 4 * C]

        values = ops.gather(input_pad, grids, axis=0)  # [N * Hgrid * Wgrid, 4 * C]
        values = values.reshape(-1, 4, C)  # [N * Hgrid * Wgrid, 4, C]

        out = w * values
        out = sum([out[:, 0], out[:, 1], out[:, 2], out[:, 3]])
        out = out.reshape(N, -1, C)  # [N, Hgrid * Wgrid, C]
        out = ops.transpose(out, (0, 2, 1))  # [N, C, Hgrid * Wgrid]

        return out


class UnnormalizedGridSample2D(NotSetPrecisionCell):
    INTERPOLATION_MODULES = dict(bilinear=_BilinearInterpolation, nearest=_NearestInterpolation)
    CLIPPING_MODULES = dict(border=_border_pad_limits, zeros=_zero_pad_limits)

    def __init__(self, interpolation_mode="bilinear", padding_mode="zeros"):
        super().__init__()

        assert interpolation_mode in self.INTERPOLATION_MODULES.keys()
        assert padding_mode in self.CLIPPING_MODULES.keys()

        pad = 1 if padding_mode == "border" else 3
        self.pad = pad

        self.interpolate = self.INTERPOLATION_MODULES[interpolation_mode](pad)
        self.clip_limits = self.CLIPPING_MODULES[padding_mode]()

    def construct(self, input_x, grid_x, grid_y):
        N, C, H, W = input_x.shape
        grid_H, grid_W = grid_x.shape[1:3]

        limits = self.clip_limits(H, W)
        grid_x = ops.clip_by_value(ops.stop_gradient(grid_x), limits[0], limits[2])
        grid_y = ops.clip_by_value(ops.stop_gradient(grid_y), limits[1], limits[3])

        input_pad = ms.numpy.pad(input_x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)))

        out = self.interpolate(input_pad, grid_x, grid_y, N, C, W)
        out = out.reshape(N, C, grid_H, grid_W)

        return out


def build_grid_sample(interpolation_mode="bilinear", padding_mode="zeros", align_corners=False, normalized_input=True):
    if is_ascend():
        assert not normalized_input
        return UnnormalizedGridSample2D(interpolation_mode, padding_mode)
    else:
        assert normalized_input
        return nn_ops.GridSampler2D(interpolation_mode, padding_mode, align_corners)
