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
from mindspore import nn
from mindspore.ops.operations.array_ops import Concat, Meshgrid, Transpose

from mindediting.models.common.unnorm_grid_sample import UnnormalizedGridSample2D
from mindediting.utils.utils import is_ascend


@ops.constexpr
def _make_grid(h, w, dtype):
    xx = ms.numpy.arange(w, dtype=dtype)
    yy = ms.numpy.arange(h, dtype=dtype)
    grid_y, grid_x = Meshgrid(indexing="ij")((yy, xx))
    grid_y, grid_x = grid_y.reshape(1, h, w), grid_x.reshape(1, h, w)

    return grid_x, grid_y


class FlowWarpAscend(nn.Cell):
    def __init__(self, interpolation="bilinear", padding_mode="border"):
        super().__init__()

        self.grid_sample = UnnormalizedGridSample2D(interpolation_mode=interpolation, padding_mode=padding_mode)

        self.transpose = Transpose()
        self.cast = ops.Cast()

    def construct(self, x, flow):
        h, w = x.shape[-2:]

        grid_x, grid_y = _make_grid(h, w, x.dtype)

        grid_x = grid_x[:, :, :, None] + self.cast(flow[:, 0, :, :, None], ms.float32)
        grid_y = grid_y[:, :, :, None] + self.cast(flow[:, 1, :, :, None], ms.float32)

        out = self.grid_sample(x, grid_x, grid_y)

        return out


class FlowWarpGPU(nn.Cell):
    def __init__(self, interpolation="bilinear", padding_mode="border", align_corners=True):
        super().__init__()

        self.grid_sample = nn_ops.GridSampler2D(
            interpolation_mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        self.concat = Concat(axis=3)

    def construct(self, x, flow):
        h, w = x.shape[-2:]

        grid_x, grid_y = _make_grid(h, w, x.dtype)
        grid_flow = self.concat(
            [
                (2.0 / (w - 1.0)) * (grid_x[..., None] + flow[:, 0, :, :, None]) - 1.0,
                (2.0 / (h - 1.0)) * (grid_y[..., None] + flow[:, 1, :, :, None]) - 1.0,
            ]
        )

        out = self.grid_sample(x, grid_flow)

        return out


def build_flow_warp(interpolation="bilinear", padding_mode="border", align_corners=True):
    if is_ascend():
        return FlowWarpAscend(interpolation, padding_mode)
    else:
        return FlowWarpGPU(interpolation, padding_mode, align_corners)
