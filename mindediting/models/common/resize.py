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

from math import ceil, floor

import mindspore as ms
import mindspore.ops as ops
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops.operations.image_ops import ResizeBilinearV2


class BilinearResize(nn.Cell):
    def __init__(self, scale_factor, align_corners=True):
        super().__init__()

        self.scales = [scale_factor, scale_factor]

        self.resize = ResizeBilinearV2(align_corners, not align_corners)

    def construct(self, x):
        size = _scale_factor_convert_size(x.shape, self.scales, x.ndim - 2)
        out = self.resize(x, size)

        return out


def resize(x, scale_factor, mode="bilinear", align_corners=False):
    out = ops.interpolate(
        x, scale_factor=float(scale_factor), recompute_scale_factor=True, mode=mode, align_corners=align_corners
    )

    return out


def aa_resize(x, scale_factor):
    b, c, h, w = x.shape
    x = x.reshape(-1, 1, h, w)

    out_h, out_w = _aa_out_size(h, w, scale_factor)

    x = _aa_resize_1d(x, -2, size=out_h, scale=scale_factor)
    x = _aa_resize_1d(x, -1, size=out_w, scale=scale_factor)

    x = x.reshape(b, c, out_h, out_w)

    return x


@ops.constexpr
def _scale_factor_convert_size(shape, scale_factor, dim):
    return [int(floor(float(shape[i + 2]) * scale_factor[i])) for i in range(dim)]


@ops.constexpr
def _aa_out_size(h, w, scale_factor):
    out_h = ceil(h * scale_factor)
    out_w = ceil(w * scale_factor)

    return out_h, out_w


@ops.constexpr
def _aa_get_artifacts(shape, dim, size, scale, dtype):
    kernel_size = 4

    if scale < 1.0:
        antialiasing_factor = scale
        kernel_size = ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sizes
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    pos = ms.numpy.linspace(0, size - 1, size, dtype=dtype)
    pos = (pos + 0.5) / scale - 0.5
    base = pos.floor() - (kernel_size // 2) + 1
    dist = pos - base
    weight = _aa_get_weight(
        dist,
        kernel_size,
        antialiasing_factor=antialiasing_factor,
    )
    pad_pre, pad_post, base = _aa_get_padding(base, kernel_size, shape[dim])

    if dim == -2:
        pad = [0, 0, pad_pre, pad_post]
    else:
        pad = [pad_pre, pad_post, 0, 0]

    return weight, pad, kernel_size, base


@ops.constexpr
def _aa_get_1d_artifacts(h, w, dim, kernel_size):
    if dim == -2:
        k = (kernel_size, 1)
        h_out = h - kernel_size + 1
        w_out = w
    else:
        k = (1, kernel_size)
        h_out = h
        w_out = w - kernel_size + 1

    return k, h_out, w_out


def _aa_get_weight(dist, kernel_size, antialiasing_factor=1):
    idx = ms.numpy.linspace(0, kernel_size - 1, kernel_size, dtype=dist.dtype)
    buffer_pos = dist.reshape(1, -1) - idx.reshape(-1, 1)

    weight = _aa_cubic_contribution(antialiasing_factor * buffer_pos)
    weight = weight / P.ReduceSum(keep_dims=True)(weight, 0)

    return weight


def _aa_cubic_contribution(x, a=-0.5):
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = (ax <= 1).astype(x.dtype)
    range_12 = ((ax > 1) & (ax <= 2)).astype(x.dtype)

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12

    cont = cont_01 + cont_12

    return cont


def _aa_get_padding(base, kernel_size, x_size):
    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1

    pad_pre = -ops.minimum(r_min, 0).asnumpy().item()
    pad_post = ops.maximum(r_max - x_size + 1, 0).asnumpy().item()

    base += pad_pre

    return pad_pre, pad_post, base


def _aa_resize_1d(x, dim, size, scale):
    weight, pad, kernel_size, base = _aa_get_artifacts(x.shape, dim, size, scale, x.dtype)

    x_pad = ops.pad(x, pad, mode="reflect")
    unfold = _aa_reshape_tensor(x_pad, dim, kernel_size)

    if dim == -2:
        sample = unfold[..., base, :]
        weight = weight.reshape(1, kernel_size, sample.shape[2], 1)
    else:
        sample = unfold[..., base]
        weight = weight.reshape(1, kernel_size, 1, sample.shape[3])

    x = sample * weight
    x = P.ReduceSum(keep_dims=True)(x, 1)

    return x


def _aa_reshape_tensor(x, dim, kernel_size):
    b, _, h, w = x.shape
    k, h_out, w_out = _aa_get_1d_artifacts(h, w, dim, kernel_size)

    unfold = ops.unfold(x, k)
    unfold = unfold.reshape(b, kernel_size, -1, h_out, w_out)
    unfold = unfold.transpose(0, 2, 1, 3, 4)
    unfold = unfold.reshape(b, -1, h_out, w_out)

    return unfold
