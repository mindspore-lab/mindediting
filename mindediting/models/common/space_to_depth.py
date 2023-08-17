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

import mindspore.nn as nn
import mindspore.ops as ops


def pixel_unshuffle(x, block_size):
    n, c, h, w = x.shape
    h_block_size, w_block_size = block_size if isinstance(block_size, (list, tuple)) else (block_size,) * 2
    h_block_size = int(h_block_size)
    w_block_size = int(w_block_size)

    if h % h_block_size != 0:
        raise ValueError(f"h ({h}) should be divisible by block_size ({h_block_size}).")
    if w % w_block_size != 0:
        raise ValueError(f"h ({w}) should be divisible by block_size ({w_block_size}).")

    out = x.reshape(n, c, h // h_block_size, h_block_size, w // w_block_size, w_block_size)
    out = out.transpose(0, 1, 3, 5, 2, 4)
    out = out.reshape(n, c * h_block_size * w_block_size, h // h_block_size, w // w_block_size)

    return out


def pixel_shuffle(x, block_size):
    n, c, h, w = x.shape
    h_block_size, w_block_size = block_size if isinstance(block_size, (list, tuple)) else (block_size,) * 2
    h_block_size = int(h_block_size)
    w_block_size = int(w_block_size)

    if c % (h_block_size * w_block_size) != 0:
        raise ValueError(
            f"channels num ({c}) should be divisible by h_block_size * w_block_size ({h_block_size} * {w_block_size})."
        )

    out_channels = c // (h_block_size * w_block_size)
    out = x.reshape(n, out_channels, h_block_size, w_block_size, h, w)
    out = out.transpose(0, 1, 4, 2, 5, 3)
    out = out.reshape(n, out_channels, h * h_block_size, w * w_block_size)

    return out


class PixelShuffle(nn.Cell):
    def __init__(self, upscale_factor):
        super().__init__()
        self.scale = upscale_factor
        self.transpose = ops.Transpose()

    def construct(self, x):
        return pixel_shuffle(x, self.scale)


def depth_to_space(x, block_size):
    n, c, h, w = x.shape
    h_block_size, w_block_size = block_size if isinstance(block_size, (list, tuple)) else (block_size,) * 2
    h_block_size = int(h_block_size)
    w_block_size = int(w_block_size)

    if c % (h_block_size * w_block_size) != 0:
        raise ValueError(
            f"channels num ({c}) should be divisible by h_block_size * w_block_size ({h_block_size} * {w_block_size})."
        )

    out_channels = c // (h_block_size * w_block_size)
    out = x.reshape(n, h_block_size, w_block_size, out_channels, h, w)
    out = out.transpose(0, 3, 4, 1, 5, 2)
    out = out.reshape(n, out_channels, h * h_block_size, w * w_block_size)

    return out


class DepthToSpace(nn.Cell):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def construct(self, x):
        return depth_to_space(x, self.block_size)
