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
from mindspore.ops import functional as F


class UnfoldAscendFP32(nn.Unfold):
    """
    Extracts patches from images. Optimized version of `mindspore.nn.Unfold` for Ascend.
    Please refer to the corresponding documentation page for details.
    """

    def __init__(self, ksizes, strides, rates, padding="valid"):
        super().__init__(ksizes, strides, rates, padding)

        self.ksizes = ksizes[0], ksizes[3], ksizes[1], ksizes[2]
        self.strides = strides[0], strides[3], strides[1], strides[2]
        self.rates = rates[0], rates[3], rates[1], rates[2]

        self.base = 256  # <= 1024 should be good for fp16.

    def get_indices(self, input_shape, dtype):
        """
        Map indices of spatial elements from the original feature map to the unfolded one.
        """

        _, _, x_row, x_col = input_shape

        # Zeroth index for padding.
        x_idx = F.tuple_to_array(range(1, x_row * x_col + 1))

        # Currently Unfold operation on Ascend does not support any input apart from fp16,
        # so using two fp16 variables to store one integer index without the loss of precision.
        x_idx = ops.stack(ms.numpy.divmod(x_idx, self.base, dtype=dtype))
        x_idx = x_idx.reshape(2, 1, x_row, x_col)
        idx = self.extract_image_patches(x_idx)

        # Convert back to integer.
        idx = idx.astype(ms.int32)
        idx = idx[0] * self.base + idx[1]

        # Make all valid indices zero-based and map gradients from padded values to the last element (-1) of dx.
        idx -= 1

        return idx

    def construct(self, input_x):
        x_shape = input_x.shape
        x_batch, x_depth, x_row, x_col = x_shape
        indices = self.get_indices(x_shape, ops.dtype(input_x))
        _, y_row, y_col = indices.shape
        indices = indices.reshape(-1)
        x = input_x.reshape(x_batch * x_depth, x_row * x_col).transpose(1, 0)

        # Add zeros in the last position to account for padding.
        x = ops.Pad(((0, 1), (0, 0)))(x)
        patches = ops.gather(x, indices, 0)
        patches = patches.reshape(self.ksizes[2], self.ksizes[3], y_row, y_col, x_batch, x_depth)
        patches = patches.transpose(4, 0, 1, 5, 2, 3).reshape(x_batch, -1, y_row, y_col)

        return patches

    def bprop(self, x, out, dout):
        x_shape = x.shape
        x_batch, x_depth, x_row, x_col = x_shape
        idx = self.get_indices(x_shape, ops.dtype(dout))
        idx = idx.reshape(-1, 1)

        _, _, y_row, y_col = dout.shape
        grad = dout.reshape(x_batch, self.ksizes[2], self.ksizes[3], x_depth, y_row, y_col).transpose(1, 2, 4, 5, 0, 3)
        grad = grad.reshape(-1, x_batch * x_depth)

        # The last element is to collect gradients from padded values. Has to be dropped afterwards.
        dx = ops.Zeros()((x_row * x_col + 1, x_batch * x_depth), ops.dtype(grad))
        dx = ops.tensor_scatter_add(dx, idx, grad)

        # Remove the last slice as it contains gradients for padded values.
        dx = dx[:-1]
        dx = dx.reshape(x_row, x_col, x_batch, x_depth)
        dx = dx.transpose(2, 3, 0, 1)

        return (dx,)


def build_unfold(ksizes, strides, rates, padding="valid"):
    device = ms.context.get_context("device_target")
    if device.lower() == "ascend":
        return UnfoldAscendFP32(ksizes, strides, rates, padding)
    else:
        return nn.Unfold(ksizes, strides, rates, padding)
