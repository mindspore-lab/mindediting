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

"""utils"""
import os
from pathlib import Path

import mindspore as ms
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.ops import functional as F

from mindediting.utils.local_adapter import get_rank_id


def check_args(cfg):
    """check args"""
    if cfg.file_format and cfg.file_format != "MINDIR":
        raise ValueError(f"Only MINDIR format is supported for export now, got {cfg.file_format}")

    if not Path(cfg.checkpoint_path).exists():
        raise FileExistsError(f"checkpoint_path {cfg.checkpoint_path} doesn`t exist.")

    if not Path(cfg.pretrained_vgg).exists():
        raise FileExistsError(f"pretrained vgg feature extractor {cfg.pretrained_vgg} " f"doesn`t exist.")

    if not isinstance(cfg.image_load_size, (tuple, list)):
        raise ValueError(f"config.image_load_size must be a list or a tuple!, " f"got {type(cfg.image_load_size)}")


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

    def get_indices(self, input_shape):
        """
        Map indices of spatial elements from the original feature map to the unfolded one.
        """

        x_batch, x_depth, x_row, x_col = input_shape
        # Zeroth index for padding.
        x_idx = F.tuple_to_array(range(1, x_row * x_col + 1))
        # Currently Unfold operation on Ascend does not support any input apart from fp16,
        # so using two fp16 variables to store one integer index without the loss of precision.
        base = 256  # <= 1024 should be good for fp16.
        x_idx = ops.stack(ms.numpy.divmod(x_idx, base, dtype=mstype.float16))
        x_idx = x_idx.reshape(2, 1, x_row, x_col)
        idx = self.extract_image_patches(x_idx)
        # Conver back to integer.
        idx = idx.astype(mstype.int32)
        idx = idx[0] * base + idx[1]
        # Make all valid indices zero-based and map gradients from padded values to the last element (-1) of dx.
        idx -= 1
        return idx

    def construct(self, input_x):
        x_shape = input_x.shape
        x_batch, x_depth, x_row, x_col = x_shape
        indices = self.get_indices(x_shape)
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
        idx = self.get_indices(x_shape)
        idx = idx.reshape(-1, 1)

        _, _, y_row, y_col = dout.shape
        grad = dout.reshape(x_batch, self.ksizes[2], self.ksizes[3], x_depth, y_row, y_col).transpose(1, 2, 4, 5, 0, 3)
        grad = grad.reshape(-1, x_batch * x_depth)

        # The last element is to collect gradients from padded values. Has to be dropped afterwards.
        dx = ops.Zeros()((x_row * x_col + 1, x_batch * x_depth), mstype.float32)
        dx = ops.tensor_scatter_add(dx, idx, grad)
        # Remove the last slice as it containes gradients for padded values.
        dx = dx[:-1]
        dx = dx.reshape(x_row, x_col, x_batch, x_depth)
        dx = dx.transpose(2, 3, 0, 1)
        return (dx,)


def extract_patches(inp, ksize, stride=1, pad=1, dilation=1):
    """unfold function"""
    batch_num, channel, height, width = inp.shape
    out_h = (height + pad + pad - ksize - (ksize - 1) * (dilation - 1)) // stride + 1
    out_w = (width + pad + pad - ksize - (ksize - 1) * (dilation - 1)) // stride + 1

    inp = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))(inp)
    res = ops.Zeros()((batch_num, channel, ksize, ksize, out_h, out_w), mstype.float32)

    for y in range(ksize):
        y_max = y + stride * out_h
        for x in range(ksize):
            x_max = x + stride * out_w
            res[:, :, y, x, :, :] = inp[:, :, y:y_max:stride, x:x_max:stride]

    res = res.transpose(0, 4, 5, 1, 2, 3)
    return res


def extract_patches_ascend(inp, ksize, stride=1, pad=1, dilation=1):
    """unfold function"""
    batch_num, channel, height, width = inp.shape
    out_h = (height + pad + pad - ksize - (ksize - 1) * (dilation - 1)) // stride + 1
    out_w = (width + pad + pad - ksize - (ksize - 1) * (dilation - 1)) // stride + 1

    inp = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))(inp)
    res = ops.Zeros()((ksize, ksize, batch_num, channel, out_h, out_w), inp.dtype)

    for y in range(ksize):
        y_max = y + stride * out_h
        for x in range(ksize):
            x_max = x + stride * out_w
            res[y, x] = inp[:, :, y:y_max:stride, x:x_max:stride]

    # batch_num, out_h, out_w, channel, ksize, ksize,
    res = res.transpose(2, 4, 5, 3, 0, 1)
    return res


def extract_patches_ascend_opt(inp, ksize, stride=1, pad=1, dilation=1):
    """unfold function optimized for Ascend"""

    inp = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))(inp)
    unfold_op = UnfoldAscendFP32(
        ksizes=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], rates=[1, dilation, dilation, 1]
    )
    patches = unfold_op(inp)
    batch_num, channel, height, width = patches.shape
    return patches.reshape(batch_num, ksize, ksize, -1, height, width).transpose(0, 4, 5, 3, 1, 2)


def extract_patches_ascend_fp16(inp, ksize, stride=1, pad=1, dilation=1):
    """unfold function optimized for Ascend"""

    inp = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))(inp)
    unfold_op = nn.Unfold(ksizes=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], rates=[1, dilation, dilation, 1])
    patches = unfold_op(inp)
    batch_num, channel, height, width = patches.shape
    return patches.reshape(batch_num, ksize, ksize, -1, height, width).transpose(0, 4, 5, 3, 1, 2)


def cast_module(type_str, module):
    """
    Cast module (params and weights) and loss to new type (float16 or float32).
    """
    if type_str == "float16":
        module.to_float(ms.float16)
    elif type_str == "float32":
        module.to_float(ms.float32)
    elif type_str == "base":
        pass
    else:
        print(f"The module does not support conversion to type {type_str}. The default value will be used.")


def change_dict_name(origin: dict) -> dict:
    new = {}
    for key in origin:
        if ".gamma" in key:
            new_key = key.replace(".gamma", ".layernorm.gamma")
            new[new_key] = origin[key]
            continue
        if ".beta" in key:
            new_key = key.replace(".beta", ".layernorm.beta")
            new[new_key] = origin[key]
            continue
        new[key] = origin[key]
    return new


def check_if_mirrored(pipline):
    mirrored = False
    for p in pipline:
        if "MirrorSequence" in p.keys():
            mirrored = True
            break
    return mirrored


def is_ascend():
    return ms.get_context("device_target").lower() == "ascend"


def current_device():
    return context.get_context("device_target").lower()


def is_gpu():
    return current_device() == "gpu"


def check_paths(cfg):
    """
    Checks for important paths from the configuration.
    """
    paths = [cfg.dataset.input_path, cfg.model.load_path]
    for path in paths:
        if not isinstance(path, list):
            path = [path]
        for p in path:
            if p and not os.path.exists(p):
                raise FileNotFoundError(f'Path "{p}" wasn\'t found.')
    return True


def get_rank_mentioned_filename(path):
    path = path.split(".")
    return ".".join(path[:-1] + [f"rank_{get_rank_id()}"] + path[-1:])
