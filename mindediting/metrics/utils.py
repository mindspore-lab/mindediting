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
import numpy as np


def tensor2img(tensor, input_order="CHW", out_type=np.uint8, min_max=(0, 1), swap_red_blue=True):
    """Convert Tensors into image numpy arrays.

    After clamping to (min, max), image values will be normalized to [0, 1].

    For different tensor shapes, this function will have different behaviors:

        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.

    Note that the image channel in input tensors should be RGB order. This
    function will convert it to cv2 convention, i.e., (H x W x C) with BGR order.

    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        input_order (str type): Input CHW or HWC.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.
        swap_red_blue: Image rgb to bgr.

    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    if not isinstance(tensor, list):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.clip(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.ndim
        if n_dim == 4:
            if input_order == "CHW":
                img_np = np.transpose(_tensor, (0, 2, 3, 1))
            else:
                img_np = _tensor
            if swap_red_blue and img_np.shape[-1] == 3:
                img_np = np.flip(img_np, axis=-1)
        elif n_dim == 3:
            if input_order == "CHW":
                img_np = np.transpose(_tensor, (1, 2, 0))
            else:
                img_np = _tensor
            if swap_red_blue and img_np.shape[-1] == 3:
                img_np = np.flip(img_np, axis=-1)
        elif n_dim == 2:
            img_np = _tensor
        else:
            raise ValueError(f"Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}")
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result


class TensorSyncer(nn.Cell):
    """sync metric values from all mindspore-processes"""

    def __init__(self, _type="sum"):
        super(TensorSyncer, self).__init__()
        self.ops = None
        self._type = _type.lower()
        if self._type == "sum":
            self.ops = ms.ops.AllReduce(ms.ops.ReduceOp.SUM)
        elif self._type == "gather":
            self.ops = ms.ops.AllGather()
        else:
            raise ValueError(f"TensorSyncer._type == {self._type} is not support")

    def construct(self, x):
        return self.ops(x)


def check_if_needed_middle_frame_only(dataset_name, pipeline):
    middle_only, mirrored = False, False
    if dataset_name == "vimeo_super_resolution":
        middle_only = True
        for p in pipeline:
            if "MirrorSequence" in p.cfg_dict.keys():
                mirrored = True
                break
    return middle_only, mirrored
