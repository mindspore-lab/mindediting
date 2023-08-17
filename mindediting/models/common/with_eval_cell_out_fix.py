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
from mindspore import Tensor
from mindspore import numpy as mnp
from mindspore.common import dtype as mstype
from mindspore.ops import Concat
from mindspore.ops import functional as F


class WithEvalCellOutFix(nn.WithEvalCell):
    r"""
    Wraps the forward network with the loss function.

    Added fix for converting network outputs to float32.
    """

    def construct(self, data, label):
        outputs = self._network(data)
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)

            outputs_cast = outputs
            for i, output in enumerate(outputs):
                outputs_cast[i] = F.cast(output, mstype.float32)
            outputs = outputs_cast
        return outputs[0], label  # for mindediting psnr, ssim


class WithEvalCellOutFix_Ctsdg(nn.WithEvalCell):
    r"""
    Wraps the forward network with the loss function.

    Added fix for converting network outputs to float32.
    """

    def postprocess(self, x: Tensor) -> Tensor:
        """
        Map tensor values from [-1, 1] to [0, 1]

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        x = (x + 1.0) * 0.5
        x = mnp.clip(x, 0, 1)
        return x

    def construct(self, ground_truth, mask, edge, gray_image):
        input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask
        outputs, _, _ = self._network(input_image, Concat(axis=1)((input_edge, input_gray_image)), mask)

        output_comp = ground_truth * mask + outputs * (1 - mask)
        output_comp = self.postprocess(output_comp)
        ground_truth_post = self.postprocess(ground_truth)
        return output_comp, ground_truth_post  # for mindediting psnr, ssim


class WithEvalCellNoLoss(nn.WithEvalCell):
    def __init__(self, network, add_cast_fp32=False):
        loss_fn = lambda x, y: None
        super().__init__(network, loss_fn, add_cast_fp32)

    def construct(self, data, label=None):
        output, label = super().construct(data, label)[1:]
        if label is None:
            return output
        return output, label
