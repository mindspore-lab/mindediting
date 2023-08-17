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

import mindspore as ms
import mindspore.ops as ops
from mindspore.nn import LossBase

from mindediting.utils.utils import is_ascend


@ops.constexpr
def _get_hann_filter(h, w, dtype):
    window_y = _hann_window(h, dtype)
    window_x = _hann_window(w, dtype)

    window_2d = ops.sqrt(ops.outer(window_y, window_x))
    window_2d = window_2d.reshape(1, 1, h, w)

    return window_2d


def _hann_window(window_length, dtype):
    n = ms.numpy.arange(window_length, dtype=dtype)
    w = 0.5 - 0.5 * ops.cos((2.0 * math.pi / (window_length - 1)) * n)

    return w


class FrequencyLoss(LossBase):
    """The class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Original repo: <https://github.com/EndlessSora/focal-frequency-loss>
    """

    def __init__(self, weight=1.0, reduction="mean", hann_filter=True, **kwargs):
        super(FrequencyLoss, self).__init__(reduction=reduction)

        self.weight = weight
        self.hann_filter = hann_filter

        self.fft2 = ops.FFTWithSize(signal_ndim=2, inverse=False, real=True, norm="ortho", onesided=False)
        self.real = ops.Real()
        self.imag = ops.Imag()

        self.cast = ms.ops.Cast()
        self.is_ascend = is_ascend()

        if self.is_ascend:
            self.fft2.set_device("CPU")

    def _tensor_to_freq(self, x):
        freq = self.fft2(x)
        out = ops.stack([self.real(freq), self.imag(freq)], -1)

        return out

    def construct(self, pred, target):
        if self.is_ascend:
            pred = self.cast(pred, ms.float32)
            target = self.cast(target, ms.float32)

        if self.hann_filter:
            hann_filter = _get_hann_filter(pred.shape[2], pred.shape[3], pred.dtype)
            pred = pred * hann_filter
            target = target * hann_filter

        pred_freq = self._tensor_to_freq(pred)
        target_freq = self._tensor_to_freq(target)

        losses = ops.abs(pred_freq - target_freq)
        losses = losses.sum(-1)

        return self.get_loss(losses, self.weight)
