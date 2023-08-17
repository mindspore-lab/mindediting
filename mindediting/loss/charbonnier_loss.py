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
import mindspore.ops as ops

from mindediting.utils.utils import is_ascend


class CharbonnierLoss(nn.Cell):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant of L1Loss).
    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution".

    Args:
        loss_weight [float]: Loss weight for L1 loss. Default: 1.0.
        reduction [str]: Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps [float]: A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, weight=1.0, reduction="mean", eps=1e-12, filter_factor=None, filtered_weight=0.1):
        super(CharbonnierLoss, self).__init__()

        assert reduction in ["mean"]

        self.weight = weight
        self.eps = eps
        self.filter_factor = filter_factor
        self.filtered_weight = filtered_weight

        self.sqrt = ops.Sqrt()
        self.cast = ops.Cast()
        self.is_ascend = is_ascend()
        self.reduce_mean = ops.ReduceMean()
        self.top_k = ops.TopK(sorted=True)

    def filter_losses(self, losses):
        h, w = losses.shape[-2:]
        target_size = int(h * w)

        num_elements = max(1, int(self.filter_factor * float(target_size)))

        if self.is_ascend:
            scores = ops.stop_gradient(losses.reshape(-1, target_size))
            largest_losses, _ = self.top_k(scores, num_elements)
            thresholds = largest_losses[:, -1].reshape(-1, 1, 1)
            out_losses = ops.where(losses > thresholds, losses, ops.zeros_like(losses))
        else:
            out_losses, _ = self.top_k(losses.reshape(-1, target_size), num_elements)

        return out_losses

    def calculate_elementwise(self, pred, target, *args, **kwargs):
        return self.sqrt((pred - target) ** 2 + self.eps)

    def construct(self, pred, target, *args, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """

        if self.is_ascend:
            pred = self.cast(pred, ms.float32)
            target = self.cast(target, ms.float32)

        losses = self.calculate_elementwise(pred, target, *args, **kwargs)

        if self.filter_factor is not None:
            spatial_losses = ms.ops.mean(losses, axis=1)
            filtered_losses = self.filter_losses(spatial_losses)

            filtered_loss = self.reduce_mean(filtered_losses)
            main_loss = self.reduce_mean(losses)

            loss = self.weight * main_loss + self.filtered_weight * filtered_loss
        else:
            loss = self.weight * self.reduce_mean(losses)

        return loss


class GeneralizedCharbonnierLoss(CharbonnierLoss):
    def calculate_elementwise(self, pred, target, robust_weight, *args, **kwargs):
        if self.is_ascend:
            robust_weight = self.cast(robust_weight, ms.float32)

        alpha = 0.5 * robust_weight
        epsilon = 10.0 ** (-(10.0 * robust_weight - 1.0) / 3.0)
        losses = ((pred - target) ** 2 + epsilon**2) ** alpha

        return losses
