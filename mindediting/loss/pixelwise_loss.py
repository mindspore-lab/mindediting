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


from abc import ABCMeta, abstractmethod

import mindspore.nn as nn
import mindspore.ops as ops


class BasePixelwiseLoss(nn.Cell):
    """Base class for Pixel-wise losses.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    __metaclass__ = ABCMeta

    def __init__(self, loss_weight=1.0, filter_factor=None, filtered_weight=0.1):
        super().__init__()

        self.loss_weight = loss_weight
        self.filter_factor = filter_factor
        self.filtered_weight = filtered_weight

        self.reduce_mean = ops.ReduceMean()
        self.top_k = ops.TopK(sorted=True)

    @abstractmethod
    def calculate_elementwise(self, pred, target):
        """Abstract method for calculating loss.

        All subclass should overwrite it.
        """

        return

    def filter_losses(self, losses):
        t, _, h, w = losses.shape[1:]
        target_size = int(h * w)

        spatial_losses = ops.mean(losses, axis=2)
        out_losses, _ = self.top_k(
            spatial_losses.view(-1, target_size), max(1, int(self.filter_factor * float(target_size)))
        )

        return out_losses

    def construct(self, *args):
        """Construct function."""

        losses = self.calculate_elementwise(*args)

        if self.filter_factor is not None:
            filtered_losses = self.filter_losses(losses)

            filtered_loss = self.reduce_mean(filtered_losses)
            main_loss = self.reduce_mean(losses)

            loss = (1.0 - self.filtered_weight) * main_loss + self.filtered_weight * filtered_loss
        else:
            loss = self.reduce_mean(losses)

        return self.loss_weight * loss


class L1Loss(BasePixelwiseLoss):
    """L1 (mean absolute error, MAE) loss."""

    def calculate_elementwise(self, pred, target):
        """Calculate function.

        Args:
            pred [Tensor]: of shape (N, C, H, W). Predicted tensor.
            target [Tensor]: of shape (N, C, H, W). Ground truth tensor.
            weight [Tensor, optional]: of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """

        return ops.abs(pred - target)


class MSELoss(BasePixelwiseLoss):
    """MSE (L2) loss."""

    def calculate_elementwise(self, pred, target):
        """Calculate function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """

        return ops.square(pred - target)


class CharbonnierLoss(BasePixelwiseLoss):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight [float]: Loss weight for L1 loss. Default: 1.0.
        reduction [str]: Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise [bool]: Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps [float]: A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def calculate_elementwise(self, pred, target):
        """Calculate function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """

        return ops.sqrt(ops.square(pred - target) + 1e-12)
