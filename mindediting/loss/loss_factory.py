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

""" loss factory """

from mindspore import nn

from .binary_cross_entropy_smooth import BinaryCrossEntropySmooth
from .charbonnier_loss import CharbonnierLoss
from .content_loss import ContentLoss
from .cross_entropy_smooth import CrossEntropySmooth
from .cstdg_discriminator_loss import CTSDGDiscriminatorLoss
from .cstdg_generator_loss import CTSDGGeneratorLoss
from .ctsdg_loss import CtsdgLoss
from .dists_loss import DISTSLoss
from .ifr_plus_loss import IFRPlusLoss
from .ipt_loss import IPTPreTrainLoss
from .psnr_loss import PSNRLoss
from .supervised_contrastive_loss import SupConLoss

__all__ = ["create_loss"]


def create_loss(
    loss_name: str = "CE",
    weight=1.0,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    aux_factor: float = 0.0,
    **kwargs
):
    r"""Creates loss function

    Args:
        loss_name (str):  loss name, : 'CE' for cross_entropy. 'BCE': binary cross entropy. Default: 'CE'.
        weight (Tensor): Class weight. Shape [C]. A rescaling weight applied to the loss of each batch element.
                Data type must be float16 or float32.
        reduction: Apply specific reduction method to the output: 'mean' or 'sum'. Default: 'mean'.
        label_smoothing: Label smoothing factor, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default: 0.0.
        aux_factor (float): Auxiliary loss factor. Set aux_fuactor > 0.0 if the model has auxilary logit outputs (i.e., deep supervision), like inception_v3.  Default: 0.0.

    Inputs:
        - logits (Tensor or Tuple of Tensor): Input logits. Shape [N, C], where N is # samples, C is # classes.
                Tuple of two input logits are supported in order (main_logits, aux_logits) for auxilary loss used in networks like inception_v3.
          where `C = number of classes`. Data type must be float16 or float32.
        - labels (Tensor): Ground truth labels. Shape: [N] or [N, C].
                (1) shape (N), sparse labels representing the class indinces. Must be int type,
                (2) shape [N, C], dense labels representing the ground truth class probability values, or the one-hot labels. Must be float type.
                If the loss type is BCE, the shape of labels must be [N, C].
    Returns:
       Loss function to compute the loss between the input logits and labels.

    """
    loss_name = loss_name.lower()

    if loss_name == "ce":
        loss = CrossEntropySmooth(smoothing=label_smoothing, aux_factor=aux_factor, reduction=reduction, weight=weight)
    elif loss_name == "bce":
        loss = BinaryCrossEntropySmooth(
            smoothing=label_smoothing, aux_factor=aux_factor, reduction=reduction, weight=weight, pos_weight=None
        )
    elif loss_name == "charbonnierloss":
        loss = CharbonnierLoss(reduction=reduction, weight=weight)
    elif loss_name == "supconloss":
        loss = SupConLoss()
    elif loss_name == "l1loss":
        loss = nn.L1Loss(reduction=reduction)
    elif loss_name == "mseloss":
        loss = nn.MSELoss(reduction=reduction)
    elif loss_name == "msssimloss":
        loss = nn.MSSSIM()
    elif loss_name == "distsloss":
        loss = DISTSLoss()
    elif loss_name == "ipt_pretrain_loss":
        loss = IPTPreTrainLoss(criterion=nn.L1Loss(reduction=reduction), con_loss=SupConLoss())
    elif loss_name == "content_loss":
        loss = ContentLoss()
    elif loss_name == "ctsdg_loss":
        g_loss = CTSDGGeneratorLoss(criterion=nn.BCELoss(reduction="mean"), **kwargs)
        d_loss = CTSDGDiscriminatorLoss(criterion=nn.BCELoss(reduction="mean"))
        loss = CtsdgLoss(g_loss, d_loss)
    elif loss_name == "ifrplusloss":
        loss = IFRPlusLoss(reduction=reduction, **kwargs)
    elif loss_name == "psnrloss":
        loss = PSNRLoss(weight=weight)
    else:
        raise NotImplementedError

    return loss
