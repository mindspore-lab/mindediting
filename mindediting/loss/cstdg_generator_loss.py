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

"""define network with loss function and train by one step"""
from mindspore import Tensor, dtype, nn, ops

from mindediting.utils.init_utils import get_current_device


def gram_matrix(feat):
    """gram matrix"""
    b, ch, h, w = feat.shape
    feat = feat.view(b, ch, h * w)
    gram = ops.BatchMatMul(False, True)(feat, feat) / (ch * h * w)
    return gram


class GramMat(nn.Cell):
    def construct(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w) / ops.sqrt(Tensor(c * h * w))
        gram = ops.BatchMatMul(False, True)(x, x)
        return gram


class CTSDGGeneratorLoss(nn.Cell):
    """Generator Loss"""

    def __init__(
        self,
        hole_loss_w,
        valid_loss_w,
        perceptual_loss_w,
        style_loss_w,
        adversarial_loss_w,
        intermediate_loss_w,
        criterion=nn.BCELoss(reduction="mean"),
        **kwargs
    ):
        super(CTSDGGeneratorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.criterion = criterion
        if get_current_device() == "ascend":
            self.gram_matrix = GramMat().to_float(dtype.float16)
        else:
            self.gram_matrix = gram_matrix
        self.hole_loss_w = hole_loss_w
        self.valid_loss_w = valid_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.style_loss_w = style_loss_w
        self.adversarial_loss_w = adversarial_loss_w
        self.intermediate_loss_w = intermediate_loss_w

    def construct(self, net_output, ground_truth, mask, edge, gray_image):
        output, projected_image, projected_edge = net_output.get("generator")
        output_pred, output_edge = net_output.get("discriminator")
        vgg_comp, vgg_output, vgg_ground_truth = net_output.get("vgg_feat_extractor")

        loss_hole = self.l1((1 - mask) * output, (1 - mask) * ground_truth)

        loss_valid = self.l1(mask * output, mask * ground_truth)
        loss_perceptual = 0.0
        for i in range(3):
            loss_perceptual += self.l1(vgg_output[i], vgg_ground_truth[i])
            loss_perceptual += self.l1(vgg_comp[i], vgg_ground_truth[i])

        loss_style = 0.0
        for i in range(3):
            mats = ops.Concat(axis=0)((vgg_ground_truth[i], vgg_output[i], vgg_comp[i]))
            gram = self.gram_matrix(mats)
            gram_gt, gram_out, gram_comp = ops.Split(axis=0, output_num=3)(gram)
            loss_style += self.l1(gram_out, gram_gt)
            loss_style += self.l1(gram_comp, gram_gt)

        real_target = self.real_target.expand_as(output_pred)
        loss_adversarial = self.criterion(output_pred, real_target) + self.criterion(output_edge, edge)

        loss_intermediate = self.criterion(projected_edge, edge) + self.l1(projected_image, ground_truth)

        loss_g = (
            loss_hole.mean() * self.hole_loss_w
            + loss_valid.mean() * self.valid_loss_w
            + loss_perceptual.mean() * self.perceptual_loss_w
            + loss_style.mean() * self.style_loss_w
            + loss_adversarial.mean() * self.adversarial_loss_w
            + loss_intermediate.mean() * self.intermediate_loss_w
        )

        result = ops.stop_gradient(output)
        return loss_g, result
