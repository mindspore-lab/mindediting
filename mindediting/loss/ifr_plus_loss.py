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

from mindediting.models.common.resize import BilinearResize
from mindediting.utils.utils import is_ascend

from .charbonnier_loss import CharbonnierLoss, GeneralizedCharbonnierLoss
from .frequency_loss import FrequencyLoss
from .lpips_loss import LPIPSLoss


def get_robust_weight(diff, beta):
    diff = ops.stop_gradient(diff)
    epe = ops.sqrt(ops.sum(ops.square(diff), 1, True))
    robust_weight = ops.exp((-beta) * epe)

    return robust_weight


class IFRPlusLoss(nn.Cell):
    def __init__(
        self,
        pixel_weight=1.0,
        freq_weight=1.0,
        iqa_weight=0.1,
        flow_weight=0.1,
        lpips_pretrained=None,
        vgg_pretrained=None,
        reduction="mean",
        **kwargs
    ):
        super().__init__()

        self.is_ascend = is_ascend()

        self.pixel_loss = CharbonnierLoss(weight=pixel_weight, reduction=reduction, eps=1e-6, filter_factor=0.2)
        self.frequency_loss = FrequencyLoss(weight=freq_weight, reduction=reduction)
        self.iqa_loss = LPIPSLoss(weight=iqa_weight, load_weights_path=lpips_pretrained, vgg_pretrained=vgg_pretrained)
        self.flow_loss = GeneralizedCharbonnierLoss(weight=flow_weight, reduction=reduction)

        self.downscale = BilinearResize(scale_factor=0.5, align_corners=False)

    def _pixel_loss(self, pred, target):
        loss_pixel = self.pixel_loss(pred, target)

        return loss_pixel

    def _frequency_loss(self, pred, target):
        if not self.is_ascend:
            loss_freq = self.frequency_loss(pred, target)
        else:
            loss_freq = 0.0

        return loss_freq

    def _iqa_loss(self, pred, target):
        loss_iqa = self.iqa_loss(pred, target)

        return loss_iqa

    def _flow_loss(self, flows_0_pred, flows_1_pred, flow_0_gt, flow_1_gt):
        up_flow0_1, up_flow0_2, up_flow0_3, up_flow0_4 = flows_0_pred
        up_flow1_1, up_flow1_2, up_flow1_3, up_flow1_4 = flows_1_pred

        robust_weight0 = get_robust_weight(up_flow0_1 - flow_0_gt, beta=0.3)
        robust_weight1 = get_robust_weight(up_flow1_1 - flow_1_gt, beta=0.3)

        flow0_gt_scaled, flow1_gt_scaled = [flow_0_gt], [flow_1_gt]
        robust_weight0_scaled, robust_weight1_scaled = [robust_weight0], [robust_weight1]
        for _ in range(3):
            flow0_gt_scaled.append(0.5 * self.downscale(flow0_gt_scaled[-1]))
            flow1_gt_scaled.append(0.5 * self.downscale(flow1_gt_scaled[-1]))
            robust_weight0_scaled.append(self.downscale(robust_weight0_scaled[-1]))
            robust_weight1_scaled.append(self.downscale(robust_weight1_scaled[-1]))

        flow01_loss = self.flow_loss(up_flow0_1, flow0_gt_scaled[0], robust_weight0_scaled[0])
        flow11_loss = self.flow_loss(up_flow1_1, flow1_gt_scaled[0], robust_weight1_scaled[0])
        flow02_loss = self.flow_loss(up_flow0_2, flow0_gt_scaled[1], robust_weight0_scaled[1])
        flow12_loss = self.flow_loss(up_flow1_2, flow1_gt_scaled[1], robust_weight1_scaled[1])
        flow03_loss = self.flow_loss(up_flow0_3, flow0_gt_scaled[2], robust_weight0_scaled[2])
        flow13_loss = self.flow_loss(up_flow1_3, flow1_gt_scaled[2], robust_weight1_scaled[2])
        flow04_loss = self.flow_loss(up_flow0_4, flow0_gt_scaled[3], robust_weight0_scaled[3])
        flow14_loss = self.flow_loss(up_flow1_4, flow1_gt_scaled[3], robust_weight1_scaled[3])

        flow_losses = [
            0.5 * (flow01_loss + flow11_loss),
            0.5 * (flow02_loss + flow12_loss),
            0.5 * (flow03_loss + flow13_loss),
            0.5 * (flow04_loss + flow14_loss),
        ]
        loss_flow = sum([(0.9 ** (len(flow_losses) - li - 1)) * l for li, l in enumerate(flow_losses)])

        return loss_flow

    def construct(self, net_output, gt_data):
        img_pred, flows_0_pred, flows_1_pred = net_output
        flow_0_gt, flow_1_gt, target = gt_data[:, 0:2], gt_data[:, 2:4], gt_data[:, 4:]

        loss_pixel = self._pixel_loss(img_pred, target)
        loss_freq = self._frequency_loss(img_pred, target)
        loss_iqa = self._iqa_loss(img_pred, target)
        loss_flow = self._flow_loss(flows_0_pred, flows_1_pred, flow_0_gt, flow_1_gt)

        loss = loss_pixel + loss_freq + loss_iqa + loss_flow

        return loss
