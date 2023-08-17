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
from mindspore import Tensor, dtype, nn


class CTSDGDiscriminatorLoss(nn.Cell):
    """Combined loss"""

    def __init__(self, criterion=nn.BCELoss(reduction="mean")):
        super(CTSDGDiscriminatorLoss, self).__init__()
        self.criterion = criterion
        self.real_target = Tensor(1.0, dtype.float32)
        self.fake_target = Tensor(0.0, dtype.float32)

    def construct(self, net_output, ground_truth, mask, edge, gray_image):
        real_pred, real_pred_edge = net_output.get("ground_truth")
        fake_pred, fake_pred_edge = net_output.get("fake")

        real_target = self.real_target.expand_as(real_pred)
        fake_target = self.fake_target.expand_as(fake_pred)

        loss_adversarial = (
            self.criterion(real_pred, real_target)
            + self.criterion(fake_pred, fake_target)
            + self.criterion(real_pred_edge, edge)
            + self.criterion(fake_pred_edge, edge)
        )
        return loss_adversarial
