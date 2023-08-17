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
import mindspore
import mindspore.ops as ops
from mindspore.nn import LossBase
from mindspore.ops import operations as ops


class IPTPreTrainLoss(LossBase):
    """Combined loss"""

    def __init__(self, criterion, con_loss, use_con=True):
        super(IPTPreTrainLoss, self).__init__()
        self.use_con = use_con
        self.con_loss = con_loss
        self.criterion = criterion
        self.cast = ops.Cast()

    def construct(self, base, target):
        if self.use_con:
            sr, x_con = base
            hr = target
            x_con = self.cast(x_con, mindspore.float32)
            loss1 = self.criterion(sr, hr)
            loss2 = self.con_loss(x_con)
            loss = loss1 + 0.1 * loss2
        else:
            sr = base
            hr = target
            sr = self.cast(sr, mindspore.float32)
            loss = self.criterion(sr, hr)
        return self.get_loss(loss)
