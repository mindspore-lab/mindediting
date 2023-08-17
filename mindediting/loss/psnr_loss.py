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


class PSNRLoss(nn.Cell):
    def __init__(self, weight=1.0):
        super(PSNRLoss, self).__init__()
        self.loss_weight = weight
        self.log = ops.Log()
        self.scale = 10 / self.log(ms.Tensor(10, dtype=ms.float32))
        self.coef = ms.Tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)

    def construct(self, pred, target):
        result = self.log(((pred - target) ** 2).mean(axis=(1, 2, 3)) + 1e-8).mean()
        return self.loss_weight * self.scale * result
