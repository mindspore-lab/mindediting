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


class CtsdgLoss(nn.Cell):
    def __init__(self, g_loss, d_loss):
        super().__init__()
        self.g_loss = g_loss
        self.d_loss = d_loss

    def construct(self, net_output, ground_truth, mask, edge, gray_image):
        return self.g_loss(net_output, ground_truth, mask, edge, gray_image)

    def get(self, name):
        if name == "generator":
            return self.g_loss
        if name == "discriminator":
            return self.d_loss
