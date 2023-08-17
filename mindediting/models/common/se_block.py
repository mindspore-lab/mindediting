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


import mindspore.ops as ops
from mindspore import nn


class SEBlock(nn.Cell):
    def __init__(self, input_dim, reduction=16):
        super().__init__()

        self.pool = ops.ReduceMean(keep_dims=True)

        mid = input_dim // reduction
        self.conv = nn.SequentialCell(
            nn.Conv2d(input_dim, mid, kernel_size=1, pad_mode="pad", padding=0, has_bias=True),
            nn.ReLU(),
            nn.Conv2d(mid, input_dim, kernel_size=1, pad_mode="pad", padding=0, has_bias=True),
            nn.Sigmoid(),
        )

    def construct(self, x):
        y = self.pool(x, (2, 3))
        y = self.conv(y)
        y = x * y

        return y
