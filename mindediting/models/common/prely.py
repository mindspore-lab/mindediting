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
import numpy as np
from mindspore import nn
from mindspore.ops import operations as P


class PReLU_PT(nn.Cell):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()

        w = np.empty((num_parameters,), dtype=np.float32)
        w.fill(init)
        self.weight = ms.Parameter(ms.Tensor(w, dtype=ms.float32), name="weight")

        self.prelu = P.PReLU()
        self.relu = P.ReLU()
        self.assign = P.Assign()
        self.cast = P.Cast()

    def construct(self, x):
        u = self.relu(self.weight)
        v = self.prelu(x, self.cast(u, x.dtype))

        if self.training:
            self.assign(self.weight, u)

        return v
