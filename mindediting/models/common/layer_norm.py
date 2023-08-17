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
from mindspore import nn
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P

from mindediting.utils.utils import is_ascend


class LayerNorm2dDefault(nn.Cell):
    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.weight = ms.Parameter(initializer("ones", (channels,)), name="weight", requires_grad=True)
        self.bias = ms.Parameter(initializer("zeros", (channels,)), name="bias", requires_grad=True)
        self.eps = eps

        self.forward_transpose = P.Transpose()
        self.backward_transpose = P.Transpose()
        self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=self.eps)

    def construct(self, x):
        y = self.forward_transpose(x, (0, 2, 3, 1))
        y, _, _ = self.norm(y, self.weight, self.bias)
        y = self.backward_transpose(y, (0, 3, 1, 2))

        return y


class LayerNorm2dAscend(nn.Cell):
    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.weight = ms.Parameter(initializer("ones", (channels,)), name="weight", requires_grad=True)
        self.bias = ms.Parameter(initializer("zeros", (channels,)), name="bias", requires_grad=True)
        self.eps = eps

        self.mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()

    def construct(self, x):
        mu = self.mean(x, 1)
        x_centered = x - mu

        var = self.mean(self.square(x_centered), 1)
        y = x_centered / self.sqrt(var + self.eps)
        y = self.weight.reshape(1, -1, 1, 1) * y + self.bias.reshape(1, -1, 1, 1)

        return y


def build_layer_norm(channels, eps=1e-6, export_mode=True):
    if export_mode or not is_ascend():
        return LayerNorm2dDefault(channels, eps)
    else:
        return LayerNorm2dAscend(channels, eps)
