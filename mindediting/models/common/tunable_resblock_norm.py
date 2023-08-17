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

import mindspore
import mindspore.nn as nn
import numpy as np
from mindspore import Parameter, Tensor, ops
from mindspore.common import initializer

from mindediting.models.common.tunable_conv import TunableConv2d
from mindediting.utils.utils import is_ascend


class InstanceNorm2d(nn.Cell):
    """myown InstanceNorm2d"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, gamma_init="ones", beta_init="zeros"):
        super().__init__()
        self.num_features = num_features
        self.moving_mean = Parameter(initializer.initializer("zeros", num_features), name="mean", requires_grad=False)
        self.moving_variance = Parameter(
            initializer.initializer("ones", num_features), name="variance", requires_grad=False
        )
        self.gamma = Parameter(initializer.initializer(gamma_init, num_features), name="gamma", requires_grad=affine)
        self.beta = Parameter(initializer.initializer(beta_init, num_features), name="beta", requires_grad=affine)
        self.sqrt = ops.Sqrt()
        self.eps = Tensor(np.array([eps]), mindspore.float32)
        self.cast = ops.Cast()

    def construct(self, x):
        """calculate InstanceNorm output"""
        mean = ops.ReduceMean(keep_dims=True)(x, (2, 3))
        mean = self.cast(mean, mindspore.float32)
        tmp = x - mean
        tmp = tmp * tmp
        var = ops.ReduceMean(keep_dims=True)(tmp, (2, 3))
        std = self.sqrt(var + self.eps)
        gamma_t = self.cast(self.gamma, mindspore.float32)
        beta_t = self.cast(self.beta, mindspore.float32)
        x = (x - mean) / std * gamma_t.reshape(1, -1, 1, 1) + beta_t.reshape(1, -1, 1, 1)
        return x


class TunableResBlockNorm(nn.Cell):
    def __init__(self, num_channels, kernel_size, num_params=1, expand_params=1, mode="mlp"):
        super(TunableResBlockNorm, self).__init__()
        self.is_ascend = is_ascend()
        self.conv1 = TunableConv2d(
            num_channels,
            num_channels,
            kernel_size,
            has_bias=True,
            num_params=num_params,
            expand_params=expand_params,
            mode=mode,
        )
        self.act1 = nn.ReLU()
        self.conv2 = TunableConv2d(
            num_channels,
            num_channels,
            kernel_size,
            has_bias=True,
            num_params=num_params,
            expand_params=expand_params,
            mode=mode,
        )
        self.act2 = nn.ReLU()
        if self.is_ascend:
            self.norm1 = InstanceNorm2d(num_channels, affine=True)
            self.norm2 = InstanceNorm2d(num_channels, affine=True)
        else:
            self.norm1 = nn.InstanceNorm2d(num_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(num_channels, affine=True)

    def construct(self, x, px):
        y = self.conv1(x, px)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y, px)
        y = self.norm2(y)
        y = x + y
        y = self.act2(y)
        return y
