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
import mindspore as ms
import mindspore.ops as ops
import numpy as np
import pytest
from mindspore import nn
from mindspore.common.initializer import initializer

from mindediting import is_ascend
from mindediting.models.common.tunable_conv import TunableConv2d, TunableParameter

ms.set_seed(1)
np.random.seed(1)


@pytest.mark.parametrize("num_params", [2, 3])
@pytest.mark.parametrize("default_input", [initializer(init="uniform", shape=(1, 1, 1), dtype=ms.float32)])
def test_tunable_parameter(default_input, num_params):
    assert num_params > 1
    batch_size = num_params
    gamma = TunableParameter(
        default_input=default_input, name="gamma", requires_grad=True, num_params=num_params, mode="linear"
    )
    px = ops.eye(num_params, num_params, ms.float32)
    g = gamma(px)
    print(gamma)
    assert g.shape == (batch_size, *default_input.shape)


@pytest.mark.parametrize("num_params", [3])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("group", [1])
def test_tunable_conv2d(num_params, kernel_size, stride, group):

    assert num_params > 1
    mse = nn.MSE()
    batch_size = num_params
    b, c, h, w, d = batch_size, 16, 24, 24, 32
    x = ops.normal(shape=(b, c, h, w), mean=ms.Tensor(0.0), stddev=ms.Tensor(1.0))
    px = ops.eye(num_params, num_params, ms.float32)

    tunable_conv = TunableConv2d(
        c, d, kernel_size, stride=stride, group=group, has_bias=True, num_params=num_params, mode="linear"
    )
    conv = nn.Conv2d(c, d, kernel_size, stride=stride, group=group, has_bias=True)
    print(tunable_conv)
    y = tunable_conv(x, px)
    for p in range(num_params):
        conv.weight = tunable_conv.weight[0, p, ...]
        conv.bias = tunable_conv.bias[0, p, ...]
        y_p = conv(x[p : p + 1, ...])
        if is_ascend():
            assert mse(y[p : p + 1, ...], y_p) < 1e-4
        else:
            assert mse(y[p : p + 1, ...], y_p) < 1e-6
