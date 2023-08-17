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
import numpy as np
import pytest
from common import init_test_environment
from mindspore import Tensor
from mindspore.common.initializer import Normal
from mindspore.nn import TrainOneStepCell, WithLossCell

init_test_environment()

from mindediting.optim import create_optimizer


class SimpleCNN(nn.Cell):
    def __init__(self, num_classes=2, in_channels=1, include_top=True):
        super(SimpleCNN, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(in_channels, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(16 * 5 * 5, num_classes, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if self.include_top:
            x = self.flatten(x)
            x = self.fc(x)
        return x


@pytest.mark.parametrize("bs", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("opt", ["adam"])
def test_bs_adam_optimizer(opt, bs):
    network = SimpleCNN(num_classes=2)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    net_opt = create_optimizer(network.trainable_params(), opt, lr=0.01, weight_decay=1e-5)

    bs = bs
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(15):
        cur_loss = train_network(input_data, label)

    print(f"{opt}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    assert cur_loss < begin_loss, "Loss does NOT decrease"


@pytest.mark.parametrize("loss_scale", [0.1])
@pytest.mark.parametrize("weight_decay", [0.01])
@pytest.mark.parametrize("lr", [0.0001])
def test_lr_weight_decay_loss_scale_optimizer(lr, weight_decay, loss_scale):
    network = SimpleCNN(num_classes=2)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    net_opt = create_optimizer(
        network.trainable_params(), "adamW", lr=lr, weight_decay=weight_decay, loss_scale=loss_scale
    )

    bs = 16
    input_data = Tensor(np.ones([bs, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([bs]).astype(np.int32))

    net_with_loss = WithLossCell(network, net_loss)
    train_network = TrainOneStepCell(net_with_loss, net_opt)

    train_network.set_train()

    begin_loss = train_network(input_data, label)
    for i in range(15):
        cur_loss = train_network(input_data, label)

    print(f"{lr}, {weight_decay}, {loss_scale}, begin loss: {begin_loss}, end loss:  {cur_loss}")

    # check output correctness
    assert cur_loss < begin_loss, "Loss does NOT decrease"
