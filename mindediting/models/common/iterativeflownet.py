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

import os
import sys
import warnings

import mindspore
from mindspore import nn
from mindspore.common.initializer import HeNormal, initializer

from mindediting.models.common.convblock import ConvBlock

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, ".."))


def initialize_weights(net_l, scale=1):
    for net in net_l:
        for m in net.cells():
            if m == "Conv2d" or m == "Conv2dTranspose":
                initializer(HeNormal(negative_slope=0, mode="fan_in"), m.weight, mindspore.float32)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()


class Block(nn.Cell):
    def __init__(self):
        super(Block, self).__init__()

    def activation_fn(self, activation):
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "lrelu":
            act = nn.LeakyReLU(alpha=0.2)
        elif activation == "prelu":
            act = nn.PReLU(channel=1)
        elif activation is None:
            act = None
        else:
            raise NotImplementedError()
        return act


class ResBlock(Block):
    """
    Residual block
    ---Conv-Act-Conv-+-
     |________________|

    Residual block with batch norm
    ---Conv-BN-Act-Conv-BN-+-
     |_____________________|
    """

    def __init__(
        self,
        num_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        bias=True,
        with_batch_norm=False,
        activation="relu",
        residual_scale=1.0,
        self_shortcut=False,
        is_adder=False,
    ):
        super(ResBlock, self).__init__()
        if activation is None:
            warnings.warn("Activation should not be None in ResBlock.")
        if self_shortcut and not is_adder:
            warnings.warn("Self-shortcut is enabled but ResBlock does not use Adder2d.")
        if is_adder:
            # batch norm is necessary for adder conv
            with_batch_norm = True
        if with_batch_norm:
            # No need for bias if batch norm is enabled
            bias = False
        Conv2d = nn.Conv2d

        self.self_shortcut = self_shortcut
        self.residual_scale = residual_scale
        self.conv1 = Conv2d(
            num_channels,
            num_channels,
            kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            group=groups,
            has_bias=bias,
        )
        self.batch_norm1 = None
        if with_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.act = self.activation_fn(activation)
        self.conv2 = Conv2d(
            num_channels,
            num_channels,
            kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            group=groups,
            has_bias=bias,
        )
        self.batch_norm2 = None
        if with_batch_norm:
            self.batch_norm2 = nn.BatchNorm2d(num_channels)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def construct(self, x):
        out1 = self.conv1(x)
        if self.self_shortcut:
            out1 += x
        if self.batch_norm1 is not None:
            out1 = self.batch_norm1(out1)
        if self.act is not None:
            out1 = self.act(out1)
        out2 = self.conv2(out1)
        if self.self_shortcut:
            out2 += out1
        if self.batch_norm2 is not None:
            out2 = self.batch_norm2(out2)
        return x + out2 * self.residual_scale


class FeatureEncoder(nn.Cell):
    def __init__(self, in_channels, num_filters, down_factor, activation="relu"):
        super(FeatureEncoder, self).__init__()

        assert down_factor in [2, 4, 8, 16, 32, 64, 128]

        modules = []
        stride = 2 if down_factor == 2 else 4
        modules.append(
            ConvBlock(in_channels * 2, num_filters, kernel_size=7, padding=3, stride=stride, activation=activation)
        )
        modules.append(ResBlock(num_filters))
        stride = 2
        if down_factor > 4:
            modules.append(
                ConvBlock(num_filters, num_filters, kernel_size=3, padding=1, stride=stride, activation=activation)
            )
        modules.append(ResBlock(num_filters))
        if down_factor > 8:
            modules.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, stride=stride, activation=activation)
            )
        modules.append(ResBlock(num_filters))
        if down_factor > 16:
            modules.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, stride=stride, activation=activation)
            )
            modules.append(ResBlock(num_filters))
        self.sequential = nn.SequentialCell(*modules)

    def construct(self, im1, im2):
        out = mindspore.ops.Concat(axis=1)([im1, im2])
        out = self.sequential(out)
        return out


class IterativeFlowNet(nn.Cell):
    def __init__(self, params):
        super(IterativeFlowNet, self).__init__()
        self.in_channels: int = params["in_channels"]
        self.num_filters: int = params["IFN_num_filters"]
        self.down_factor: int = params["IFN_down_factor"]
        self.pool_size: int = params["IFN_pool_size"]
        self.interpolation_mode: str = params["IFN_interpolation_mode"]
        self.num_iterations: int = params["IFN_num_iterations"]
        self.warm_start: bool = params["IFN_warm_start"]

        self.FeatureEncoder = FeatureEncoder(self.in_channels, self.num_filters, self.down_factor)
        self.UpdateBlock = UpdateBlock(self.num_filters)

        self.pooling = None
        if self.pool_size is not None:
            if self.pool_size == -1:
                self.pooling = nn.AdaptiveAvgPool2D(1)
            elif self.pool_size > 1:
                self.pooling = nn.AvgPool2d(self.pool_size, stride=self.pool_size)
            else:
                raise ValueError()

    def construct(self, im1, im2, f):

        features = self.FeatureEncoder(im1, im2)
        b, c, h, w = features.shape

        u = mindspore.ops.Zeros()((b, self.num_filters, h, w), mindspore.float32)
        if not self.warm_start:
            f = mindspore.ops.Zeros()((b, 2, h, w), mindspore.float32)

        for i in range(self.num_iterations):
            u, delta_f = self.UpdateBlock(features, u, f)
            f += delta_f

        if self.pooling:
            size = f.shape[2:4]
            f = self.pooling(f)
            f = mindspore.ops.interpolate(x=f, sizes=size, mode=self.interpolation_mode)
        return f


class UpdateBlock(nn.Cell):
    def __init__(self, num_filters, activation="relu"):
        super(UpdateBlock, self).__init__()

        self.conv1 = ConvBlock(num_filters * 2 + 2, num_filters * 2, kernel_size=3, padding=1, activation=activation)
        self.conv2 = ConvBlock(num_filters * 2, num_filters, kernel_size=3, padding=1, activation=None)
        self.conv3 = ConvBlock(num_filters * 2, 2, kernel_size=3, padding=1, activation=None)
        self.conv3.conv.weight = mindspore.common.initializer.initializer(
            self.conv3.conv.weight, shape=[2, 64, 3, 3], dtype=mindspore.float32
        )

    def construct(self, features, u, f):
        out = mindspore.ops.Concat(axis=1)([features, u, f])
        out = self.conv1(out)
        u = self.conv2(out)
        delta_f = self.conv3(out)

        return u, delta_f
