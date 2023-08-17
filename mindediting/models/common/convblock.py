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

import warnings

from mindspore import nn


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


class ConvBlock(Block):
    """
    Convolutional block
    ---Conv---

    Convolutional block with activation
    ---Conv-Act---

    Convolutional block with batch norm and activation
    ---Conv-BN-Act---
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        bias=True,
        with_batch_norm=False,
        activation="relu",
        self_shortcut=False,
        is_adder=False,
    ):
        super(ConvBlock, self).__init__()
        if self_shortcut and not is_adder:
            warnings.warn("Self-shortcut is enabled but ConvBlock does not use Adder2d.")
        assert (
            in_channels == out_channels or not self_shortcut
        ), f"Input and output channels must be equal if `self_shortcut` is enabled"
        assert stride == 1 or not self_shortcut, f"Stride should be 1 if `self_shortcut` is enabled"
        self.self_shortcut = self_shortcut
        if is_adder:
            with_batch_norm = True
        if with_batch_norm:
            bias = False
        Conv2d = nn.Conv2d

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            group=groups,
            has_bias=bias,
        )
        self.batch_norm = None
        if with_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = self.activation_fn(activation)

    def construct(self, x):
        out = self.conv(x)
        if self.self_shortcut:
            out += x
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.act is not None:
            out = self.act(out)
        return out
