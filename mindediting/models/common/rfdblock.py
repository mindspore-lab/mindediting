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

from collections import OrderedDict

import mindspore
from mindspore import nn


def activationLayer(activation="leakyrelu", num_parameters=1):
    activation_layers = {
        "relu": nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(alpha=0.1),
        "prelu": nn.PReLU(channel=num_parameters),
    }
    if activation in activation_layers:
        return activation_layers[activation]
    print("unknown activation")
    return


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        pad_mode="pad",
        padding=padding,
        has_bias=True,
        dilation=dilation,
        group=groups,
    )


def mean_channels(T):
    assert T.dim() == 4
    spatial_sum = T.sum(3, keepdims=True).sum(2, keepdims=True)
    return spatial_sum / (T.size(2) * T.size(3))


def stdv_channels(T):
    assert T.dim() == 4
    T_mean = mean_channels(T)
    T_variance = (T - T_mean).pow(2).sum(3, keepdims=True).sum(2, keepdims=True) / (T.size(2) * T.size(3))
    return T_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    layers = []
    for module in args:
        if isinstance(module, nn.SequentialCell):
            [layers.append(submodule) for submodule in module.children()]
        elif isinstance(module, nn.Cell):
            layers.append(module)
    return nn.SequentialCell(*layers)


class NoneLayer(nn.Cell):
    def __init__(self):
        super(NoneLayer, self).__init__()

    def construct(self, x):
        return x


class SRB_layer(nn.Cell):
    def __init__(self, in_channels, kernel_size, activation="activation"):
        super(SRB_layer, self).__init__()
        self.conv = conv_layer(in_channels, in_channels, kernel_size)
        activation_layers = {
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(alpha=0.1),
            "prelu": nn.PReLU(channel=in_channels),
        }
        if activation in activation_layers:
            self.activation_layer = activation_layers[activation]
        else:
            self.activation_layer = NoneLayer()

    def construct(self, _input):
        out = self.conv(_input)
        _input = self.activation_layer(out + _input)
        return _input


class CCALayer(nn.Cell):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        channel_div_reduction = channel // reduction
        self.conv_du = nn.SequentialCell(
            [
                nn.Conv2d(channel, channel_div_reduction, 1, padding=0, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(channel_div_reduction, channel, 1, padding=0, has_bias=True),
                nn.Sigmoid(),
            ]
        )

    def construct(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Enhanced Spatial Attention (SA) Layer
class ESALayer(nn.Cell):
    def __init__(self, in_channels, reduce_ratio=0.3):
        super(ESALayer, self).__init__()
        self.channels = int(in_channels * reduce_ratio)
        self.conv1x1_1 = conv_layer(in_channels, self.channels, 1)
        self.strideConv = conv_layer(self.channels, self.channels, kernel_size=3, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.upscale = 4
        self.maxpool = nn.MaxPool2d(2)
        self.convgroup1 = conv_layer(self.channels, self.channels, 1)
        self.convgroup2 = conv_layer(self.channels, self.channels, 1)
        self.leaky_relu = nn.LeakyReLU(alpha=0.05)
        self.conv1x1_2 = conv_layer(self.channels, in_channels, 1)

    def construct(self, x):
        x0 = x

        x1 = self.leaky_relu(self.conv1x1_1(x))
        x = self.strideConv(x1)
        x = self.maxpool(x)
        x = self.leaky_relu(self.convgroup1(x))
        x = self.convgroup2(x)
        x = mindspore.ops.interpolate(
            x=x, scales=self.upscale, mode="bilinear", coordinate_transformation_mode="align_corners"
        )
        x = x + x1
        x = self.leaky_relu(self.conv1x1_2(x))
        x = self.sigmoid(x)
        x = x * x0
        return x


# Enhanced Channel Spatial Attention (SA) Layer
class ECSALayer(nn.Cell):
    def __init__(self, in_channels, reduction=16, reduce_ratio=0.3):
        super(ECSALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        in_channels_div_reduction = in_channels // reduction
        self.conv_du = nn.SequentialCell(
            [
                nn.Conv2d(in_channels, in_channels_div_reduction, 1, padding=0, has_bias=True),
                nn.ReLU(),
                nn.Conv2d(in_channels_div_reduction, in_channels, 1, padding=0, has_bias=True),
                nn.Sigmoid(),
            ]
        )

        self.channels = int(in_channels * reduce_ratio)
        self.conv1x1_1 = conv_layer(in_channels, self.channels, 1)
        self.strideConv = conv_layer(self.channels, self.channels, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(2)
        self.convgroup1 = conv_layer(self.channels, self.channels, 1)
        self.convgroup2 = conv_layer(self.channels, self.channels, 1)
        self.leaky_relu = nn.LeakyReLU(alpha=0.05)
        self.conv1x1_2 = conv_layer(self.channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.upscale = 4

    def construct(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        x = x * y

        x0 = x
        x1 = self.leaky_relu(self.conv1x1_1(x))
        x = self.strideConv(x1)
        x = self.maxpool(x)
        x = self.leaky_relu(self.convgroup1(x))
        x = self.convgroup2(x)
        x = mindspore.ops.interpolate(
            x=x, scales=self.upscale, mode="bilinear", coordinate_transformation_mode="align_corners"
        )
        x = x + x1
        x = self.leaky_relu(self.conv1x1_2(x))
        x = self.sigmoid(x)
        x = x * x0
        return x


class RFDB(nn.Cell):
    def __init__(self, in_channels=64, activation="leakyrelu", attentionType="NONE"):
        super(RFDB, self).__init__()
        self.conv1 = conv_layer(in_channels, in_channels, 1)
        self.SRB1 = SRB_layer(in_channels, 3, activation=activation)
        self.conv2 = conv_layer(in_channels, in_channels, 1)
        self.SRB2 = SRB_layer(in_channels, 3, activation=activation)
        self.conv3 = conv_layer(in_channels, in_channels, 1)
        self.SRB3 = SRB_layer(in_channels, 3, activation=activation)
        self.conv4 = conv_layer(in_channels, in_channels, 3)
        self.conv5 = conv_layer(in_channels * 4, in_channels, 1)

        if attentionType == "CCA":
            self.attention = CCALayer(in_channels)
        elif attentionType == "ESA":
            self.attention = ESALayer(in_channels)
        elif attentionType == "ECSA":
            self.attention = ECSALayer(in_channels)
        else:
            self.attention = NoneLayer()

    def construct(self, _input):
        out_c1 = self.conv1(_input)
        out_SRB1 = self.SRB1(_input)

        out_c2 = self.conv2(out_SRB1)
        out_SRB2 = self.SRB2(out_SRB1)

        out_c3 = self.conv3(out_SRB2)
        out_SRB3 = self.SRB3(out_SRB2)

        out_c4 = self.conv4(out_SRB3)

        out_cat = mindspore.ops.Concat(axis=1)([out_c1, out_c2, out_c3, out_c4])

        out_c5 = self.conv5(out_cat)

        out_attention = self.attention(out_c5)

        out_fused = out_attention + _input
        return out_fused


class RRFDB(nn.Cell):
    def __init__(self, in_channels=64, activation="leakyrelu", attentionType="NONE"):
        super(RRFDB, self).__init__()
        self.RFDB1 = RFDB(in_channels, activation, attentionType)
        self.RFDB2 = RFDB(in_channels, activation, attentionType)
        self.RFDB3 = RFDB(in_channels, activation, attentionType)

    def construct(self, x):
        out = self.RFDB1(x)
        out = self.RFDB2(out)
        out = self.RFDB3(out)
        return out * 0.2 + x
