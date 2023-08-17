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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.nn.layer.activation import get_activation

from mindediting.utils.utils import is_ascend


class TunableSequentialCell(nn.SequentialCell):
    def __init__(self, cell_list):
        super().__init__(cell_list)

    def construct(self, x, px):
        for cell in self.cell_list:
            x = cell(x, px)
        return x


class TunableCell(nn.Cell):
    def __init__(self, num_params=2, expand_params=1, mode="mlp"):
        super(TunableCell, self).__init__()

        self.num_params = num_params
        self.expand_params = expand_params or 1
        self.mode = mode or "mlp"

        self.num_weights = self.num_params * self.expand_params

        assert self.mode in ["linear", "mlp"]

        self.mlp = None
        if self.expand_params > 1 or self.mode == "mlp":
            self.mlp = nn.Dense(
                self.num_params, self.num_weights, has_bias=True, weight_init="normal", bias_init="zeros"
            )

    def extend_repr(self):
        s = "num_params={}, expand_params={}, mode={}".format(self.num_params, self.expand_params, self.mode)
        return s

    def _init_parameter(self, init, shape, name):
        return ms.Parameter(initializer(init, shape), name=name, requires_grad=True)

    def check_input(self, px):
        assert px is not None
        assert px.ndim == 2, px.shape
        assert px.shape[1] == self.num_params, (px.shape, self.num_params)

        return 0


class TunableParameter(TunableCell):
    def __init__(self, default_input, name=None, requires_grad=True, num_params=2, expand_params=1, mode="mlp"):
        super(TunableParameter, self).__init__(num_params=num_params, expand_params=expand_params, mode=mode)

        self.name = name
        self.requires_grad = requires_grad

        self.ndim = default_input.ndim
        default_input = ops.repeat_elements(default_input.view(1, 1, -1), rep=self.num_weights, axis=1).view(
            1, self.num_weights, *default_input.shape
        )
        self.data = ms.Parameter(default_input=default_input, name=name, requires_grad=requires_grad)

    def extend_repr(self):
        s = "name={}, shape={}, dtype={}, requires_grad={}".format(
            self.name, self.data.shape, self.data.dtype, self.requires_grad
        )
        s += ", {}".format(TunableCell.extend_repr(self))
        return s

    def construct(self, px):
        self.check_input(px)

        b = px.shape[0]
        w = self.num_weights

        if self.mlp is not None:
            px = self.mlp(px)

        px = px.view(b, w, *[1 for _ in range(self.ndim)])
        data = (px * self.data).sum(axis=1)

        return data


class TunableDense(TunableCell):
    def __init__(
        self,
        in_channels,
        out_channels,
        weight_init="normal",
        bias_init="zeros",
        has_bias=True,
        activation=None,
        num_params=1,
        expand_params=1,
        mode="mlp",
    ):
        super(TunableDense, self).__init__(num_params=num_params, expand_params=expand_params, mode=mode)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.has_bias = has_bias

        self.bmm = ops.BatchMatMul(transpose_b=True)

        self.weight = self._init_parameter(weight_init, [1, self.num_weights, out_channels, in_channels], "weight")
        self.bias = self._init_parameter(bias_init, [1, self.num_weights, out_channels], "bias") if has_bias else None

        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        assert activation is None or isinstance(self.activation, (nn.Cell, ops.primitive.Primitive))

    def extend_repr(self):
        s = "input_channels={}, output_channels={}".format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ", has_bias={}".format(self.has_bias)
        if self.activation is not None:
            s += ", activation={}".format(self.activation)
        s += ", {}".format(TunableCell.extend_repr(self))
        return s

    def construct(self, x, px):
        self.check_input(px)

        b = x.shape[0]
        w = self.num_weights
        c = self.in_channels
        d = self.out_channels

        if self.mlp is not None:
            px = self.mlp(px)

        weight = (px.view(b, w, 1, 1) * self.weight).sum(axis=1, keepdims=False)
        bias = (px.view(b, w, 1) * self.bias).sum(axis=1, keepdims=True) if self.bias is not None else None

        assert weight.shape == (b, d, c), weight.shape
        assert bias.shape == (b, 1, d), bias.shape

        y = self.bmm(x.view(b, -1, c), weight) + bias

        return y


class TunableConv2d(TunableCell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        pad_mode="same",
        padding=0,
        dilation=1,
        group=1,
        has_bias=False,
        weight_init="normal",
        bias_init="zeros",
        data_format="NCHW",
        num_params=1,
        expand_params=1,
        mode="mlp",
    ):
        super(TunableConv2d, self).__init__(num_params=num_params, expand_params=expand_params, mode=mode)

        # mindspore.ops.conv2d only supports 'NCHW' data_format
        assert data_format == "NCHW"
        self.is_ascend = is_ascend()
        self.cast = ms.ops.Cast()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.has_bias = has_bias
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.data_format = data_format

        assert in_channels % self.group == 0

        self.weight = self._init_parameter(
            weight_init,
            [1, self.num_weights, out_channels, in_channels // self.group, kernel_size, kernel_size],
            "weight",
        )
        self.bias = self._init_parameter(bias_init, [1, self.num_weights, out_channels], "bias") if has_bias else None

    def extend_repr(self):
        s = (
            "input_channels={}, output_channels={}, kernel_size={},"
            "stride={}, pad_mode={}, padding={}, dilation={}, "
            "group={}, has_bias={}, "
            "weight_init={}, bias_init={}, data_format={}, {}".format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.pad_mode,
                self.padding,
                self.dilation,
                self.group,
                self.has_bias,
                self.weight_init,
                self.bias_init,
                self.data_format,
                TunableCell.extend_repr(self),
            )
        )
        return s

    def _batch_conv2d(self, x, weight, bias=None):
        assert weight.shape[1] % self.group == 0, weight.shape
        assert x.ndim == 4 and weight.ndim == 5
        assert x.shape[0] == weight.shape[0]

        b, in_ch, h, w = x.shape
        _, out_ch, _, kh, kw = weight.shape
        y = ops.conv2d(
            x.view(1, b * in_ch, h, w),
            weight.view(b * out_ch, in_ch // self.group, kh, kw),
            pad_mode=self.pad_mode,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            group=b * self.group,
        )
        y = y.view(b, out_ch, y.shape[2], y.shape[3])

        if bias is not None:
            assert bias.ndim == 2
            assert b == bias.shape[0] and out_ch == bias.shape[1]
            y = y + bias.view(b, out_ch, 1, 1)

        return y

    def construct(self, x, px):
        self.check_input(px)

        b = x.shape[0]
        w = self.num_weights

        if self.mlp is not None:
            px = self.mlp(px)

        weight = (px.view(b, w, 1, 1, 1, 1) * self.weight).sum(axis=1)
        bias = (px.view(b, w, 1) * self.bias).sum(axis=1) if self.bias is not None else None

        if self.is_ascend:
            x = self.cast(x, ms.float16)
            weight = self.cast(weight, ms.float16)
            bias = self.cast(bias, ms.float16) if self.bias is not None else None
        y = self._batch_conv2d(x, weight, bias=bias)
        if self.is_ascend:
            y = self.cast(y, ms.float32)
        return y
