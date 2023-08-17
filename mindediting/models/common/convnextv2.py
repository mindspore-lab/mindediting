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
from mindspore.common.initializer import TruncatedNormal, Zero, initializer
from mindspore.ops import operations as P

from mindediting.models.common.layer_norm import build_layer_norm
from mindediting.utils.checkpoint import load_param_into_net


def is_static_pad(kernel_size, stride=1, dilation=1):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


class GlobalResponseNorm(nn.Cell):
    """Global Response Normalization layer"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps

        self.weight = ms.Parameter(initializer("zeros", (dim,)), name="weight", requires_grad=True)
        self.bias = ms.Parameter(initializer("zeros", (dim,)), name="bias", requires_grad=True)

        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.cast = P.Cast()

    def construct(self, x):
        x_g = self.sqrt(self.reduce_sum(self.square(x), (2, 3)))
        x_n = x_g / (self.reduce_mean(x_g, 1) + self.eps)
        out = x + self.weight.reshape(1, -1, 1, 1) * (x * x_n) + self.bias.reshape(1, -1, 1, 1)

        return out


class GlobalResponseNormMlp(nn.Cell):
    """MLP w/ Global Response Norm , 1x1 Conv2d"""

    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(
            in_features, hidden_features, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=bias
        )
        self.act = nn.GELU(approximate=False)
        self.grn = GlobalResponseNorm(hidden_features)
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=bias
        )

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.fc2(x)

        return x


class ConvNeXtBlock(nn.Cell):
    """ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    Args:
        in_chs (int): Number of input channels.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_chs, out_chs=None, kernel_size=7, stride=1, dilation=1, mlp_ratio=4, conv_bias=True):
        super().__init__()

        out_chs = out_chs or in_chs
        padding_value = ((stride - 1) + dilation * (kernel_size - 1)) // 2

        self.conv_dw = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding_value,
            pad_mode="pad",
            group=in_chs,
            has_bias=conv_bias,
        )
        self.norm = build_layer_norm(out_chs)
        self.mlp = GlobalResponseNormMlp(out_chs, int(mlp_ratio * out_chs))

    def construct(self, x):
        shortcut = x

        x = self.conv_dw(x)
        x = self.norm(x)
        x = self.mlp(x)

        x = shortcut + x

        return x


class ConvNeXtStage(nn.Cell):
    def __init__(self, in_chs, out_chs, kernel_size=7, stride=2, depth=2, dilation=(1, 1), conv_bias=True):
        super().__init__()

        self.downsample = None
        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1

            padding_value = 0
            if dilation[1] > 1 and is_static_pad(ds_ks, stride, dilation[1]):
                padding_value = ((stride - 1) + dilation[1] * (ds_ks - 1)) // 2

            self.downsample = nn.SequentialCell(
                build_layer_norm(in_chs),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    padding=padding_value,
                    pad_mode="pad",
                    dilation=dilation[0],
                    has_bias=conv_bias,
                ),
            )

            in_chs = out_chs

        stage_blocks = []
        for _ in range(depth):
            stage_blocks.append(
                ConvNeXtBlock(
                    in_chs=in_chs, out_chs=out_chs, kernel_size=kernel_size, dilation=dilation[1], conv_bias=conv_bias
                )
            )
            in_chs = out_chs
        self.blocks = nn.SequentialCell(*stage_blocks)

    def construct(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.blocks(x)

        return x


class ConvNeXt(nn.Cell):
    r"""ConvNeXt
    A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
        self,
        in_chans=3,
        output_stride=32,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        kernel_sizes=7,
        stem_kernel=4,
        stem_stride=4,
        stem_padding=0,
        conv_bias=True,
        num_stages=4,
        pretrained=None,
    ):
        """
        Args:
            in_chans (int): Number of input image channels (default: 3)
            output_stride (int): Output stride of network, one of (8, 16, 32) (default: 32)
            depths (tuple(int)): Number of blocks at each stage. (default: [3, 3, 9, 3])
            dims (tuple(int)): Feature dimension at each stage. (default: [96, 192, 384, 768])
            kernel_sizes (Union[int, List[int]]: Depthwise convolution kernel-sizes for each stage (default: 7)
            patch_size (int): Stem patch size for patch stem (default: 4)
            conv_bias (bool): Use bias layers w/ all convolutions (default: True)
        """
        super().__init__()

        assert num_stages > 0
        assert output_stride in (8, 16, 32)
        if not isinstance(kernel_sizes, (tuple, list)):
            kernel_sizes = [kernel_sizes] * num_stages
        assert len(kernel_sizes) == num_stages

        self.stem = nn.SequentialCell(
            nn.Conv2d(
                in_chans,
                dims[0],
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=stem_padding,
                pad_mode="pad",
                has_bias=conv_bias,
            ),
            build_layer_norm(dims[0]),
        )

        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        for i in range(num_stages):
            stride = 2 if i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    dilation=(first_dilation, dilation),
                    depth=depths[i],
                    conv_bias=conv_bias,
                )
            )
            prev_chs = out_chs
        self.stages = nn.SequentialCell(*stages)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            ignore_list = ["head.norm.weight", "head.norm.bias"]
            load_param_into_net(self, pretrained, strict_load=True, ignore_list=ignore_list, verbose=True)
        elif pretrained is None:
            for _, cell in self.cells_and_names():
                if isinstance(cell, nn.Conv2d):
                    cell.weight = initializer(TruncatedNormal(0.02), cell.weight.shape, cell.weight.dtype)
                    if hasattr(cell, "bias") and cell.bias is not None:
                        cell.bias = initializer(Zero(), cell.bias.shape, cell.bias.dtype)
        else:
            raise TypeError("pretrained must be a str or None")

    def construct(self, x):
        y = self.stem(x)
        y = self.stages(y)

        return y
