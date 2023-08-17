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


from mindspore import nn

from mindediting.models.common.prely import PReLU_PT as PReLU
from mindediting.models.common.resize import BilinearResize


def first(input_channels, output_channels):
    return nn.SequentialCell(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
        PReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
    )


def lateral(input_channels, output_channels):
    return nn.SequentialCell(
        PReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
        PReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
    )


def downsampling(input_channels, output_channels):
    return nn.SequentialCell(
        PReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, pad_mode="pad", has_bias=True),
        PReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
    )


def upsampling(input_channels, output_channels):
    return nn.SequentialCell(
        BilinearResize(scale_factor=2, align_corners=True),
        PReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
        PReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
    )


def last(input_channels, output_channels):
    return nn.SequentialCell(
        PReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
        PReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True),
    )


class GridNet(nn.Cell):
    def __init__(self, input_channels, internal_channels, output_channels):
        super().__init__()

        assert len(input_channels) == 3
        assert len(internal_channels) == 3

        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * 3
        assert len(output_channels) == 3

        self.row_1_in = first(input_channels[0], internal_channels[0])
        self.row_1_1 = lateral(internal_channels[0], internal_channels[0])
        self.row_1_2 = lateral(internal_channels[0], internal_channels[0])
        self.row_1_3 = lateral(internal_channels[0], internal_channels[0])
        self.row_1_4 = lateral(internal_channels[0], internal_channels[0])
        self.row_1_5 = lateral(internal_channels[0], internal_channels[0])
        self.row_1_out = last(internal_channels[0], output_channels[0])

        self.row_2_in = first(input_channels[1], internal_channels[1])
        self.row_2_1 = lateral(internal_channels[1], internal_channels[1])
        self.row_2_2 = lateral(internal_channels[1], internal_channels[1])
        self.row_2_3 = lateral(internal_channels[1], internal_channels[1])
        self.row_2_4 = lateral(internal_channels[1], internal_channels[1])
        self.row_2_5 = lateral(internal_channels[1], internal_channels[1])

        self.row_3_in = first(input_channels[2], internal_channels[2])
        self.row_3_1 = lateral(internal_channels[2], internal_channels[2])
        self.row_3_2 = lateral(internal_channels[2], internal_channels[2])
        self.row_3_3 = lateral(internal_channels[2], internal_channels[2])
        self.row_3_4 = lateral(internal_channels[2], internal_channels[2])
        self.row_3_5 = lateral(internal_channels[2], internal_channels[2])

        self.col_1_1 = downsampling(internal_channels[0], internal_channels[1])
        self.col_2_1 = downsampling(internal_channels[1], internal_channels[2])
        self.col_1_2 = downsampling(internal_channels[0], internal_channels[1])
        self.col_2_2 = downsampling(internal_channels[1], internal_channels[2])
        self.col_1_3 = downsampling(internal_channels[0], internal_channels[1])
        self.col_2_3 = downsampling(internal_channels[1], internal_channels[2])

        self.col_1_4 = upsampling(internal_channels[1], internal_channels[0])
        self.col_2_4 = upsampling(internal_channels[2], internal_channels[1])
        self.col_1_5 = upsampling(internal_channels[1], internal_channels[0])
        self.col_2_5 = upsampling(internal_channels[2], internal_channels[1])
        self.col_1_6 = upsampling(internal_channels[1], internal_channels[0])
        self.col_2_6 = upsampling(internal_channels[2], internal_channels[1])

    def construct(self, feature_1, feature_2, feature_3):
        var_1_1 = self.row_1_in(feature_1)
        var_1_2 = self.row_1_1(var_1_1) + var_1_1
        var_1_3 = self.row_1_2(var_1_2) + var_1_2

        var_2_0 = self.row_2_in(feature_2)
        var_2_1 = self.col_1_1(var_1_1) + var_2_0
        var_2_2 = self.col_1_2(var_1_2) + self.row_2_1(var_2_1) + var_2_1
        var_2_3 = self.col_1_3(var_1_3) + self.row_2_2(var_2_2) + var_2_2

        var_3_0 = self.row_3_in(feature_3)
        var_3_1 = self.col_2_1(var_2_1) + var_3_0
        var_3_2 = self.col_2_2(var_2_2) + self.row_3_1(var_3_1) + var_3_1
        var_3_3 = self.col_2_3(var_2_3) + self.row_3_2(var_3_2) + var_3_2

        var_3_4 = self.row_3_3(var_3_3) + var_3_3
        var_3_5 = self.row_3_4(var_3_4) + var_3_4
        var_3_6 = self.row_3_5(var_3_5) + var_3_5

        var_2_4 = self.col_2_4(var_3_4) + self.row_2_3(var_2_3) + var_2_3
        var_2_5 = self.col_2_5(var_3_5) + self.row_2_4(var_2_4) + var_2_4
        var_2_6 = self.col_2_6(var_3_6) + self.row_2_5(var_2_5) + var_2_5

        var_1_4 = self.col_1_4(var_2_4) + self.row_1_3(var_1_3) + var_1_3
        var_1_5 = self.col_1_5(var_2_5) + self.row_1_4(var_1_4) + var_1_4
        var_1_6 = self.col_1_6(var_2_6) + self.row_1_5(var_1_5) + var_1_5

        out = self.row_1_out(var_1_6)

        return out
