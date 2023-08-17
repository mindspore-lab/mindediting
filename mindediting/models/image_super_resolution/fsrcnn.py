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
import mindspore.ops.operations as P
from mindspore.common.initializer import Normal, Zero, initializer


class FSRCNN(nn.Cell):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.cast = P.Cast()

        self.first_part = nn.SequentialCell(
            [
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=d,
                    kernel_size=5,
                    pad_mode="pad",
                    padding=5 // 2,
                    has_bias=True,
                ),
                nn.PReLU(channel=d),
            ]
        )

        self.mid_part = [
            nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True),
            nn.PReLU(channel=s),
        ]

        for i in range(m):
            self.mid_part.extend(
                [
                    nn.Conv2d(
                        in_channels=s,
                        out_channels=s,
                        kernel_size=3,
                        stride=1,
                        pad_mode="pad",
                        padding=3 // 2,
                        has_bias=True,
                    ),
                    nn.PReLU(channel=s),
                ]
            )

        self.mid_part.extend(
            [
                nn.Conv2d(
                    in_channels=s, out_channels=d, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=True
                ),
                nn.PReLU(channel=d),
            ]
        )

        self.mid_part = nn.SequentialCell([*self.mid_part])
        self.expand_dims = P.ExpandDims()
        self.last_part = nn.Conv3dTranspose(
            d,
            num_channels,
            kernel_size=(1, 9, 9),
            stride=(1, scale_factor, scale_factor),
            pad_mode="pad",
            padding=(0, 0, 9 // 2, 9 // 2, 9 // 2, 9 // 2),
            output_padding=(0, scale_factor - 1, scale_factor - 1),
            has_bias=True,
            weight_init="Normal",
            bias_init="zeros",
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                m.weight.data.set_data(
                    initializer(
                        Normal(2 / (m.out_channels * mindspore.ops.Size()(m.weight.data[0][0]))),
                        mindspore.ops.Shape()(m.weight.data),
                    )
                )
                m.bias.data.set_data(initializer(Zero(), mindspore.ops.Size()(m.bias.data)))

        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                m.weight.data.set_data(
                    initializer(
                        Normal(2 / (m.out_channels * mindspore.ops.Size()(m.weight.data[0][0]))),
                        mindspore.ops.Shape()(m.weight.data),
                    )
                )
                m.bias.data.set_data(initializer(Zero(), mindspore.ops.Size()(m.bias.data)))
        self.last_part.weight.data.set_data(
            initializer(
                Normal(2 / (self.last_part.out_channels * mindspore.ops.Size()(self.last_part.weight.data[0][0]))),
                mindspore.ops.Shape()(self.last_part.weight.data),
            )
        )
        self.last_part.bias.set_data(initializer(Zero(), mindspore.ops.Size()(self.last_part.bias)))

    def construct(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.expand_dims(x, 2)
        x = self.last_part(x)
        x = x.squeeze(2)
        return x
