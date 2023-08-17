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

from mindediting.models.common.tunable_conv import TunableConv2d
from mindediting.utils.init_weights import default_init_weights


class TunableResBlock(nn.Cell):
    def __init__(self, num_channels, kernel_size=3, has_bias=True, res_scale=0.1, num_params=1, mode="mlp"):
        super().__init__()

        self.res_scale = res_scale
        if res_scale == 1.0:
            self.init_weights()

        self.conv1 = TunableConv2d(
            num_channels, num_channels, kernel_size=kernel_size, has_bias=has_bias, num_params=num_params, mode=mode
        )
        self.act = nn.ReLU()
        self.conv2 = TunableConv2d(
            num_channels, num_channels, kernel_size=kernel_size, has_bias=has_bias, num_params=num_params, mode=mode
        )

    def construct(self, x, px):
        """
            Construct function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            px (Tensor): tunable parameter
        Returns:
            Tensor: Results.
        """
        y = self.conv1(x, px)
        y = self.act(y)
        y = self.conv2(y, px)
        y = x + y * self.res_scale
        return y

    def init_weights(self):
        """Initialize weights for ResidualBlock.
        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)
