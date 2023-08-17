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
from typing import Any

import mindspore as ms
from mindspore import nn, ops

__all__ = ["AlexNet", "alexnet"]


class AlexNet(nn.Cell):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, pad_mode="pad", padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, pad_mode="pad", padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, pad_mode="pad", padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, pad_mode="pad", padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, pad_mode="pad", padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = ops.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, num_classes),
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: str = "/data/LLVT/IQA/alexnet.ckpt", progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained and os.path.exists(pretrained):
        param_dict = ms.load_checkpoint(pretrained)
        ms.load_param_into_net(model, param_dict, strict_load=True)
    return model
