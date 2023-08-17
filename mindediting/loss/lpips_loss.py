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

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .dists_loss import VGG, L2pooling, cfgs, make_layers


class LPIPS(nn.Cell):
    """
    Refer to https://github.com/richzhang/PerceptualSimilarity
    """

    def __init__(self, load_weights_path=None, vgg_pretrained_path=None):
        super(LPIPS, self).__init__()

        vgg16model = VGG(make_layers(cfgs["D"], batch_norm=False))
        if vgg_pretrained_path is None:
            print("[WARNING] VGG is not initialized to use as LPIPS loss")
        else:
            assert os.path.exists(vgg_pretrained_path)
            parameter_dict = ms.load_checkpoint(vgg_pretrained_path)
            ms.load_param_into_net(net=vgg16model, parameter_dict=parameter_dict, strict_load=True)

        vgg_pretrained_features = vgg16model.features

        self.stage1 = nn.SequentialCell()
        for x in range(0, 4):
            self.stage1.append(vgg_pretrained_features[x])

        self.stage2 = nn.SequentialCell([L2pooling(channels=64)])
        for x in range(5, 9):
            self.stage2.append(vgg_pretrained_features[x])

        self.stage3 = nn.SequentialCell([L2pooling(channels=128)])
        for x in range(10, 16):
            self.stage3.append(vgg_pretrained_features[x])

        self.stage4 = nn.SequentialCell([L2pooling(channels=256)])
        for x in range(17, 23):
            self.stage4.append(vgg_pretrained_features[x])

        self.stage5 = nn.SequentialCell([L2pooling(channels=512)])
        for x in range(24, 30):
            self.stage5.append(vgg_pretrained_features[x])

        for param in self.get_parameters():
            param.requires_grad = True

        self.mean = ms.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = ms.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        self.channels = [64, 128, 256, 512, 512]
        self.weights = ms.Parameter(ms.ops.StandardNormal(seed=2)((1, sum(self.channels), 1, 1)))

        if load_weights_path is None:
            print("[WARNING] LPIPS weights are not loaded")
        else:
            assert os.path.exists(load_weights_path)
            weights = ms.load_checkpoint(load_weights_path)
            self.weights = ops.cat(list(weights.values()), 1)

        self.norm = ops.L2Normalize(axis=1)

    def forward_once(self, x):
        h = (x - self.mean) / self.std

        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h

        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

        for k in range(len(outs)):
            outs[k] = self.norm(outs[k])

        return outs

    def construct(self, x, y):
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)

        weights = ops.split(self.weights, self.channels, 1)

        scores = 0
        for k in range(len(self.channels)):
            scores = scores + (weights[k] * (feats0[k] - feats1[k]) ** 2).mean([2, 3]).sum(1)

        return scores


class LPIPSLoss(nn.Cell):
    def __init__(self, weight=1.0, load_weights_path=None, vgg_pretrained=None):
        super(LPIPSLoss, self).__init__()

        self.weight = weight

        self.model = LPIPS(load_weights_path=load_weights_path, vgg_pretrained_path=vgg_pretrained)

    def construct(self, pred, target):
        losses = self.model(pred, target)
        loss = losses.mean()

        return self.weight * loss
