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

import mindspore
import numpy as np
from mindspore import nn, ops

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def make_layers(cfg, batch_norm: bool = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode="pad", padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(*layers)


class VGG(nn.Cell):
    def __init__(self, features: nn.Cell, num_classes: int = 1000, init_weights: bool = True) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = ops.AdaptiveAvgPool2D((7, 7))
        self.classifier = nn.SequentialCell(
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = ops.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Dense):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class L2pooling(nn.Cell):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.channels = channels
        self.stride = stride
        a = np.hanning(filter_size)[1:-1]
        g = mindspore.Tensor(a[:, None] * a[None, :], mindspore.float32)
        g = g / mindspore.ops.ReduceSum()(g)
        self.filter = mindspore.numpy.tile(g[None, None, :, :], (self.channels, 1, 1, 1))

    def construct(self, input_value):
        input_value = input_value**2
        out = ops.Conv2D(
            out_channel=self.filter.shape[0],
            kernel_size=self.filter.shape[-1],
            stride=self.stride,
            pad_mode="pad",
            pad=self.padding,
            group=input_value.shape[1],
        )(input_value, self.filter)
        return mindspore.ops.Sqrt()((out + 1e-12))


class DISTSLoss(nn.Cell):
    def __init__(self, load_weights_path=None, pretrain_weight_path=None):
        super(DISTSLoss, self).__init__()
        vgg16model = VGG(make_layers(cfgs["D"], batch_norm=False))
        if pretrain_weight_path and os.path.exists(pretrain_weight_path):
            parameter_dict = mindspore.load_checkpoint(pretrain_weight_path)
            mindspore.load_param_into_net(net=vgg16model, parameter_dict=parameter_dict, strict_load=True)
        vgg_pretrained_features = vgg16model.features

        self.stage1 = mindspore.nn.SequentialCell()
        self.stage2 = mindspore.nn.SequentialCell()
        self.stage3 = mindspore.nn.SequentialCell()
        self.stage4 = mindspore.nn.SequentialCell()
        self.stage5 = mindspore.nn.SequentialCell()
        for x in range(0, 4):
            self.stage1.append(vgg_pretrained_features[x])

        self.stage2.append(L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.append(vgg_pretrained_features[x])

        self.stage3.append(L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.append(vgg_pretrained_features[x])

        self.stage4.append(L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.append(vgg_pretrained_features[x])

        self.stage5.append(L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.append(vgg_pretrained_features[x])

        for param in self.get_parameters():
            param.requires_grad = True

        self.mean = mindspore.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = mindspore.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        self.chns = [3, 64, 128, 256, 512, 512]
        self.alpha = mindspore.Parameter(mindspore.ops.StandardNormal(seed=2)((1, sum(self.chns), 1, 1)))
        self.beta = mindspore.Parameter(mindspore.ops.StandardNormal(seed=2)((1, sum(self.chns), 1, 1)))

        if load_weights_path and os.path.exists(load_weights_path):
            weights = mindspore.load_checkpoint(load_weights_path)
            self.alpha = weights["alpha"]
            self.beta = weights["beta"]

    def forward_once(self, x):
        y = (x - self.mean) / self.std
        y = self.stage1(y)
        h_relu1_2 = y
        y = self.stage2(y)
        h_relu2_2 = y
        y = self.stage3(y)
        h_relu3_3 = y
        y = self.stage4(y)
        h_relu4_3 = y
        y = self.stage5(y)
        h_relu5_3 = y
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def construct(self, x, y, require_grad=True, batch_average=True):
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        c1 = 1e-6
        c2 = 1e-6
        dist1 = 0
        dist2 = 0
        w_sum = self.alpha.sum() + self.beta.sum()

        alpha = []
        beta = []
        start_ = 0
        for i in self.chns:
            end = start_ + i
            alpha.append((self.beta / w_sum)[:, start_:end, :, :])
            beta.append((self.beta / w_sum)[:, start_:end, :, :])
            start_ = end

        start = 1 if require_grad else 0
        for k in range(start, len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keep_dims=True)
            y_mean = feats1[k].mean([2, 3], keep_dims=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdims=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keep_dims=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keep_dims=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keep_dims=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdims=True)

        score_value = 1 - (dist1 + dist2).squeeze()
        if batch_average:
            return score_value.mean()
        else:
            return score_value
