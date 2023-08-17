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
from collections import namedtuple

import mindspore as ms
import numpy as np
from mindspore import nn, ops
from mindspore.common.initializer import Normal, initializer

from mindediting.metrics.base_metrics import BaseMetric
from mindediting.models.common.vgg import vgg16 as Vgg16


class L2pooling(nn.Cell):
    """from https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/DISTS_pt.py"""

    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = a[:, None] * a[None, :]
        g = g / g.sum()
        g = ms.Tensor(g, ms.float32)
        self.filter = ms.numpy.tile(g[None, None, :, :], (self.channels, 1, 1, 1)).astype("float32")

    def construct(self, x):
        x = x**2
        conv2d = ops.Conv2D(
            out_channel=self.filter.shape[0],
            kernel_size=self.filter.shape[2],
            stride=self.stride,
            pad_mode="pad",
            pad=self.padding,
            group=x.shape[1],
        )
        out = conv2d(x, self.filter)
        return (out + 1e-12).sqrt()


class vgg16(nn.Cell):
    def __init__(self, requires_grad=False, vgg16_model_path=None, l2pooling=False):
        super(vgg16, self).__init__()
        vgg_pretrained_features = Vgg16(vgg16_model_path).features
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        self.N_slices = 5

        if l2pooling:
            for x in range(0, 4):
                self.slice1.append(vgg_pretrained_features[x])
            self.slice2.append(L2pooling(channels=64))
            for x in range(5, 9):
                self.slice2.append(vgg_pretrained_features[x])
            self.slice3.append(L2pooling(channels=128))
            for x in range(10, 16):
                self.slice3.append(vgg_pretrained_features[x])
            self.slice4.append(L2pooling(channels=256))
            for x in range(17, 23):
                self.slice4.append(vgg_pretrained_features[x])
            self.slice5.append(L2pooling(channels=512))
            for x in range(24, 30):
                self.slice5.append(vgg_pretrained_features[x])
        else:
            for x in range(4):
                self.slice1.append(vgg_pretrained_features[x])
            for x in range(4, 9):
                self.slice2.append(vgg_pretrained_features[x])
            for x in range(9, 16):
                self.slice3.append(vgg_pretrained_features[x])
            for x in range(16, 23):
                self.slice4.append(vgg_pretrained_features[x])
            for x in range(23, 30):
                self.slice5.append(vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def construct(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class DISTSModel(nn.Cell):
    def __init__(self, vgg16_model_path=None, alpha_beta_model_path=None, pnet_tune=False, load_path=None):
        super(DISTSModel, self).__init__()
        self.pnet_tune = pnet_tune

        self.mean = ms.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = ms.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        self.chns = [3, 64, 128, 256, 512, 512]

        self.alpha = ms.Parameter(
            initializer(
                Normal(sigma=0.01, mean=0.1),
                shape=(1, sum(self.chns), 1, 1),
                dtype=ms.float32,
            )
        )
        self.beta = ms.Parameter(
            initializer(
                Normal(sigma=0.01, mean=0.1),
                shape=(1, sum(self.chns), 1, 1),
                dtype=ms.float32,
            )
        )

        # find model weights for: backbone + alpha/beta
        if alpha_beta_model_path == None:
            # there is only one weights file => file format is BMVC-DISTS
            self.net = vgg16(vgg16_model_path=None, requires_grad=self.pnet_tune, l2pooling=True)
            if vgg16_model_path and os.path.exists(vgg16_model_path):
                # load VGG parameters
                param_dict = ms.load_checkpoint(vgg16_model_path)
                ms.load_param_into_net(self.net, param_dict, strict_load=True)
                # load ALPHA/BETA
                self.alpha = param_dict["alpha"]
                self.beta = param_dict["beta"]
        else:
            # DISTS
            if vgg16_model_path and os.path.exists(vgg16_model_path):
                # load VGG parameters
                self.net = vgg16(vgg16_model_path=vgg16_model_path, requires_grad=self.pnet_tune, l2pooling=True)
            if alpha_beta_model_path and os.path.exists(alpha_beta_model_path):
                param_dict = ms.load_checkpoint(alpha_beta_model_path)
                self.alpha = param_dict["alpha"]
                self.beta = param_dict["beta"]

    def custom_net_construct(self, x):
        h = (x - self.mean) / self.std
        h = self.net.slice1(h)
        h_relu1_2 = h
        h = self.net.slice2(h)
        h_relu2_2 = h
        h = self.net.slice3(h)
        h_relu3_3 = h
        h = self.net.slice4(h)
        h_relu4_3 = h
        h = self.net.slice5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def construct(self, ref, x, normalize=False):
        feats0 = self.custom_net_construct(ref)
        feats1 = self.custom_net_construct(x)

        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = ms.numpy.split(self.alpha / w_sum, np.cumsum(self.chns)[:-1].tolist(), axis=1)
        beta = ms.numpy.split(self.beta / w_sum, np.cumsum(self.chns)[:-1].tolist(), axis=1)

        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keep_dims=True)
            y_mean = feats1[k].mean([2, 3], keep_dims=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdims=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keep_dims=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keep_dims=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keep_dims=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdims=True)

        score = 1 - (dist1 + dist2)
        return score.squeeze(-1).squeeze(-1)


class DISTS(BaseMetric):
    def __init__(
        self,
        reduction="avg",
        crop_border=0,
        input_order="HWC",
        convert_to=None,
        process_middle_image=(False, False),
        **kwargs
    ):
        super().__init__(reduction, crop_border, input_order, convert_to, process_middle_image)
        self.vgg16_model_path = kwargs.get("vgg16_model_path")
        self.alpha_beta_model_path = kwargs.get("alpha_beta_model_path")
        self.model = DISTSModel(
            pnet_tune=True, vgg16_model_path=self.vgg16_model_path, alpha_beta_model_path=self.alpha_beta_model_path
        )
        self.model.set_train(False)

    def calculate_metrics(self, pred, gt):
        pred, gt = self.preprocess(pred=pred, gt=gt)
        pred = pred[None, :] / 255.0
        gt = gt[None, :] / 255.0
        if isinstance(self.convert_to, str) and self.convert_to.lower() == "y":
            pred = np.expand_dims(pred, axis=0)
            gt = np.expand_dims(gt, axis=0)
        else:
            pred = pred.transpose(0, 3, 1, 2)
            gt = gt.transpose(0, 3, 1, 2)
        pred = ms.Tensor(pred, ms.float32)
        gt = ms.Tensor(gt, ms.float32)
        output = self.model(pred, gt)
        return output.asnumpy().squeeze()
