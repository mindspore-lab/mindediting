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
from mindspore import nn, ops

from mindediting.metrics.base_metrics import BaseMetric
from mindediting.models.common.alexnet import alexnet as models_alexnet
from mindediting.models.common.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from mindediting.models.common.squeezenet import SqueezeNet
from mindediting.models.common.vgg import vgg16 as models_vgg16


class squeezenet(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=None):
        super(squeezenet, self).__init__()
        pretrained_features = SqueezeNet(num_classes=1000, pretrained=pretrained).features
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        self.slice6 = nn.SequentialCell()
        self.slice7 = nn.SequentialCell()
        self.N_slices = 7
        for x in range(2):
            self.slice1.append(pretrained_features[x])
        for x in range(2, 5):
            self.slice2.append(pretrained_features[x])
        for x in range(5, 8):
            self.slice3.append(pretrained_features[x])
        for x in range(8, 10):
            self.slice4.append(pretrained_features[x])
        for x in range(10, 11):
            self.slice5.append(pretrained_features[x])
        for x in range(11, 12):
            self.slice6.append(pretrained_features[x])
        for x in range(12, 13):
            self.slice7.append(pretrained_features[x])
        if not requires_grad:
            for param in self.get_parameters():
                param.requires_grad = False

    def construct(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        return h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7


class alexnet(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=None):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = models_alexnet(pretrained).features
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        self.N_slices = 5
        for x in range(2):
            self.slice1.append(alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.append(alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.append(alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.append(alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.append(alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.get_parameters():
                param.requires_grad = False

    def construct(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        return h_relu1, h_relu2, h_relu3, h_relu4, h_relu5


class vgg16(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=None):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models_vgg16(pretrained).features
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        self.N_slices = 5
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
            for param in self.get_parameters():
                param.requires_grad = False

    def construct(self, X):
        h1 = self.slice1(X)
        h_relu1_2 = h1
        h2 = self.slice2(h1)
        h_relu2_2 = h2
        h3 = self.slice3(h2)
        h_relu3_3 = h3
        h4 = self.slice4(h3)
        h_relu4_3 = h4
        h5 = self.slice5(h4)
        h_relu5_3 = h5
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3


class resnet(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def construct(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        return h_relu1, h_conv2, h_conv3, h_conv4, h_conv5


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keep_dims=keepdim)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = ops.Sqrt()((in_feat**2).sum(axis=1, keepdims=True))
    return in_feat / (norm_factor + eps)


class ScalingLayer(nn.Cell):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = ms.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        self.scale = ms.Tensor([0.458, 0.448, 0.450])[None, :, None, None]

    def construct(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Cell):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, has_bias=False),
        ]
        self.model = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.model(x)


class LPIPSModel(nn.Cell):
    """LPIPS model.
    Args:
        lpips (Boolean) : Whether to use linear layers on top of base/trunk network.
        pretrained (Boolean): Whether means linear layers are calibrated with human
            perceptual judgments.
        pnet_rand (Boolean): Whether to randomly initialized trunk.
        net (String): ['alex','vgg','squeeze'] are the base/trunk networks available.
        version (String): choose the version ['v0.1'] is the default and latest;
            ['v0.0'] contained a normalization bug.
        pretrained_model_path (String): Petrained model path.

        The following parameters should only be changed if training the network:

        eval_mode (Boolean): choose the mode; True is for test mode (default).
        pnet_tune (Boolean): Whether to tune the base/trunk network.
        use_dropout (Boolean): Whether to use dropout when training linear layers.

    Reference:
        Zhang, Richard, et al. "The unreasonable effectiveness of deep features as
        a perceptual metric." Proceedings of the IEEE conference on computer vision
        and pattern recognition. 2018.

    """

    def __init__(
        self,
        lpips_model_path=None,
        pnet_model_path=None,
        net="alex",
        version="0.1",
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        eval_mode=True,
        **kwargs
    ):
        super(LPIPSModel, self).__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == "squeeze":  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.CellList(self.lins)

        if lpips_model_path and os.path.exists(lpips_model_path):
            ms.load_checkpoint(lpips_model_path, self)

        self.net = net_type(pretrained=pnet_model_path, requires_grad=self.pnet_tune)

        if eval_mode:
            self.set_train(False)

    def construct(self, in1, in0, retPerLayer=False, normalize=True):
        r"""Computation IQA using LPIPS.
        Args:
            in1: An input tensor. Shape :math:`(N, C, H, W)`.
            in0: A reference tensor. Shape :math:`(N, C, H, W)`.
            retPerLayer (Boolean): return result contains ressult of
                each layer or not. Default: False.
            normalize (Boolean): Whether to normalize image data range
                in [0,1] to [-1,1]. Default: True.

        Returns:
            Quality score.

        """
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == "0.1" else (in0, in1)
        )

        outs0, outs1 = self.net.construct(in0_input), self.net.construct(in1_input)
        diffs = []
        for kk in range(self.L):
            diffs.append((normalize_tensor(outs0[kk]) - normalize_tensor(outs1[kk])) ** 2)

        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for i in range(self.L):
            val += res[i]

        if retPerLayer:
            return (val, res)
        else:
            return val.squeeze()


class LPIPS(BaseMetric):
    def __init__(
        self,
        reduction="avg",
        crop_border=0,
        input_order="HWC",
        convert_to=None,
        process_middle_image=(False, False),
        pnet_model_path=None,
        lpips_model_path=None,
        net="alex",
    ):
        super().__init__(reduction, crop_border, input_order, convert_to, process_middle_image)
        self.pnet_model_path = pnet_model_path
        self.lpips_model_path = lpips_model_path
        self.net_name = net
        self.init()

    def init(self):
        self.model = LPIPSModel(
            lpips_model_path=self.lpips_model_path,
            pnet_model_path=self.pnet_model_path,
            pnet_tune=True,
            net=self.net_name,
        )
        self.model.set_train(False)

    def calculate_metrics(self, pred, gt):
        pred, gt = self.preprocess(pred=pred, gt=gt)
        pred = pred[None, :] / 255.0
        gt = gt[None, :] / 255.0
        if isinstance(self.convert_to, str) and self.convert_to.lower() == "y":
            pred = pred[None, :]
            gt = gt[None, :]
        else:
            pred = pred.transpose(0, 3, 1, 2)
            gt = gt.transpose(0, 3, 1, 2)
        pred = ms.Tensor(pred, ms.float32)
        gt = ms.Tensor(gt, ms.float32)
        output = self.model(pred, gt)
        return output.asnumpy().squeeze()
