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

import math

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import numpy as np
from mindspore import Parameter, Tensor


class ColorTransfer(nn.Cell):
    def __init__(self):
        super(ColorTransfer, self).__init__()

        cfa = np.array(
            [
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5, -0.5],
                [0.65, 0.2784, -0.2784, -0.65],
                [-0.2784, 0.65, -0.65, 0.2764],
            ]
        )
        cfa = np.expand_dims(cfa, axis=2)
        cfa = np.expand_dims(cfa, axis=3)
        cfa = Tensor(cfa, mindspore.float32)

        self.net1 = nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=False
        )
        self.net1.weight = Parameter(cfa)

    def construct(self, x):
        out = self.net1(x)
        return out


class ColorTransferInv(nn.Cell):
    def __init__(self):
        super(ColorTransferInv, self).__init__()

        cfa = np.array(
            [
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5, -0.5],
                [0.65, 0.2784, -0.2784, -0.65],
                [-0.2784, 0.65, -0.65, 0.2764],
            ]
        )

        cfa = np.expand_dims(cfa, axis=2)
        cfa = np.expand_dims(cfa, axis=3)
        cfa = Tensor(cfa, mindspore.float32)
        cfa_inv = mindspore.ops.Transpose()(cfa, (1, 0, 2, 3))

        self.net1 = nn.Conv2d(
            in_channels=4, out_channels=4, kernel_size=1, stride=1, pad_mode="pad", padding=0, has_bias=False
        )
        self.net1.weight = Parameter(cfa_inv)

    def construct(self, x):
        out = self.net1(x)
        return out


class HandMadeConv2dTransposeWidth(nn.Cell):
    def __init__(self):
        super(HandMadeConv2dTransposeWidth, self).__init__()
        self.kernel_size = (1, 2)
        self.stride_size = (1, 2)

    def construct(self, x, weights):
        N, C, H, W = x.shape
        k_h, k_w = self.kernel_size
        H_out = (H - 1) * self.stride_size[0] + (k_h - 1) + 1
        W_out = (W - 1) * self.stride_size[1] + (k_w - 1) + 1

        x_ravel = x.ravel()
        x1 = mindspore.ops.mul(x_ravel, weights.view(-1)[0])
        x2 = mindspore.ops.mul(x_ravel, weights.view(-1)[1])
        res = mindspore.ops.stack((x1, x2), 1)
        res = res.view(N, -1, H_out, W_out)

        return res


class HandMadeConv2dTransposeHeight(nn.Cell):
    def __init__(self):
        super(HandMadeConv2dTransposeHeight, self).__init__()
        self.kernel_size = (2, 1)
        self.stride_size = (2, 1)

    def construct(self, x, weights):
        N, C, H, W = x.shape
        k_h, k_w = self.kernel_size
        H_out = (H - 1) * self.stride_size[0] + (k_h - 1) + 1
        W_out = (W - 1) * self.stride_size[1] + (k_w - 1) + 1

        x_ravel = x.ravel()
        x1 = mindspore.ops.mul(x_ravel, weights.view(-1)[0])
        x2 = mindspore.ops.mul(x_ravel, weights.view(-1)[1])
        res = mindspore.ops.stack((x1, x2), 1)
        res = res.view(N, -1, W, 2)
        res = res.transpose((0, 1, 3, 2)).view((N, C, H_out, W_out))

        return res


class FreTransfer(nn.Cell):
    def __init__(self):
        super(FreTransfer, self).__init__()

        h0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
        h1 = np.array([-1 / math.sqrt(2), 1 / math.sqrt(2)])
        h0 = np.array(h0[::-1]).ravel()
        h1 = np.array(h1[::-1]).ravel()
        h0 = Tensor(h0, mindspore.float32).reshape((1, 1, 1, -1))  # row lowpass
        h1 = Tensor(h1, mindspore.float32).reshape((1, 1, 1, -1))  # row highpass
        ll_filt = mindspore.ops.Concat()((h0, h1))

        self.net1 = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=(1, 2), stride=(1, 2), pad_mode="pad", padding=0, has_bias=False
        )  # Cin = 1, Cout = 4, kernel_size = (1,2)
        self.net1.weight = Parameter(ll_filt)  # torch.Size([2, 1, 1, 2])
        self.conv2d = mindspore.ops.Conv2D(out_channel=2, kernel_size=(2, 1), stride=(2, 1), pad_mode="pad", pad=0)
        self.net2 = nn.Conv2d(
            in_channels=1, out_channels=2, kernel_size=(2, 1), stride=(2, 1), pad_mode="pad", padding=0
        )

    def construct(self, x):
        B, C, H, W = x.shape

        curr_list = []
        ll_list, hl_list, lh_list, hh_list = [], [], [], []
        for i in range(C):
            ll_ = self.net1(x[:, i : (i + 1) * 1, :, :])  # 1 * 2 * 128 * 64
            y = []
            for j in range(2):
                weight = self.net1.weight.transpose((0, 1, 3, 2))
                y_out = self.conv2d(ll_[:, j : (j + 1) * 1, :, :], weight)
                y.append(y_out)  #
            y_ = P.Concat(1)([y[0], y[1]])
            ll_list.append(y_[:, 0:1, :, :])
            hl_list.append(y_[:, 1:2, :, :])
            lh_list.append(y_[:, 2:3, :, :])
            hh_list.append(y_[:, 3:4, :, :])
        ll = P.Concat(1)(ll_list)
        hl = P.Concat(1)(hl_list)
        lh = P.Concat(1)(lh_list)
        hh = P.Concat(1)(hh_list)
        out = P.Concat(1)([ll, hl, lh, hh])
        return out


class FreTransferInv(nn.Cell):
    def __init__(self):
        super(FreTransferInv, self).__init__()

        g0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
        g1 = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
        g0 = Tensor(g0, mindspore.float32).reshape((1, 1, -1, 1))
        g1 = Tensor(g1, mindspore.float32).reshape((1, 1, -1, 1))

        self.deconv1 = HandMadeConv2dTransposeHeight()
        self.net1_weight = Parameter(g0)  # torch.Size([1,1,2,1])
        self.deconv2 = HandMadeConv2dTransposeHeight()
        self.net2_weight = Parameter(g1)  # torch.Size([1,1,2,1])
        self.deconv3 = HandMadeConv2dTransposeWidth()

    def construct(self, x):
        lls = x[:, 0:4, :, :]
        hls = x[:, 4:8, :, :]
        lhs = x[:, 8:12, :, :]
        hhs = x[:, 12:16, :, :]
        B, C, H, W = lls.shape

        out_list = []

        tt, weight = [], []
        for i in range(C):
            ll = lls[:, i : i + 1, :, :]
            hl = hls[:, i : i + 1, :, :]
            lh = lhs[:, i : i + 1, :, :]
            hh = hhs[:, i : i + 1, :, :]

            lo1 = self.deconv1(ll, self.net1_weight)
            lo2 = self.deconv2(hl, self.net2_weight)
            lo = lo1 + lo2

            hi = self.deconv1(lh, self.net1_weight) + self.deconv2(hh, self.net2_weight)

            weight_l = self.net1_weight.transpose((0, 1, 3, 2))
            weight_h = self.net2_weight.transpose((0, 1, 3, 2))
            l = self.deconv3(lo, weight_l)
            h = self.deconv3(hi, weight_h)

            out_list.append(l + h)
        out = P.Concat(1)(out_list)
        return out


class FreTransferInv_weight(nn.Cell):
    def __init__(self):
        super(FreTransferInv_weight, self).__init__()
        self.deconv1 = HandMadeConv2dTransposeHeight()
        self.deconv2 = HandMadeConv2dTransposeHeight()
        self.deconv3 = HandMadeConv2dTransposeWidth()

    def construct(self, x, net1_weight, net2_weight):
        lls = x[:, 0:4, :, :]
        hls = x[:, 4:8, :, :]
        lhs = x[:, 8:12, :, :]
        hhs = x[:, 12:16, :, :]
        B, C, H, W = lls.shape

        out_list = []

        tt, weight = [], []
        for i in range(C):
            ll = lls[:, i : i + 1, :, :]
            hl = hls[:, i : i + 1, :, :]
            lh = lhs[:, i : i + 1, :, :]
            hh = hhs[:, i : i + 1, :, :]

            lo1 = self.deconv1(ll, net1_weight)
            lo2 = self.deconv2(hl, net2_weight)
            lo = lo1 + lo2

            hi = self.deconv1(lh, net1_weight) + self.deconv2(hh, net2_weight)

            weight_l = net1_weight.transpose((0, 1, 3, 2))
            weight_h = net2_weight.transpose((0, 1, 3, 2))
            l = self.deconv3(lo, weight_l)
            h = self.deconv3(hi, weight_h)

            out_list.append(l + h)
        out = P.Concat(1)(out_list)
        return out


class Fusion_down(nn.Cell):
    def __init__(self):
        super(Fusion_down, self).__init__()
        self.net1 = nn.Conv2d(
            in_channels=5, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )

    def construct(self, x):
        net1 = P.ReLU()(self.net1(x))
        net2 = P.ReLU()(self.net2(net1))
        out = P.Sigmoid()(self.net3(net2))
        return out


class Fusion_up(nn.Cell):
    def __init__(self):
        super(Fusion_up, self).__init__()
        self.net1 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )

    def construct(self, x):
        net1 = P.ReLU()(self.net1(x))
        net2 = P.ReLU()(self.net2(net1))
        out = P.Sigmoid()(self.net3(net2))
        return out


class Denoise_down(nn.Cell):
    def __init__(self):
        super(Denoise_down, self).__init__()
        self.net1 = nn.Conv2d(
            in_channels=21, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )

    def construct(self, x):
        net1 = P.ReLU()(self.net1(x))
        net2 = P.ReLU()(self.net2(net1))
        out = self.net3(net2)
        return out


class Denoise_up(nn.Cell):
    def __init__(self):
        super(Denoise_up, self).__init__()
        self.net1 = nn.Conv2d(
            in_channels=25, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )

    def construct(self, x):
        net1 = P.ReLU()(self.net1(x))
        net2 = P.ReLU()(self.net2(net1))
        out = self.net3(net2)
        return out


class Refine(nn.Cell):
    def __init__(self):
        super(Refine, self).__init__()
        self.net1 = nn.Conv2d(
            in_channels=33, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net2 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.net3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )

    def construct(self, x):
        net1 = P.ReLU()(self.net1(x))
        net2 = P.ReLU()(self.net2(net1))
        out = P.Sigmoid()(self.net3(net2))
        return out


class VideoDenoise(nn.Cell):
    def __init__(self):
        super(VideoDenoise, self).__init__()

        self.fusion = Fusion_down()
        self.denoise = Denoise_down()

    def construct(self, ft0, ft1, coeff_a, coeff_b):
        ll0 = ft0[:, 0:4, :, :]
        ll1 = ft1[:, 0:4, :, :]

        # fusion
        sigma_ll1 = mindspore.ops.clip_by_value(ll1[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = P.Concat(1)([(ll1 - ll0).abs(), sigma_ll1])
        gamma = self.fusion(fusion_in)
        fusion_out = mindspore.ops.Mul()(ft0, (1 - gamma)) + mindspore.ops.Mul()(ft1, gamma)

        # denoise
        sigma_ll0 = mindspore.ops.clip_by_value(ll0[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = P.Concat(1)([fusion_out, ll1, sigma])
        denoise_out = self.denoise(denoise_in)
        return gamma, denoise_out


class MultiVideoDenoise(nn.Cell):
    def __init__(self):
        super(MultiVideoDenoise, self).__init__()
        self.fusion = Fusion_up()
        self.denoise = Denoise_up()

    def construct(self, ft0, ft1, gamma_up, denoise_down, coeff_a, coeff_b):
        ll0 = ft0[:, 0:4, :, :]
        ll1 = ft1[:, 0:4, :, :]

        # fusion
        sigma_ll1 = mindspore.ops.clip_by_value(ll1[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = P.Concat(1)([(ll1 - ll0).abs(), gamma_up, sigma_ll1])
        gamma = self.fusion(fusion_in)
        fusion_out = mindspore.ops.Mul()(ft0, (1 - gamma)) + mindspore.ops.Mul()(ft1, gamma)

        # denoise
        sigma_ll0 = mindspore.ops.clip_by_value(ll0[:, 0:1, :, :], 0, 1) * coeff_a + coeff_b
        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = P.Concat(1)([fusion_out, denoise_down, ll1, sigma])
        denoise_out = self.denoise(denoise_in)

        return gamma, fusion_out, denoise_out, sigma


class EMVDNet(nn.Cell):
    def __init__(self):
        super(EMVDNet, self).__init__()
        self.ct = ColorTransfer()
        self.cti = ColorTransferInv()
        self.ft = FreTransfer()
        self.vd = VideoDenoise()
        self.fti = FreTransferInv()
        self.fti1 = FreTransferInv_weight()
        self.fti2 = FreTransferInv_weight()
        self.res = nn.ResizeBilinear()
        self.md1 = MultiVideoDenoise()
        self.md0 = MultiVideoDenoise()
        self.refine = Refine()

    def transform(self, x):
        net1 = self.ct(x)
        out = self.ft(net1)
        return out

    def transforminv(self, x, w1, w2):
        net1 = self.fti2(x, w1, w2)
        out = self.cti(net1)
        return out

    def construct(self, x, coeff_a=1, coeff_b=1):
        ft0 = x[:, 0:4, :, :]  # 1*4*128*128, the t-1 fusion frame
        ft1 = x[:, 4:8, :, :]  # 1*4*128*128, the t frame

        ft0_d0 = self.transform(ft0)  # scale0, torch.Size([1, 16, 256, 256])
        ft1_d0 = self.transform(ft1)

        ft0_d1 = self.ft(ft0_d0[:, 0:4, :, :])  # scale1,torch.Size([1, 16, 128, 128])
        ft1_d1 = self.ft(ft1_d0[:, 0:4, :, :])

        ft0_d2 = self.ft(ft0_d1[:, 0:4, :, :])  # scale2, torch.Size([1, 16, 64, 64])
        ft1_d2 = self.ft(ft1_d1[:, 0:4, :, :])

        gamma, denoise_out = self.vd(ft0_d2, ft1_d2, coeff_a, coeff_b)

        denoise_out_d2 = self.fti(denoise_out)

        gamma_up_d2 = self.res(gamma, scale_factor=2)

        gamma, fusion_out, denoise_out, sigma = self.md1(ft0_d1, ft1_d1, gamma_up_d2, denoise_out_d2, coeff_a, coeff_b)
        denoise_up_d1 = self.fti1(denoise_out, self.fti.net1_weight, self.fti.net2_weight)

        gamma_up_d1 = self.res(gamma, scale_factor=2)
        gamma, fusion_out, denoise_out, sigma = self.md0(ft0_d0, ft1_d0, gamma_up_d1, denoise_up_d1, coeff_a, coeff_b)

        # # # # # refine
        refine_in = P.Concat(1)([fusion_out, denoise_out, sigma])  # 1 * 36 * 128 * 128
        omega = self.refine(refine_in)  # 1 * 16 * 128 * 128
        refine_out = mindspore.ops.Mul()(denoise_out, (1 - omega)) + mindspore.ops.Mul()(fusion_out, omega)

        fusion_out = self.transforminv(fusion_out, self.fti.net1_weight, self.fti.net2_weight)
        refine_out = self.transforminv(refine_out, self.fti.net1_weight, self.fti.net2_weight)
        denoise_out = self.transforminv(denoise_out, self.fti.net1_weight, self.fti.net2_weight)

        return gamma, fusion_out, denoise_out, omega, refine_out


class EMVDWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn, frame_num):
        super(EMVDWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.frame_num = frame_num

    def construct(self, in_tensor, gt_tensor, coeff_a, coeff_b):
        # 0-iteration
        ft1 = in_tensor[:, :4, :, :]  # the t-th input frame
        fgt = gt_tensor[:, :4, :, :]  # the t-th gt frame

        ft0_fusion = ft1

        _input = mindspore.ops.Concat(1)((ft0_fusion, ft1))
        gamma, fusion_out, denoise_out, omega, refine_out = self._backbone(_input, coeff_a, coeff_b)

        l1loss_list = []
        l1loss_total = 0

        loss_refine = self._loss_fn(refine_out, fgt)
        l1loss = loss_refine
        l1loss_list.append(l1loss)
        l1loss_total += l1loss
        ft0_fusion_data = fusion_out

        for time_ind in range(1, self.frame_num):
            # the t-th input frame
            ft1 = in_tensor[:, time_ind * 4 : (time_ind + 1) * 4, :, :]
            # the t-th gt frame
            fgt = gt_tensor[:, time_ind * 4 : (time_ind + 1) * 4, :, :]
            ft0_fusion = ft0_fusion_data  # the t-1 fusion fram

            _input = mindspore.ops.Concat(1)((ft0_fusion, ft1))
            gamma, fusion_out, denoise_out, omega, refine_out = self._backbone(_input, coeff_a, coeff_b)

            loss_refine = self._loss_fn(refine_out, fgt)
            l1loss = loss_refine
            l1loss_list.append(l1loss)
            l1loss_total += l1loss
            ft0_fusion_data = fusion_out

        loss_ct = Tensor(0.0, mindspore.float32)
        ct = self._backbone.ct.net1.weight.squeeze()
        cti = self._backbone.cti.net1.weight.squeeze()
        weight_squared = P.MatMul()(ct, cti)
        diag = P.Eye()(weight_squared.shape[0], weight_squared.shape[0], mindspore.float32)
        curr_loss = P.ReduceSum()((weight_squared - diag) ** 2)
        loss_ct += curr_loss

        loss_ft = Tensor(0.0, mindspore.float32)
        ft = self._backbone.ft.net1.weight.squeeze()
        fti = mindspore.ops.Concat()((self._backbone.fti.net1_weight, self._backbone.fti.net2_weight)).squeeze()
        weight_squared = P.MatMul()(ft, fti)
        diag = P.Eye()(weight_squared.shape[1], weight_squared.shape[1], mindspore.float32)
        curr_loss = P.ReduceSum()((weight_squared - diag) ** 2)
        loss_ft += curr_loss

        total_loss = l1loss_total / (self.frame_num) + loss_ct + loss_ft
        return total_loss


class EMVD_post(nn.Cell):
    def __init__(self, backbone, frame_num):
        super(EMVD_post, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.frame_num = frame_num
        self.ft0_fusion_data = None

    def construct(self, in_tensor, gt_tensor, coeff_a, coeff_b, idx):
        if idx[0] % self.frame_num == 0 or self.ft0_fusion_data == None:
            ft0_fusion = in_tensor
        else:
            ft0_fusion = Tensor(self.ft0_fusion_data)
        _input = mindspore.ops.Concat(1)((ft0_fusion, in_tensor))
        gamma, fusion_out, denoise_out, omega, refine_out = self._backbone(_input, coeff_a, coeff_b)
        self.ft0_fusion_data = fusion_out
        return refine_out, gt_tensor
