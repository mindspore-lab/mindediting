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
import mindspore.nn as nn
import numpy as np
from mindspore.ops import operations as ops

from mindediting.models.common.interpolation import interpolate_through_grid_sample
from mindediting.models.common.pixel_shuffle_pack import PixelShufflePack
from mindediting.models.common.resblock_with_input_conv import ResidualBlocksWithInputConv
from mindediting.models.common.spynet import FlowWarp, SPyNet


class BasicVSRNet(nn.Cell):
    def __init__(
        self,
        mid_channels=64,
        num_blocks=30,
        spynet_pretrained=None,
        is_mirror_extended_train=False,
        is_mirror_extended_test=False,
        precompute_grid=True,
        base_resolution=([64, 64], [64, 112]),
        levels=1,
        sp_base_resolution=([64, 64], [64, 128]),
        sp_levels=1,
        eliminate_gradient_for_gather=False,
        preprocess=False,
        postprocess=False,
    ):
        super(BasicVSRNet, self).__init__()
        self.is_mirror_extended_train = is_mirror_extended_train
        self.is_mirror_extended_test = is_mirror_extended_test
        self.mid_channels = mid_channels
        # optical flow network for feature alignment
        self.spynet = SPyNet(
            pretrained=spynet_pretrained,
            precompute_grid=precompute_grid,
            base_resolution=sp_base_resolution,
            levels=sp_levels,
            eliminate_gradient_for_gather=eliminate_gradient_for_gather,
        )
        # propagation branches
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        # upsample
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, padding=0, has_bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, padding=1, pad_mode="pad", has_bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, padding=1, pad_mode="pad", has_bias=True)
        self.img_upsample = nn.ResizeBilinear(half_pixel_centers=False)
        # activation function
        self.lrelu = nn.LeakyReLU(alpha=0.1)
        self.chunk = ops.Split(axis=1, output_num=2)
        self.norm = ops.LpNorm(axis=[0, 1, 2, 3, 4])
        self.flip = ms.ops.ReverseV2(axis=[1])
        self.reshape = ops.Reshape()
        self.new_zeros = ops.Zeros()
        self.zeros_like = ops.ZerosLike()
        self.flow_warp = FlowWarp(
            padding_mode="zeros",
            precompute_grid=precompute_grid,
            base_resolution=base_resolution,
            levels=levels,
            eliminate_gradient_for_gather=eliminate_gradient_for_gather,
        )
        self.transpose = ops.Transpose()
        self.concat_1 = ops.Concat(axis=1)
        self.stack_1 = ops.Stack(axis=1)
        device = ms.get_context("device_target")
        self.interpolate_func = ms.ops.interpolate if device in ["GPU", "Ascend"] else interpolate_through_grid_sample
        self.preprocess = preprocess
        self.postprocess = postprocess

    def construct(self, lrs):
        if len(lrs.shape) == 4:
            lrs = ms.ops.expand_dims(lrs, axis=0)
        if self.preprocess:
            lrs = ms.ops.transpose(lrs, (0, 1, 4, 2, 3)) / 255.0
        dtype = lrs.dtype
        n, t, c, h, w = lrs.shape
        assert h >= 64 and w >= 64, "The height and width of inputs should be at least 64, " f"but got {h} and {w}."

        # compute optical flow
        lrs_1 = self.reshape(lrs[:, :-1, :, :, :], (-1, c, h, w))
        lrs_2 = self.reshape(lrs[:, 1:, :, :, :], (-1, c, h, w))

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if (self.training and self.is_mirror_extended_train) or (
            not self.training and self.is_mirror_extended_test
        ):  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        # backward-time propagation
        outputs = []
        feat_prop = self.new_zeros((n, self.mid_channels, h, w), dtype)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = self.flow_warp(feat_prop, self.transpose(flow, (0, 2, 3, 1)), 0)
            feat_prop = self.concat_1([lrs[:, i, :, :, :], feat_prop])
            feat_prop = self.backward_resblocks(feat_prop)
            outputs.append(feat_prop)

        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = self.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is None:
                    flow = flows_backward[:, -i, :, :, :]
                else:
                    flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = self.flow_warp(feat_prop, self.transpose(flow, (0, 2, 3, 1)), 0)

            feat_prop = self.concat_1([lr_curr, feat_prop])
            feat_prop = self.forward_resblocks(feat_prop)

            # upsampling given the backward and forward features
            out = self.concat_1([outputs[i], feat_prop])
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = self.interpolate_func(
                lr_curr, scales=(1.0, 1.0, 4.0, 4.0), coordinate_transformation_mode="half_pixel", mode="bilinear"
            )
            out += base
            if self.postprocess:
                out = out[0].clip(0, 1)
                out = (ms.ops.transpose(out, (1, 2, 0)) * 255.0).round()
            outputs[i] = out
        if self.postprocess:
            return ms.ops.stack(outputs, axis=0)
        else:
            return self.stack_1(outputs)


if __name__ == "__main__":
    f = BasicVSRNet()
    x = ms.Tensor(np.random.rand(1, 14, 3, 64, 112), ms.float32)
    y = f(x)
    print(x.shape, "->", y.shape)
