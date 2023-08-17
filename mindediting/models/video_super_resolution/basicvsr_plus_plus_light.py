# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as ops

from mindediting.models.common.interpolation import interpolate_through_grid_sample
from mindediting.models.common.pixel_shuffle_pack import PixelShufflePack
from mindediting.models.common.resblock_with_input_conv import ResidualBlocksWithInputConv


def act(act_type="leakyReLU"):
    if act_type == "ReLU":
        return nn.ReLU()
    elif act_type == "LeakyReLU":
        return nn.LeakyReLU(alpha=0.1)
    else:
        raise ValueError(f"Unknown activation type {act_type}")


class Alignment(nn.Cell):
    def __init__(self, in_channels, out_channels, num_blocks=5, has_bias=True, act=None):
        super().__init__()
        self.conv_block = ResidualBlocksWithInputConv(in_channels, in_channels, num_blocks, has_bias=has_bias, act=act)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, pad_mode="pad", has_bias=has_bias)

    def construct(self, x):
        out = self.conv_block(x)
        return self.conv(out)


class BasicVSRPlusPlusLightNet(nn.Cell):
    def __init__(
        self,
        mid_channels=64,
        num_blocks=7,
        num_blocks_align=5,
        is_low_res_input=True,
        preprocess=False,
        postprocess=False,
        upsample_blocks_last=True,
        has_bias=True,
        activation="LeakyReLU",
        custom_depth_to_space=False,
        pad_input=False,
    ):
        super().__init__()
        print("BasicVSR++Light")
        print(f"\tnum_blocks: {num_blocks}")
        print(f"\tnum_blocks_align: {num_blocks_align}")
        print(f"\tbias: {has_bias}")
        print(f"\tactivation: {activation}")
        print(f"\tpixel_shuffle: {upsample_blocks_last}")
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input

        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5, has_bias=has_bias, act=act(activation))
        else:
            self.feat_extract = nn.SequentialCell(
                nn.Conv2d(3, mid_channels, 3, 2, padding=1, pad_mode="pad", has_bias=has_bias),
                act(activation),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, padding=1, pad_mode="pad", has_bias=has_bias),
                act(activation),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5, has_bias=has_bias, act=act(activation)),
            )

        # propagation branches
        self.deform_align = {}
        self.backbone = {}

        self.deform_align_backward_1 = Alignment(
            in_channels=2 * mid_channels,
            out_channels=mid_channels,
            num_blocks=num_blocks_align,
            has_bias=has_bias,
            act=act(activation),
        )
        self.backbone_backward_1 = ResidualBlocksWithInputConv(
            (2 + 0) * mid_channels, mid_channels, num_blocks, has_bias=has_bias, act=act(activation)
        )

        self.deform_align_forward_1 = Alignment(
            in_channels=2 * mid_channels,
            out_channels=mid_channels,
            num_blocks=num_blocks_align,
            has_bias=has_bias,
            act=act(activation),
        )
        self.backbone_forward_1 = ResidualBlocksWithInputConv(
            (2 + 1) * mid_channels, mid_channels, num_blocks, has_bias=has_bias, act=act(activation)
        )

        self.deform_align_backward_2 = Alignment(
            in_channels=2 * mid_channels,
            out_channels=mid_channels,
            num_blocks=num_blocks_align,
            has_bias=has_bias,
            act=act(activation),
        )
        self.backbone_backward_2 = ResidualBlocksWithInputConv(
            (2 + 2) * mid_channels, mid_channels, num_blocks, has_bias=has_bias, act=act(activation)
        )

        self.deform_align_forward_2 = Alignment(
            in_channels=2 * mid_channels,
            out_channels=mid_channels,
            num_blocks=num_blocks_align,
            has_bias=has_bias,
            act=act(activation),
        )
        self.backbone_forward_2 = ResidualBlocksWithInputConv(
            (2 + 3) * mid_channels, mid_channels, num_blocks, has_bias=has_bias, act=act(activation)
        )

        self.deform_align["backward_1"] = self.deform_align_backward_1
        self.deform_align["forward_1"] = self.deform_align_forward_1
        self.deform_align["backward_2"] = self.deform_align_backward_2
        self.deform_align["forward_2"] = self.deform_align_forward_2

        self.backbone["backward_1"] = self.backbone_backward_1
        self.backbone["forward_1"] = self.backbone_forward_1
        self.backbone["backward_2"] = self.backbone_backward_2
        self.backbone["forward_2"] = self.backbone_forward_2

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5, has_bias=has_bias, act=act(activation)
        )
        self.upsample1 = PixelShufflePack(
            mid_channels,
            mid_channels,
            2,
            upsample_kernel=3,
            blocks_last=upsample_blocks_last,
            has_bias=has_bias,
            custom_depth_to_space=custom_depth_to_space,
        )
        self.upsample2 = PixelShufflePack(
            mid_channels,
            mid_channels,
            2,
            upsample_kernel=3,
            blocks_last=upsample_blocks_last,
            has_bias=has_bias,
            custom_depth_to_space=custom_depth_to_space,
        )

        self.pad_input = int(pad_input)
        self.pad_input_op = nn.Pad(
            ((0, 0), (0, 0), (self.pad_input * 2, self.pad_input * 2), (self.pad_input * 2, self.pad_input * 2)),
            "REFLECT",
        )
        self.conv_hr = nn.Conv2d(
            mid_channels, mid_channels, 3, 1, padding=1 - self.pad_input, pad_mode="pad", has_bias=has_bias
        )
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, padding=1 - self.pad_input, pad_mode="pad", has_bias=has_bias)

        self.img_upsample = nn.ResizeBilinear(half_pixel_centers=False)
        self.lrelu = act(activation)

        device = ms.get_context("device_target")
        self.interpolate_func = ms.ops.interpolate if device in ["GPU", "Ascend"] else interpolate_through_grid_sample

        self.new_zeros = ops.Zeros()
        self.zeros_like = ops.ZerosLike()

        self.concat_1 = ops.Concat(axis=1)
        self.stack_1 = ops.Stack(axis=1)

        self.preprocess = preprocess
        self.postprocess = postprocess

    def construct(self, lrs):
        if len(lrs.shape) == 4:
            lrs = ms.ops.expand_dims(lrs, axis=0)
        if self.preprocess:
            lrs = ms.ops.transpose(lrs, (0, 1, 4, 2, 3)) / 255.0
        n, t, c, h, w = lrs.shape
        lqs = lrs.view(-1, c, h, w)
        if self.pad_input > 0:
            lqs = self.pad_input_op(lqs)
            _, c, h, w = lqs.shape

        feats = {}
        # compute spatial features
        feats_ = self.feat_extract(lqs)
        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)
        feats["spatial"] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # feature propagation
        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"
                feats[module] = []
                feats = self.propagate(feats, module)

        return self.upsample(lrs, feats)

    def propagate(self, feats, module_name):
        n, _, h, w = feats["spatial"][0].shape
        dtype = feats["spatial"][0].dtype
        t = len(feats["spatial"])

        frame_idx = range(t)
        mapping_idx = list(range(t))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]

        feat_prop = self.new_zeros((n, self.mid_channels, h, w), dtype)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            # second-order deformable alignment
            if i > 0:
                # initialize second-order features
                if i < 2:
                    feat_n2 = self.zeros_like(feat_prop)
                else:  # second-order features
                    feat_n2 = feats[module_name][-2]
                # flow-guided deformable convolution
                feat_prop = self.concat_1([feat_prop, feat_n2])
                feat_prop = self.deform_align[module_name](feat_prop)

            # concatenate and residual blocks
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ["spatial", module_name]] + [feat_prop]

            feat = self.concat_1(feat)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats["spatial"])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.shape[1]):
            hr = [
                feats["spatial"][mapping_idx[i]],
            ]
            hr += [feats[k][i] for k in feats if k != "spatial"]
            hr = self.concat_1(hr)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.interpolate_func(
                    lqs[:, i, :, :, :],
                    scales=(1.0, 1.0, 4.0, 4.0),
                    coordinate_transformation_mode="half_pixel",
                    mode="bilinear",
                )
            else:
                hr += lqs[:, i, :, :, :]

            if self.postprocess:
                hr = hr[0].clip(0, 1)
                hr = (ms.ops.transpose(hr, (1, 2, 0)) * 255.0).round()

            outputs.append(hr)
        if self.postprocess:
            return ms.ops.stack(outputs, axis=0)
        else:
            return self.stack_1(outputs)
