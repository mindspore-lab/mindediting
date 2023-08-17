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
import mindspore.ops as ops
from mindspore import nn

from mindediting.models.common.deform_conv import build_deform_conv
from mindediting.models.mutil_task.vrt import ModulatedDeformConv


class FGDCL(ModulatedDeformConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        feature_in_channels,
        feature_mid_channels,
        cascade_channels=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deformable_groups=deform_groups,
            bias=bias,
        )

        self.conv1 = nn.SequentialCell(
            [
                nn.Conv2d(
                    feature_in_channels,
                    feature_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_mode="pad",
                    has_bias=True,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    feature_mid_channels,
                    feature_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_mode="pad",
                    has_bias=True,
                ),
                nn.ReLU(),
            ]
        )

        mid_channels = feature_mid_channels
        if cascade_channels is not None:
            mid_channels += cascade_channels
        self.conv2 = nn.SequentialCell(
            [
                nn.Conv2d(
                    mid_channels,
                    feature_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_mode="pad",
                    has_bias=True,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    feature_mid_channels,
                    feature_mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pad_mode="pad",
                    has_bias=True,
                ),
                nn.ReLU(),
            ]
        )

        self.conv_offset = nn.SequentialCell(
            [
                nn.Conv2d(
                    feature_mid_channels,
                    deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    pad_mode="pad",
                    dilation=self.dilation,
                    has_bias=True,
                    weight_init="zeros",
                    bias_init="zeros",
                )
            ]
        )
        self.conv_mask = nn.SequentialCell(
            [
                nn.Conv2d(
                    feature_mid_channels,
                    deform_groups * self.kernel_size[0] * self.kernel_size[1],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    pad_mode="pad",
                    dilation=self.dilation,
                    has_bias=True,
                    weight_init="zeros",
                    bias_init="zeros",
                ),
                nn.Sigmoid(),
            ]
        )

        self.reverse_flow = ops.ReverseV2([1])
        self.deformable_conv = build_deform_conv(
            deformable_groups=deform_groups,
            strides=(1, self.stride, self.stride, 1),
            padding=(1, self.padding, self.padding, 1),
            dilations=(1, self.dilation, self.dilation, 1),
            kernel_size=self.kernel_size,
            modulated=True,
        )

    @staticmethod
    def _convert_offsets_from_pt_ms(offsets):
        """
        This method does conversion of offsets used in PyTorch to offsets used in MindSpore.
        PyTorch offsets shape: [B, GROUPS x Hf x Wf x 2, Hout, Wout]
        MindSpore offsets shape: [B, 2 x GROUPS x Hf x Wf, Hout, Wout]
        Where the '2' corresponds to coordinates. Moreover, order of offset coordinates in Pytorch is (y, x),
            in MindSpore: it is (x, y).
        """

        b, _, h, w = offsets.shape

        offsets = offsets.reshape(b, -1, 2, h, w)
        offsets_x = offsets[:, :, 1]
        offsets_y = offsets[:, :, 0]

        return offsets_x, offsets_y

    def construct(self, x, fea_ref_warp, fea_rel_warp, flow=None, fea_cascade=None):
        fea_weight = ops.concat([fea_ref_warp, fea_rel_warp], axis=1)
        fea_weight = self.conv1(fea_weight)

        if fea_cascade is not None:
            fea_weight = ops.concat([fea_weight, fea_cascade], axis=1)
        fea_weight = self.conv2(fea_weight)

        offset = self.conv_offset(fea_weight)
        if flow is not None:
            reverse_flow = self.reverse_flow(flow)
            offset = offset + ms.numpy.tile(reverse_flow, (1, offset.shape[1] // 2, 1, 1))

        offset_x, offset_y = self._convert_offsets_from_pt_ms(offset)
        mask = self.conv_mask(fea_weight)

        out = self.deformable_conv(x, offset_x, offset_y, mask, self.weight, self.bias)
        out = fea_ref_warp + out

        return out, fea_weight
