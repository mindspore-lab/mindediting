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
import numpy as np
from mindspore import nn

from mindediting.models.common.convnextv2 import ConvNeXt
from mindediting.models.common.deform_conv import ModulatedDeformConvPack2D
from mindediting.models.common.fgdcl import FGDCL
from mindediting.models.common.flow_warp import build_flow_warp
from mindediting.models.common.grid_net import GridNet
from mindediting.models.common.prely import PReLU_PT as PReLU
from mindediting.models.common.resize import BilinearResize
from mindediting.models.common.se_block import SEBlock
from mindediting.utils.utils import cast_module


def conv_act(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.SequentialCell(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, "pad", padding, dilation, groups, has_bias=bias),
        PReLU(out_channels),
    )


class ResBlock(nn.Cell):
    def __init__(self, in_channels, side_channels, bias=True, enable_se=True):
        super().__init__()

        self.side_channels = side_channels

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=bias),
            PReLU(in_channels),
        )
        self.conv2 = nn.SequentialCell(
            ModulatedDeformConvPack2D(
                side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias, deformable_groups=4
            ),
            PReLU(side_channels),
            SEBlock(side_channels) if enable_se else nn.Identity(),
        )
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=bias),
            PReLU(in_channels),
        )
        self.conv4 = nn.SequentialCell(
            ModulatedDeformConvPack2D(
                side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias, deformable_groups=4
            ),
            PReLU(side_channels),
            SEBlock(side_channels) if enable_se else nn.Identity(),
        )
        self.conv5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=bias
        )
        self.prelu = PReLU(in_channels)

        self.attention = SEBlock(in_channels) if enable_se else None

    def construct(self, x):
        y = self.conv1(x)

        y1 = y[:, : -self.side_channels]
        y2 = y[:, -self.side_channels :]
        y2 = self.conv2(y2)

        y = ops.concat([y1, y2], axis=1)
        y = self.conv3(y)

        y1 = y[:, : -self.side_channels]
        y2 = y[:, -self.side_channels :]
        y2 = self.conv4(y2)

        y = ops.concat([y1, y2], axis=1)
        y = self.conv5(y)

        if self.attention is not None:
            y = self.attention(y)

        out = x + y
        out = self.prelu(out)

        return out


class ConvNeXTV2Encoder(nn.Cell):
    cfg = dict(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384))

    def __init__(self, input_channels, pretrained=None, **kwargs):
        super().__init__()

        model = ConvNeXt(
            in_chans=input_channels, stem_kernel=4, stem_stride=2, stem_padding=1, pretrained=pretrained, **self.cfg
        )

        self.stem = model.stem
        self.pyramid1 = nn.SequentialCell([model.stages[0]])
        self.pyramid2 = nn.SequentialCell([model.stages[1]])
        self.pyramid3 = nn.SequentialCell([model.stages[2]])
        self.pyramid4 = nn.SequentialCell([model.stages[3]])

    @property
    def out_channels(self):
        return self.cfg["dims"]

    def construct(self, img):
        y = self.stem(img)

        out_features = []
        out_features.append(self.pyramid1(y))
        out_features.append(self.pyramid2(out_features[-1]))
        out_features.append(self.pyramid3(out_features[-1]))
        out_features.append(self.pyramid4(out_features[-1]))

        return out_features


class Adapter(nn.Cell):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        assert len(input_channels) == len(output_channels)

        self.conv = nn.CellList(
            [conv_act(in_ch, out_ch, 1, 1, 0) for in_ch, out_ch in zip(input_channels, output_channels)]
        )

    def construct(self, inputs):
        return [self.conv[i](x) for i, x in enumerate(inputs)]


class BaseDecoder(nn.Cell):
    _cfgs = dict(internal=dict(channel_mul=2, channel_add=1), default=dict(channel_mul=3, channel_add=0))

    def __init__(self, feature_channels, side_channels, out_channels, mode="internal"):
        super(BaseDecoder, self).__init__()

        assert mode in self._cfgs.keys()
        cfg = self._cfgs[mode]

        if isinstance(out_channels, (tuple, list)):
            self.out_channels = out_channels
        else:
            self.out_channels = [out_channels]
        assert len(out_channels) > 0
        self.out_channels_total = sum(self.out_channels)
        self.out_channels_cumsum = np.cumsum(self.out_channels)[:-1].tolist()

        feature_channels = cfg["channel_mul"] * feature_channels
        in_channels = feature_channels + cfg["channel_add"]

        layers = [
            conv_act(in_channels, feature_channels),
            ResBlock(feature_channels, side_channels),
        ]
        self.convblock = nn.SequentialCell(*layers)

        self.upsample = nn.Conv2dTranspose(
            feature_channels, self.out_channels_total, 4, 2, padding=1, pad_mode="pad", has_bias=True
        )

    def _process_output(self, x):
        return ms.numpy.split(x, self.out_channels_cumsum, 1)


class InternalDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super(InternalDecoder, self).__init__(*args, **kwargs, mode="internal")

    def construct(self, f0, f1, emb_t):
        b, _, h, w = f0.shape

        emb_t = ms.numpy.tile(emb_t, (b, 1, h, w))
        f_in = ops.concat([f0, f1, emb_t], axis=1)

        f_out = self.convblock(f_in)
        f_out = self.upsample(f_out)

        out = self._process_output(f_out)

        return out


class DefaultDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super(DefaultDecoder, self).__init__(*args, **kwargs, mode="default")

        self.warp = build_flow_warp()

    def construct(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = self.warp(f0, up_flow0)
        f1_warp = self.warp(f1, up_flow1)
        f_in = ops.concat([ft_, f0_warp, f1_warp], axis=1)

        f_out = self.convblock(f_in)
        f_out = self.upsample(f_out)

        out = self._process_output(f_out)

        return out


class ConvNeXtV2RefineEncoder(nn.Cell):
    def __init__(self, out_channels, depths=(2, 2, 2)):
        super().__init__()

        assert len(out_channels) == 3
        assert len(depths) == 3

        model = ConvNeXt(
            in_chans=3, stem_kernel=3, stem_stride=1, stem_padding=1, depths=depths, dims=out_channels, num_stages=3
        )
        self.stem = model.stem
        self.pyramid1 = nn.SequentialCell([model.stages[0]])
        self.pyramid2 = nn.SequentialCell([model.stages[1]])
        self.pyramid3 = nn.SequentialCell([model.stages[2]])

    def construct(self, img):
        y = self.stem(img)

        f1 = self.pyramid1(y)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)

        return f1, f2, f3


class FGDCNRefiner(nn.Cell):
    def __init__(self, input_channels=[64, 96, 144], fgdcl_channels=[24, 48, 96], mfsn_channels=[32, 64, 96]):
        super().__init__()

        assert len(input_channels) == 3
        assert len(mfsn_channels) == 3

        self.fgdcl_0_1 = FGDCL(
            input_channels[0],
            input_channels[0],
            3,
            2 * input_channels[0],
            fgdcl_channels[0],
            padding=1,
            deform_groups=8,
            cascade_channels=fgdcl_channels[1],
        )
        self.fgdcl_1_1 = FGDCL(
            input_channels[0],
            input_channels[0],
            3,
            2 * input_channels[0],
            fgdcl_channels[0],
            padding=1,
            deform_groups=8,
            cascade_channels=fgdcl_channels[1],
        )

        self.fgdcl_0_2 = FGDCL(
            input_channels[1],
            input_channels[1],
            3,
            2 * input_channels[1],
            fgdcl_channels[1],
            padding=1,
            deform_groups=8,
            cascade_channels=fgdcl_channels[2],
        )
        self.fgdcl_1_2 = FGDCL(
            input_channels[1],
            input_channels[1],
            3,
            2 * input_channels[1],
            fgdcl_channels[1],
            padding=1,
            deform_groups=8,
            cascade_channels=fgdcl_channels[2],
        )

        self.fgdcl_0_3 = FGDCL(
            input_channels[2],
            input_channels[2],
            3,
            2 * input_channels[2],
            fgdcl_channels[2],
            padding=1,
            deform_groups=8,
        )
        self.fgdcl_1_3 = FGDCL(
            input_channels[2],
            input_channels[2],
            3,
            2 * input_channels[2],
            fgdcl_channels[2],
            padding=1,
            deform_groups=8,
        )

        mfsn_input_channels = [3 * num_channels for num_channels in input_channels]
        mfsn_output_channels = 3
        self.mfsn = GridNet(mfsn_input_channels, mfsn_channels, mfsn_output_channels)

        self.downscale_quarter = BilinearResize(0.25, align_corners=False)
        self.downscale_half = BilinearResize(0.5, align_corners=False)
        self.upscale_double = BilinearResize(2, align_corners=False)

        self.warp = build_flow_warp()

    def construct(self, features_0, features_1, features_t, flow0_1, flow1_1):
        f0_1, f0_2, f0_3 = features_0
        f1_1, f1_2, f1_3 = features_1
        ft_1, ft_2, ft_3 = features_t

        # scale 1/4
        flow0_3 = 0.25 * self.downscale_quarter(flow0_1)
        flow1_3 = 0.25 * self.downscale_quarter(flow1_1)
        f0_3_warp = self.warp(f0_3, flow0_3)
        f1_3_warp = self.warp(f1_3, flow1_3)
        f0_3_refined, f0_3_cascade = self.fgdcl_0_3(f0_3, f0_3_warp, f1_3_warp, flow0_3)
        f1_3_refined, f1_3_cascade = self.fgdcl_1_3(f1_3, f1_3_warp, f0_3_warp, flow1_3)
        f3_refined = ops.concat([f0_3_refined, f1_3_refined, ft_3], axis=1)

        # scale 1/2
        flow0_2 = 0.5 * self.downscale_half(flow0_1)
        flow1_2 = 0.5 * self.downscale_half(flow1_1)
        f0_2_warp = self.warp(f0_2, flow0_2)
        f1_2_warp = self.warp(f1_2, flow1_2)
        f0_2_refined, f0_2_cascade = self.fgdcl_0_2(
            f0_2, f0_2_warp, f1_2_warp, flow0_2, self.upscale_double(f0_3_cascade)
        )
        f1_2_refined, f1_2_cascade = self.fgdcl_1_2(
            f1_2, f1_2_warp, f0_2_warp, flow1_2, self.upscale_double(f1_3_cascade)
        )
        f2_refined = ops.concat([f0_2_refined, f1_2_refined, ft_2], axis=1)

        # scale 1/1
        f0_1_warp = self.warp(f0_1, flow0_1)
        f1_1_warp = self.warp(f1_1, flow1_1)
        f0_1_refined, _ = self.fgdcl_0_1(f0_1, f0_1_warp, f1_1_warp, flow0_1, self.upscale_double(f0_2_cascade))
        f1_1_refined, _ = self.fgdcl_1_1(f1_1, f1_1_warp, f0_1_warp, flow1_1, self.upscale_double(f1_2_cascade))
        f1_refined = ops.concat([f0_1_refined, f1_1_refined, ft_1], axis=1)

        # multiscale fusing
        out = self.mfsn(f1_refined, f2_refined, f3_refined)

        return out


class IFRNetPlus(nn.Cell):
    """Improved version of IFRNet architecture for video frame interpolation.

    Original paper:
        IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation
    Ref repo: https://github.com/ltkong218/ifrnet
    """

    def __init__(
        self,
        input_channels,
        decoder_channels,
        side_channels,
        refiner_channels,
        encoder_pretrained=None,
        flow_scale_factor=1.0,
        refiner_scale_factor=1.0,
        multiflow_fusing=True,
    ):
        super().__init__()

        self.flow_scale_factor = flow_scale_factor
        self.refiner_scale_factor = refiner_scale_factor
        self.multiflow_fusing = multiflow_fusing
        self.input_channels = input_channels
        self.flow_up_chs = 2 + 1
        self.num_flows = 5 if self.multiflow_fusing else 1
        self.mask_chs = 1 if self.multiflow_fusing else 2 * (3**2)

        self.flow_downscale = BilinearResize(self.flow_scale_factor, align_corners=False)
        self.flow_upscale = BilinearResize(1.0 / self.flow_scale_factor, align_corners=False)
        self.refiner_downscale = BilinearResize(self.refiner_scale_factor, align_corners=False)
        self.refiner_upscale = BilinearResize(1.0 / self.refiner_scale_factor, align_corners=False)
        self.upscale_double = BilinearResize(2.0, align_corners=False)
        self.warp = build_flow_warp()
        self.unfold = nn.Unfold(
            ksizes=(1, 3, 3, 1),
            strides=(1, 1, 1, 1),
            rates=(1, 1, 1, 1),
            padding="same",
        )

        self.encoder = ConvNeXTV2Encoder(self.input_channels, pretrained=encoder_pretrained)
        self.adapter = Adapter(self.encoder.out_channels, decoder_channels)

        self.decoder4 = InternalDecoder(
            feature_channels=decoder_channels[3],
            side_channels=side_channels,
            out_channels=[2, 2, decoder_channels[2]],
        )
        self.decoder3 = DefaultDecoder(
            feature_channels=decoder_channels[2],
            side_channels=side_channels,
            out_channels=[2, 2, self.flow_up_chs, self.flow_up_chs, decoder_channels[1]],
        )
        self.decoder2 = DefaultDecoder(
            feature_channels=decoder_channels[1],
            side_channels=side_channels,
            out_channels=[2, 2, self.flow_up_chs, self.flow_up_chs, decoder_channels[0]],
        )
        self.decoder1 = DefaultDecoder(
            feature_channels=decoder_channels[0],
            side_channels=side_channels,
            out_channels=[
                2 * self.num_flows,
                2 * self.num_flows,
                self.flow_up_chs * self.num_flows,
                self.flow_up_chs * self.num_flows,
                self.mask_chs * self.num_flows,
                self.input_channels * self.num_flows,
            ],
        )

        self.comb_block = None
        if self.multiflow_fusing:
            internal_comb_size = 2 * input_channels * self.num_flows
            self.comb_block = nn.SequentialCell(
                nn.Conv2d(input_channels * self.num_flows, internal_comb_size, 7, 1, "pad", 3, has_bias=True),
                PReLU(internal_comb_size),
                nn.Conv2d(internal_comb_size, input_channels, 7, 1, "pad", 3, has_bias=True),
            )

        self.refiner_encoder = ConvNeXtV2RefineEncoder(refiner_channels)
        self.refine_adapter = Adapter(refiner_channels, refiner_channels)
        self.refiner = FGDCNRefiner(input_channels=refiner_channels)

        self.emb_t = ms.Tensor([0.5]).reshape(1, 1, 1, 1)
        self.mean = ms.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = ms.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    def _image_to_tensor(self, img):
        return (img - self.mean) / self.std

    def _tensor_to_image(self, tensor):
        return self.std * tensor + self.mean

    def _single_flow_combine(self, img0, img1, flow0, flow1, mask_logits, img_res, k=3):
        _, c, h, w = img0.shape

        img0_warp = self.warp(img0, flow0)
        img1_warp = self.warp(img1, flow1)

        mask = ops.softmax(mask_logits, axis=1)
        mask = mask.reshape(-1, 2 * k * k, 1, h, w)

        input0_unfold = self.unfold(img0_warp).reshape(-1, k * k, c, h, w)
        input1_unfold = self.unfold(img1_warp).reshape(-1, k * k, c, h, w)
        weighted_unfold = ops.concat([input0_unfold, input1_unfold], 1) * mask
        img_t_merge = ops.reduce_sum(weighted_unfold, 1)

        imgt_pred = img_t_merge + img_res

        return imgt_pred

    def _multi_flow_combine(self, img0, img1, flow0, flow1, mask_logits, img_res):
        b, c, h, w = flow0.shape
        num_flows = c // 2

        flow0 = flow0.reshape(-1, 2, h, w)
        flow1 = flow1.reshape(-1, 2, h, w)
        mask = ops.sigmoid(mask_logits.reshape(-1, self.mask_chs, h, w))
        img_res = img_res.reshape(-1, self.input_channels, h, w)

        img0 = ops.stack([img0] * num_flows, 1).reshape(-1, self.input_channels, h, w)
        img1 = ops.stack([img1] * num_flows, 1).reshape(-1, self.input_channels, h, w)

        img0_warp = self.warp(img0, flow0)
        img1_warp = self.warp(img1, flow1)

        img_warps = mask * img0_warp + (1.0 - mask) * img1_warp + img_res
        img_warps = img_warps.reshape(b, num_flows, 3, h, w)

        imgt_pred = img_warps.mean(1) + self.comb_block(img_warps.view(b, -1, h, w))

        return imgt_pred

    def _upsample_single_flow(self, flow, inter_flow, inter_mask):
        flow_init = 2.0 * self.upscale_double(flow)

        inter_mask = ops.sigmoid(inter_mask)
        out = self.warp(flow_init, inter_flow) * (1.0 - inter_mask) + flow_init * inter_mask

        return out

    def _upsample_multi_flow(self, flow, inter_flow, inter_mask):
        h_original, w_original = flow.shape[-2:]
        num_flows = inter_flow.shape[1] // 2

        flow = ops.stack([flow] * num_flows, 1).reshape(-1, 2, h_original, w_original)
        flow_init = 2.0 * self.upscale_double(flow)

        h, w = inter_flow.shape[-2:]
        inter_flow = inter_flow.reshape(-1, 2, h, w)
        inter_mask = ops.sigmoid(inter_mask.reshape(-1, 1, h, w))

        out = self.warp(flow_init, inter_flow) * (1.0 - inter_mask) + flow_init * inter_mask
        out = out.reshape(-1, 2 * num_flows, h, w)

        return out

    def construct(self, img0, img1):
        norm_img0 = self._image_to_tensor(img0)
        norm_img1 = self._image_to_tensor(img1)

        if self.flow_scale_factor != 1.0:
            flow_input_img0 = self.flow_downscale(norm_img0)
            flow_input_img1 = self.flow_downscale(norm_img1)
        else:
            flow_input_img0 = norm_img0
            flow_input_img1 = norm_img1

        f0_original = self.encoder(flow_input_img0)
        f1_original = self.encoder(flow_input_img1)

        f0_1, f0_2, f0_3, f0_4 = self.adapter(f0_original)
        f1_1, f1_2, f1_3, f1_4 = self.adapter(f1_original)

        flow0_4, flow1_4, ft_3 = self.decoder4(f0_4, f1_4, self.emb_t)

        flow_diff0_3, flow_diff1_3, flow_upflow0_3, flow_upflow1_3, ft_2 = self.decoder3(
            ft_3, f0_3, f1_3, flow0_4, flow1_4
        )
        flow0_3 = flow_diff0_3 + self._upsample_single_flow(flow0_4, flow_upflow0_3[:, 0:2], flow_upflow0_3[:, 2:3])
        flow1_3 = flow_diff1_3 + self._upsample_single_flow(flow1_4, flow_upflow1_3[:, 0:2], flow_upflow1_3[:, 2:3])

        flow_diff0_2, flow_diff1_2, flow_upflow0_2, flow_upflow1_2, ft_1 = self.decoder2(
            ft_2, f0_2, f1_2, flow0_3, flow1_3
        )
        flow0_2 = flow_diff0_2 + self._upsample_single_flow(flow0_3, flow_upflow0_2[:, 0:2], flow_upflow0_2[:, 2:3])
        flow1_2 = flow_diff1_2 + self._upsample_single_flow(flow1_3, flow_upflow1_2[:, 0:2], flow_upflow1_2[:, 2:3])

        flow_diff0_1, flow_diff1_1, flow_upflow0_1, flow_upflow1_1, merge_mask, image_res = self.decoder1(
            ft_1, f0_1, f1_1, flow0_2, flow1_2
        )

        if self.multiflow_fusing:
            flow_upflow0_1_flow, flow_upflow0_1_mask = ms.numpy.split(
                flow_upflow0_1, [(self.flow_up_chs - 1) * self.num_flows], 1
            )
            flow_upflow1_1_flow, flow_upflow1_1_mask = ms.numpy.split(
                flow_upflow1_1, [(self.flow_up_chs - 1) * self.num_flows], 1
            )
            flow0_1 = flow_diff0_1 + self._upsample_multi_flow(flow0_2, flow_upflow0_1_flow, flow_upflow0_1_mask)
            flow1_1 = flow_diff1_1 + self._upsample_multi_flow(flow1_2, flow_upflow1_1_flow, flow_upflow1_1_mask)
        else:
            flow0_1 = flow_diff0_1 + self._upsample_single_flow(flow0_2, flow_upflow0_1[:, 0:2], flow_upflow0_1[:, 2:3])
            flow1_1 = flow_diff1_1 + self._upsample_single_flow(flow1_2, flow_upflow1_1[:, 0:2], flow_upflow1_1[:, 2:3])

        if self.flow_scale_factor != 1.0:
            flow0_1 = (1.0 / self.flow_scale_factor) * self.flow_upscale(flow0_1)
            flow1_1 = (1.0 / self.flow_scale_factor) * self.flow_upscale(flow1_1)
            merge_mask = self.flow_upscale(merge_mask)
            image_res = self.flow_upscale(image_res)

        if self.multiflow_fusing:
            img_t_update = self._multi_flow_combine(norm_img0, norm_img1, flow0_1, flow1_1, merge_mask, image_res)
            flow0_1 = flow0_1.reshape(-1, self.num_flows, 2, *flow0_1.shape[-2:]).mean(1)
            flow1_1 = flow1_1.reshape(-1, self.num_flows, 2, *flow1_1.shape[-2:]).mean(1)
        else:
            img_t_update = self._single_flow_combine(norm_img0, norm_img1, flow0_1, flow1_1, merge_mask, image_res)

        if self.refiner_scale_factor != 1.0:
            refiner_input_img0 = self.refiner_downscale(norm_img0)
            refiner_input_img1 = self.refiner_downscale(norm_img1)
            refiner_input_imgt = self.refiner_downscale(img_t_update)
            refiner_flow0_1 = self.refiner_scale_factor * self.refiner_downscale(flow0_1)
            refiner_flow1_1 = self.refiner_scale_factor * self.refiner_downscale(flow1_1)
        else:
            refiner_input_img0 = norm_img0
            refiner_input_img1 = norm_img1
            refiner_input_imgt = img_t_update
            refiner_flow0_1 = flow0_1
            refiner_flow1_1 = flow1_1

        f0_detailed = self.refiner_encoder(refiner_input_img0)
        f1_detailed = self.refiner_encoder(refiner_input_img1)
        ft_detailed = self.refiner_encoder(refiner_input_imgt)

        features_0 = self.refine_adapter(f0_detailed)
        features_1 = self.refine_adapter(f1_detailed)
        features_t = self.refine_adapter(ft_detailed)

        img_t_refine_diff = self.refiner(features_0, features_1, features_t, refiner_flow0_1, refiner_flow1_1)

        if self.refiner_scale_factor != 1.0:
            img_t_refine_diff = self.refiner_upscale(img_t_refine_diff)

        img_t_refine = img_t_update + img_t_refine_diff

        img_pred = self._tensor_to_image(img_t_refine)
        flows_0 = [flow0_1, flow0_2, flow0_3, flow0_4]
        flows_1 = [flow1_1, flow1_2, flow1_3, flow1_4]

        return img_pred, flows_0, flows_1


class IFRPlus(nn.Cell):
    def __init__(
        self,
        input_channels=3,
        decoder_channels=[64, 96, 144, 192],
        side_channels=56,
        refiner_channels=[24, 48, 96],
        encoder_pretrained=None,
        to_float16=False,
        flow_scale_factor=1.0,
        refiner_scale_factor=1.0,
    ):
        super().__init__()

        self.generator = IFRNetPlus(
            input_channels=input_channels,
            decoder_channels=decoder_channels,
            side_channels=side_channels,
            refiner_channels=refiner_channels,
            encoder_pretrained=encoder_pretrained,
            flow_scale_factor=flow_scale_factor,
            refiner_scale_factor=refiner_scale_factor,
        )

        if to_float16:
            cast_module("float16", self)

    def construct(self, inputs):
        img0, img1 = inputs[:, 0], inputs[:, 1]
        img_pred, flows_0_pred, flows_1_pred = self.generator(img0, img1)

        if self.training:
            return img_pred, flows_0_pred, flows_1_pred
        else:
            return ops.clip_by_value(img_pred, 0.0, 1.0)
