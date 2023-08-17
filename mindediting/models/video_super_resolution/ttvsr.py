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

from mindediting.models.common.grid_sample import build_grid_sample
from mindediting.models.common.pixel_shuffle_pack import PixelShufflePack
from mindediting.models.common.resblock import ResidualBlockNoBN
from mindediting.models.common.space_to_depth import pixel_unshuffle
from mindediting.models.common.unfold import build_unfold
from mindediting.utils.checkpoint import load_param_into_net
from mindediting.utils.init_weights import init_weights, make_layer


def check_if_mirror_extended(lrs):
    """Check whether the input is a mirror-extended sequence.

    If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
    (t-1-i)-th frame.

    Args:
            lrs [tensor]: Input LR images with shape (n, t, c, h, w)
    """

    is_mirror_extended = False
    if lrs.shape[1] % 2 == 0:
        lrs_1, lrs_2 = ms.numpy.split(lrs, 2, axis=1)
        if ms.numpy.norm(lrs_1 - ms.numpy.flip(lrs_2, 1)) == 0:
            is_mirror_extended = True

    return is_mirror_extended


class TTVSRNet(nn.Cell):
    """TTVSR

    Support only x4 upsampling.
    Paper:
        Learning Trajectory-Aware Transformer for Video Super-Resolution, CVPR, 2022

    Args:
        mid_channels (int): Channel number of the intermediate features.
            Default: 64.
        num_blocks (int): Number of residual blocks in propagation branch.
            Default: 60.
        stride (int): the scale of tokens.
            Default: 4.
        frame_stride (int): Number determining the stride of frames. If frame_stride=3,
            then the (0, 3, 6, 9, ...)-th frame will be the slected frames.
            Default: 3.
        spynet_pretrained (str): Pre-trained model path of SPyNet.
            Default: None.
    """

    def __init__(
        self,
        mid_channels=64,
        num_blocks=60,
        stride=4,
        frame_stride=3,
        check_mirrored_input=False,
        spynet_pretrained=None,
    ):
        super().__init__()

        self.mid_channels = mid_channels
        self.keyframe_stride = frame_stride
        self.stride = stride
        self.check_mirrored_input = check_mirrored_input

        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        self.feat_extractor = ResidualBlocksWithInputConv(3, mid_channels, 5)
        self.LTAM = LTAM(stride=self.stride)

        # propagation branches
        self.resblocks = ResidualBlocksWithInputConv(2 * mid_channels, mid_channels, num_blocks)

        # upsample
        self.fusion = nn.Conv2d(3 * mid_channels, mid_channels, 1, stride=1, pad_mode="pad", padding=0, has_bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, stride=1, pad_mode="pad", padding=1, has_bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(alpha=0.1)

        # unfold operations
        self.sparse_s2_unfold = build_unfold(
            ksizes=(1, int(1.5 * self.stride), int(1.5 * self.stride), 1),
            strides=(1, self.stride, self.stride, 1),
            rates=(1, 1, 1, 1),
            padding="valid",
        )
        self.sparse_s3_unfold = build_unfold(
            ksizes=(1, int(2 * self.stride), int(2 * self.stride), 1),
            strides=(1, self.stride, self.stride, 1),
            rates=(1, 1, 1, 1),
            padding="valid",
        )

        # flow_warp operations
        self.flow_warp_nearest = FlowWarp(padding_mode="border", interpolation="nearest", align_corners=True)
        self.flow_warp_bilinear = FlowWarp(padding_mode="border", interpolation="bilinear", align_corners=True)

    def compute_flow(self, lrs, is_mirror_extended=False):
        """Compute optical flow using SPyNet for feature warping.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lrs [tensor]: Input LR images with shape (n, t, c, h, w)
            is_mirror_extended (bool): Whether the input is an extended sequence

        Return:
            tuple[Tensor]: Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lrs.shape
        lrs_1 = lrs[:, :-1, :, :, :].view(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].view(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def construct(self, lrs):
        """Main function for BasicVSR.

        Args:
            lrs [Tensor]: Input LR sequence with shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lrs.shape
        n, t, c, h, w = int(n), int(t), int(c), int(h), int(w)

        # check whether the input is an extended sequence
        is_mirror_extended = check_if_mirror_extended(lrs) if self.check_mirrored_input else False

        # compute optical flow
        flows_forward, flows_backward = self.compute_flow(lrs, is_mirror_extended)
        outputs = self.feat_extractor(lrs.view(-1, c, h, w)).view(n, t, -1, h, w)
        keyframe_idx_forward = tuple(range(0, t, self.keyframe_stride))
        keyframe_idx_backward = tuple(range(t - 1, 0, -self.keyframe_stride))

        # backward-time propagation
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []
        feat_prop = ops.zeros((n, self.mid_channels, h, w), lrs.dtype)
        grid_y, grid_x = ops.meshgrid(
            (ms.numpy.arange(h // self.stride), ms.numpy.arange(w // self.stride)), indexing="ij"
        )
        location_update = ops.stack([grid_x, grid_y], axis=0).astype(lrs.dtype).broadcast_to((n, -1, -1, -1))
        for i in range(t - 1, -1, -1):
            lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[:, i]
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = self.flow_warp_bilinear(feat_prop, flow.transpose(0, 2, 3, 1))

                flow = ops.avg_pool2d(flow, self.stride, self.stride) / self.stride
                location_update = self.flow_warp_nearest(location_update, flow.transpose(0, 2, 3, 1))  # n , 2t , h , w

                sparse_feat_buffer_s1 = ops.stack(sparse_feat_buffers_s1, axis=1)
                sparse_feat_buffer_s2 = ops.stack(sparse_feat_buffers_s2, axis=1)
                sparse_feat_buffer_s3 = ops.stack(sparse_feat_buffers_s3, axis=1)
                index_feat_buffer_s1 = ops.stack(index_feat_buffers_s1, axis=1)
                feat_prop = self.LTAM(
                    lr_curr_feat,
                    index_feat_buffer_s1,
                    feat_prop,
                    sparse_feat_buffer_s1,
                    sparse_feat_buffer_s2,
                    sparse_feat_buffer_s3,
                    location_update,
                )

                if i in keyframe_idx_backward:
                    location_update_step = ops.stack([grid_x, grid_y], axis=0)
                    location_update_step = location_update_step.astype(lrs.dtype)
                    location_update_step = location_update_step.broadcast_to((n, -1, -1, -1))
                    location_update = ops.concat([location_update, location_update_step], axis=1)  # n , 2t , h , w

            feat_prop = ops.concat([lr_curr_feat, feat_prop], axis=1)
            feat_prop = self.resblocks(feat_prop)
            feat_buffers.append(feat_prop)
            if i in keyframe_idx_backward:
                # bs * c * h * w --> # bs * (c*4*4) * h//4 * w//4
                index_feat_prop_s1 = pixel_unshuffle(lr_curr_feat, self.stride)
                index_feat_buffers_s1.append(index_feat_prop_s1)

                # feature tokenization *4
                # bs * c * h * w --> # bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s1 = pixel_unshuffle(feat_prop, self.stride)
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                # feature tokenization *6
                # bs * c * h * w --> # bs * (c*6*6) * h//4 * w//4 -->  bs * c * (h*1.5) * (w*1.5)
                pad = int(0.25 * self.stride)
                ksize = int(1.5 * self.stride)
                feat_prop_pad1 = ops.Pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))(feat_prop)
                sparse_feat_prop_s2 = self.sparse_s2_unfold(feat_prop_pad1)
                sparse_feat_prop_s2 = sparse_feat_prop_s2.reshape(
                    n, ksize, ksize, -1, h // self.stride, w // self.stride
                )
                sparse_feat_prop_s2 = sparse_feat_prop_s2.transpose(0, 3, 4, 1, 5, 2)
                sparse_feat_prop_s2 = sparse_feat_prop_s2.reshape(n, -1, int(1.5 * h), int(1.5 * w))
                # bs * c * (h*1.5) * (w*1.5) -->  bs * c * h * w
                sparse_feat_prop_s2 = ops.interpolate(sparse_feat_prop_s2, None, None, (h, w), "half_pixel", "bilinear")
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s2 = pixel_unshuffle(sparse_feat_prop_s2, self.stride)
                sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                # feature tokenization * 8
                # bs * c * h * w --> # bs * (c*8*8) * (h//4*w//4) -->  bs * c * (h*2) * (w*2)
                pad = int(0.5 * self.stride)
                ksize = int(2 * self.stride)
                feat_prop_pad2 = ops.Pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))(feat_prop)
                sparse_feat_prop_s3 = self.sparse_s3_unfold(feat_prop_pad2)
                sparse_feat_prop_s3 = sparse_feat_prop_s3.reshape(
                    n, ksize, ksize, -1, h // self.stride, w // self.stride
                )
                sparse_feat_prop_s3 = sparse_feat_prop_s3.transpose(0, 3, 4, 1, 5, 2)
                sparse_feat_prop_s3 = sparse_feat_prop_s3.reshape(n, -1, int(2 * h), int(2 * w))
                # bs * c * (h*2) * (w*2) -->  bs * c * h * w
                sparse_feat_prop_s3 = ops.avg_pool2d(sparse_feat_prop_s3, 2, 2)
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s3 = pixel_unshuffle(sparse_feat_prop_s3, self.stride)
                sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

        outputs_back = feat_buffers[::-1]

        # forward-time propagation and upsampling
        final_out = []
        feat_buffers = []
        sparse_feat_buffers_s1 = []
        sparse_feat_buffers_s2 = []
        sparse_feat_buffers_s3 = []
        index_feat_buffers_s1 = []

        feat_prop = ops.zeros_like(feat_prop)
        grid_y, grid_x = ops.meshgrid(
            (ms.numpy.arange(h // self.stride), ms.numpy.arange(w // self.stride)), indexing="ij"
        )
        location_update = ops.stack([grid_x, grid_y], axis=0).astype(lrs.dtype).broadcast_to((n, -1, -1, -1))
        for i in range(t):
            lr_curr = lrs[:, i, :, :, :]
            lr_curr_feat = outputs[:, i]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = self.flow_warp_bilinear(feat_prop, flow.transpose(0, 2, 3, 1))

                flow = ops.avg_pool2d(flow, self.stride, self.stride) / self.stride
                location_update = self.flow_warp_nearest(location_update, flow.transpose(0, 2, 3, 1))  # n , 2t , h , w

                sparse_feat_buffer_s1 = ops.stack(sparse_feat_buffers_s1, axis=1)
                sparse_feat_buffer_s2 = ops.stack(sparse_feat_buffers_s2, axis=1)
                sparse_feat_buffer_s3 = ops.stack(sparse_feat_buffers_s3, axis=1)
                index_feat_buffer_s1 = ops.stack(index_feat_buffers_s1, axis=1)
                feat_prop = self.LTAM(
                    lr_curr_feat,
                    index_feat_buffer_s1,
                    feat_prop,
                    sparse_feat_buffer_s1,
                    sparse_feat_buffer_s2,
                    sparse_feat_buffer_s3,
                    location_update,
                )

                if i in keyframe_idx_forward:
                    location_update_step = (
                        ops.stack([grid_x, grid_y], axis=0).astype(lrs.dtype).broadcast_to((n, -1, -1, -1))
                    )
                    location_update = ops.concat([location_update, location_update_step], axis=1)

            feat_prop = ops.concat([outputs[:, i], feat_prop], axis=1)
            feat_prop = self.resblocks(feat_prop)
            feat_buffers.append(feat_prop)

            if i in keyframe_idx_forward:
                # bs * c * h * w --> # bs * (c*4*4) * h//4 * w//4
                index_feat_prop_s1 = pixel_unshuffle(lr_curr_feat, self.stride)
                index_feat_buffers_s1.append(index_feat_prop_s1)

                # feature tokenization *4
                # bs * c * h * w --> # bs * (c*4*4) * h//4 * w//4
                sparse_feat_prop_s1 = pixel_unshuffle(feat_prop, self.stride)
                sparse_feat_buffers_s1.append(sparse_feat_prop_s1)

                # feature tokenization *6
                # bs * c * h * w --> # bs * (c*6*6) * (h//4*w//4) -->  bs * c * (h*1.5) * (w*1.5)
                pad = int(0.25 * self.stride)
                ksize = int(1.5 * self.stride)
                feat_prop_pad1 = ops.Pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))(feat_prop)
                sparse_feat_prop_s2 = self.sparse_s2_unfold(feat_prop_pad1)
                sparse_feat_prop_s2 = sparse_feat_prop_s2.reshape(
                    n, ksize, ksize, -1, h // self.stride, w // self.stride
                )
                sparse_feat_prop_s2 = sparse_feat_prop_s2.transpose(0, 3, 4, 1, 5, 2)
                sparse_feat_prop_s2 = sparse_feat_prop_s2.reshape(n, -1, int(1.5 * h), int(1.5 * w))
                # bs * c * (h*1.5) * (w*1.5) -->  bs * c * h * w
                sparse_feat_prop_s2 = ops.interpolate(sparse_feat_prop_s2, None, None, (h, w), "half_pixel", "bilinear")
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s2 = pixel_unshuffle(sparse_feat_prop_s2, self.stride)
                sparse_feat_buffers_s2.append(sparse_feat_prop_s2)

                # feature tokenization *8
                # bs * c * h * w --> # bs * (c*8*8) * (h//4*w//4) -->  bs * c * (h*2) * (w*2)
                pad = int(0.5 * self.stride)
                ksize = int(2 * self.stride)
                feat_prop_pad2 = ops.Pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))(feat_prop)
                sparse_feat_prop_s3 = self.sparse_s3_unfold(feat_prop_pad2)
                sparse_feat_prop_s3 = sparse_feat_prop_s3.reshape(
                    n, ksize, ksize, -1, h // self.stride, w // self.stride
                )
                sparse_feat_prop_s3 = sparse_feat_prop_s3.transpose(0, 3, 4, 1, 5, 2)
                sparse_feat_prop_s3 = sparse_feat_prop_s3.reshape(n, -1, int(2 * h), int(2 * w))
                # bs * c * (h*2) * (w*2) -->  bs * c * h * w
                sparse_feat_prop_s3 = ops.avg_pool2d(sparse_feat_prop_s3, 2, 2)
                # bs * c * h * w --> # bs * (c*4*4) * (h//4*w//4)
                sparse_feat_prop_s3 = pixel_unshuffle(sparse_feat_prop_s3, self.stride)
                sparse_feat_buffers_s3.append(sparse_feat_prop_s3)

            # upsampling given the backward and forward features
            out = ops.concat([outputs_back[i], lr_curr_feat, feat_prop], axis=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            base = ops.interpolate(lr_curr, None, (1.0, 1.0, 4.0, 4.0), None, "half_pixel", "bilinear")
            out += base

            final_out.append(out)

        final_out = ops.stack(final_out, axis=1)

        return final_out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """

        if isinstance(pretrained, str):
            load_param_into_net(self, pretrained, strict_load=strict, verbose=True)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')
        else:
            init_weights(self.fusion, init_type="he")
            init_weights(self.conv_hr, init_type="he")
            init_weights(self.conv_last, init_type="he")


class LTAM(nn.Cell):
    def __init__(self, stride=4):
        super().__init__()

        self.stride = stride

        self.fusion = nn.Conv2d(
            in_channels=3 * 64, out_channels=64, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        init_weights(self.fusion, init_type="he")

        self.fea_norm = ops.L2Normalize(axis=2)
        self.index_norm = ops.L2Normalize(axis=3)

        self.grid_sample = build_grid_sample(interpolation_mode="nearest", padding_mode="zeros", align_corners=True)

    def construct(
        self,
        curr_feat,
        index_feat_set_s1,
        anchor_feat,
        sparse_feat_set_s1,
        sparse_feat_set_s2,
        sparse_feat_set_s3,
        location_feat,
    ):
        """Compute the long-range trajectory-aware attention.

        Args:
            anchor_feat (tensor): Input feature with shape (n, c, h, w)
            sparse_feat_set_s1 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s2 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            sparse_feat_set_s3 (tensor): Input tokens with shape (n, t, c*4*4, h//4, w//4)
            location_feat (tensor): Input map with shape (n, 2*t, h//4, w//4)

        Return:
            fusion_feature (tensor): Output fusion feature with shape (n, c, h, w).
        """

        n, c, h, w = anchor_feat.shape
        n, c, h, w = int(n), int(c), int(h), int(w)
        t = sparse_feat_set_s1.shape[1]
        feat_len = int(c * self.stride * self.stride)
        feat_num = int((h // self.stride) * (w // self.stride))

        # grid_flow [0,h-1][0,w-1] -> [-1,1][-1,1]
        grid_flow = location_feat.view(n, t, 2, h // self.stride, w // self.stride).transpose(0, 1, 3, 4, 2)
        w_shifted = ops.maximum(ms.Tensor(float(w // self.stride) - 1.0), ms.Tensor(1.0))
        h_shifted = ops.maximum(ms.Tensor(float(h // self.stride) - 1.0), ms.Tensor(1.0))
        grid_flow_x = 2.0 * grid_flow[:, :, :, :, 0] / w_shifted - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, :, 1] / h_shifted - 1.0
        grid_flow = ops.stack((grid_flow_x, grid_flow_y), axis=4).astype(ops.dtype(curr_feat))

        output_s1 = self.grid_sample(
            sparse_feat_set_s1.view(-1, c * self.stride * self.stride, h // self.stride, w // self.stride),
            grid_flow.view(-1, h // self.stride, w // self.stride, 2),
        )  # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s2 = self.grid_sample(
            sparse_feat_set_s2.view(-1, c * self.stride * self.stride, h // self.stride, w // self.stride),
            grid_flow.view(-1, h // self.stride, w // self.stride, 2),
        )  # (nt) * (c*4*4) * (h//4) * (w//4)
        output_s3 = self.grid_sample(
            sparse_feat_set_s3.view(-1, c * self.stride * self.stride, h // self.stride, w // self.stride),
            grid_flow.view(-1, h // self.stride, w // self.stride, 2),
        )  # (nt) * (c*4*4) * (h//4) * (w//4)

        # n * c * h * w --> # n * (c*4*4) * (h//4*w//4)
        curr_feat = pixel_unshuffle(curr_feat, self.stride)
        curr_feat = curr_feat.view(n, -1, int(h // self.stride) * int(w // self.stride))
        # n * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * (c*4*4)
        curr_feat = curr_feat.transpose(0, 2, 1)
        curr_feat = self.fea_norm(curr_feat).expand_dims(3)  # n * (h//4*w//4) * (c*4*4) * 1

        # cross-scale attention * 4
        index_output_s1 = self.grid_sample(
            index_feat_set_s1.view(-1, c * self.stride * self.stride, h // self.stride, w // self.stride),
            grid_flow.view(-1, h // self.stride, w // self.stride, 2),
        )  # (nt) * (c*4*4) * (h//4) * (w//4)
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        index_output_s1 = index_output_s1.view(n, -1, feat_len, feat_num)
        # n * t * (c*4*4) * (h//4*w//4) --> n * (h//4*w//4) * t * (c*4*4)
        index_output_s1 = index_output_s1.transpose(0, 3, 1, 2)
        index_output_s1 = self.index_norm(index_output_s1)  # n * (h//4*w//4) * t * (c*4*4)
        # [ n * (h//4*w//4) * t * (c*4*4) ]  *  [ n * (h//4*w//4) * (c*4*4) * 1 ]  -->  n * (h//4*w//4) * t
        matrix_index = ops.BatchMatMul(False, False)(index_output_s1, curr_feat)  # n * (h//4*w//4) * t * 1
        matrix_index = matrix_index.view(n, feat_num, t)  # n * (h//4*w//4) * t
        corr_index, corr_soft = ops.ArgMaxWithValue(axis=2)(matrix_index)  # n * (h//4*w//4)
        # n * (h//4*w//4) --> n * 1 * (c*4*4) * (h//4*w//4)
        corr_index = corr_index.view(n, 1, 1, feat_num).broadcast_to((-1, -1, feat_len, -1))
        # n * (h//4*w//4) --> n * (c*4*4) * (h//4*w//4)
        corr_soft = corr_soft.expand_dims(1).broadcast_to((-1, feat_len, -1))
        # n * (c*4*4) * (h//4*w//4) --> n * c * h * w
        corr_soft = corr_soft.reshape(n, -1, h, w)

        # Aggr
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s1 = output_s1.view(n, -1, feat_len, feat_num)
        output_s1 = ops.gather_elements(output_s1, 1, corr_index)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s1 = output_s1.reshape(n, -1, h, w)

        # Aggr
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s2 = output_s2.view(n, -1, feat_len, feat_num)
        output_s2 = ops.gather_elements(output_s2, 1, corr_index)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s2 = output_s2.reshape(n, -1, h, w)

        # Aggr
        # nt * (c*4*4) * h//4 * w//4 --> n * t * (c*4*4) * (h//4*w//4)
        output_s3 = output_s3.view(n, -1, feat_len, feat_num)
        output_s3 = ops.gather_elements(output_s3, 1, corr_index)
        # n * 1 * (c*4*4) * (h//4*w//4) --> n * c * h * w
        output_s3 = output_s3.reshape(n, -1, h, w)

        out = ops.concat([output_s1, output_s2, output_s3], axis=1)
        out = self.fusion(out)
        out = out * corr_soft
        out += anchor_feat

        return out


class ResidualBlocksWithInputConv(nn.Cell):
    """Residual blocks with a convolution in front.

    Args:
        in_channels [int]: Number of input channels of the first conv.
        out_channels [int]: Number of channels of the residual blocks.
            Default: 64.
        num_blocks [int]: Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        input_conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, pad_mode="pad", padding=1, has_bias=True)
        init_weights(input_conv, init_type="he")

        layers = [
            input_conv,
            nn.LeakyReLU(alpha=0.1),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        ]

        self.main = nn.SequentialCell(*layers)

    def construct(self, x):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            x (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """

        out = self.main(x)

        return out


class SPyNet(nn.Cell):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained=None):
        super().__init__()

        self.basic_module = nn.CellList([SPyNetBasicModule() for _ in range(6)])
        self.avg_pool2d = ops.AvgPool(kernel_size=2, strides=2, pad_mode="valid")

        self.flow_warp = FlowWarp(padding_mode="border", interpolation="bilinear", align_corners=True)

        if isinstance(pretrained, str):
            load_param_into_net(self, pretrained, strict_load=True, ignore_list=["spynet.mean", "spynet.std"])
        elif pretrained is not None:
            raise TypeError("[pretrained] should be str or None, but got {type(pretrained)}.")

        self.mean = ms.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = ms.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref [Tensor]: Reference image with shape of (n, 3, h, w).
            supp [Tensor]: Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        n, _, h, w = ref.shape

        # normalize the input images
        supp = [(supp - self.mean) / self.std]
        ref = [(ref - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(self.avg_pool2d(ref[-1]))
            supp.append(self.avg_pool2d(supp[-1]))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ops.zeros((n, 2, h // 32, w // 32), ref[0].dtype)
        flow_up = flow
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = 2.0 * ops.interpolate(flow, None, (1.0, 1.0, 2.0, 2.0), None, "align_corners", "bilinear")

            # add the residual to the upsampled flow
            flow_warp_feat = self.flow_warp(supp[level], flow_up.transpose(0, 2, 3, 1))
            flow_feat = ops.concat([ref[level], flow_warp_feat, flow_up], 1)
            flow_feat = self.basic_module[level](flow_feat)
            flow = flow_up + flow_feat

        return flow

    def construct(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref [Tensor]: Reference image with shape of (n, 3, h, w).
            supp [Tensor]: Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsample to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)

        ref = ops.interpolate(ref, None, None, (h_up, w_up), "half_pixel", "bilinear")
        supp = ops.interpolate(supp, None, None, (h_up, w_up), "half_pixel", "bilinear")

        # compute flow, and resize back to the original resolution
        flow = self.compute_flow(ref, supp)
        flow = ops.interpolate(flow, None, None, (h, w), "half_pixel", "bilinear")

        # adjust the flow values
        scales = ms.Tensor([float(w) / float(w_up), float(h) / float(h_up)]).reshape(1, 2, 1, 1)
        flow = scales * flow

        return flow


class SPyNetBasicModule(nn.Cell):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.SequentialCell(
            nn.Conv2dBnAct(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                pad_mode="pad",
                padding=3,
                has_bias=True,
                has_bn=False,
                activation="relu",
            ),
            nn.Conv2dBnAct(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                pad_mode="pad",
                padding=3,
                has_bias=True,
                has_bn=False,
                activation="relu",
            ),
            nn.Conv2dBnAct(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                pad_mode="pad",
                padding=3,
                has_bias=True,
                has_bn=False,
                activation="relu",
            ),
            nn.Conv2dBnAct(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                pad_mode="pad",
                padding=3,
                has_bias=True,
                has_bn=False,
                activation="relu",
            ),
            nn.Conv2dBnAct(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                pad_mode="pad",
                padding=3,
                has_bias=True,
                has_bn=False,
                activation=None,
            ),
        )

    def construct(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """

        y = self.basic_module(x)

        return y


@ms.ops.constexpr
def _compute_grid(h, w):
    # create mesh grid
    grid_y, grid_x = ops.meshgrid((ms.numpy.arange(h), ms.numpy.arange(w)), indexing="ij")
    grid = ops.stack((grid_x, grid_y), axis=2)  # (w, h, 2)
    return grid


class FlowWarp(nn.Cell):
    def __init__(self, interpolation="bilinear", padding_mode="zeros", align_corners=True, fast_grid_sample=True):
        super().__init__()

        self.grid_sample = build_grid_sample(
            interpolation_mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners,
            fast_grid_sample=fast_grid_sample,
        )

    def construct(self, x, flow):
        h, w = x.shape[2:4]
        grid = ops.stop_gradient(_compute_grid(h, w))

        grid_flow = grid.astype(x.dtype).expand_dims(0) + flow

        # scale grid_flow to [-1, 1]
        scales = ms.Tensor([2.0 / (float(w) - 1.0), 2.0 / (float(h) - 1.0)])
        scales = scales.reshape(1, 1, 1, 2).astype(x.dtype)
        grid_flow = (scales * grid_flow - 1.0).astype(x.dtype)

        output = self.grid_sample(x, grid_flow)

        return output
