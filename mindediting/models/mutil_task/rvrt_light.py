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

import mindspore as ms
import mindspore.nn as nn

from mindediting.utils.init_weights import HeUniform, Uniform, _calculate_fan_in_and_fan_out, initializer
from mindediting.utils.utils import UnfoldAscendFP32, cast_module

from .rvrt import CellDict, Mlp, RSTBWithInputConv, Upsample, WindowAttention, nn_Transpose


class AttnPack(nn.Cell):
    """Attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        attention_window=[3, 3],
        attention_heads=12,
        clip_size=1,
        **kwargs,
    ):
        super(AttnPack, self).__init__()
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = attention_window[0]
        self.kernel_w = attention_window[1]
        self.attn_size = self.kernel_h * self.kernel_w
        self.attention_heads = attention_heads
        self.clip_size = clip_size
        self.stride = 1
        self.padding = self.kernel_h // 2
        self.dilation = 1

        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = int(self.in_channels * 2)
        self.proj_q = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                nn.Dense(self.in_channels, self.proj_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )
        self.proj_k = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                nn.Dense(self.in_channels, self.proj_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )
        self.proj_v = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                nn.Dense(self.in_channels, self.proj_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )
        self.proj = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                nn.Dense(self.proj_channels, self.in_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )
        self.mlp = nn.SequentialCell(
            [
                nn_Transpose("n d c h w -> n d h w c"),
                Mlp(self.in_channels, self.in_channels * 2, self.in_channels),
                nn_Transpose("n d h w c -> n d c h w"),
            ]
        )

        self.upscale = ms.nn.Unfold([1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding="same")

    def construct(self, q, k, v):
        b, t, _, h, w = q.shape
        q = self.proj_q(q).reshape(b * t, 1, self.proj_channels, h, w)
        kv = ms.ops.concat([self.proj_k(k), self.proj_v(v)], 2)

        v = self.deform_attn(
            q,
            kv,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding,
            self.dilation,
            self.attention_heads,
            self.clip_size,
        ).reshape(b, t, self.proj_channels, h, w)

        v = self.proj(v)
        v = v + self.mlp(v)

        return v

    def get_columns(self, q, kv, b, clip_size, height, width, attn_size, attn_head, attn_dim):
        columns = []

        TWO = 2

        for n in range(clip_size):
            cur_x = kv[b // clip_size, (n + b) % clip_size]
            cur_x = cur_x.reshape(1, -1, cur_x.shape[-2], cur_x.shape[-1])
            unfolded = self.upscale(cur_x)
            unfolded = unfolded.reshape(attn_size, TWO * attn_head, attn_dim, height * width)
            columns.append(unfolded)

        columns = ms.ops.concat(columns, axis=0)
        columns = columns.transpose(1, 3, 2, 0)
        columns = columns.reshape(TWO, attn_head, height * width, attn_dim, clip_size * attn_size)

        attns = ms.ops.matmul(q[b], columns[0])
        attns = ms.ops.softmax(attns, -1)  # (attn_head x (height*width) x 1 x (clip_size*attn_size))
        output = (
            ms.ops.matmul(attns, columns[1].swapaxes(2, 3)).swapaxes(1, 3).reshape(attn_head, attn_dim, height, width)
        )

        return output

    def deform_attn(self, q, kv, kernel_h, kernel_w, stride, padding, dilation, attn_head, clip_size):
        assert stride == 1 and padding == 1 and dilation == 1
        batch = q.shape[0]
        kv_channels = kv.shape[2]
        assert kv_channels % 2 == 0
        channels = kv_channels // 2
        height = kv.shape[3]
        width = kv.shape[4]
        area = height * width
        assert channels % attn_head == 0
        attn_dim = channels // attn_head
        attn_size = kernel_h * kernel_w
        attn_scale = attn_dim ** (-0.5)

        q = (q.reshape(batch, attn_head, attn_dim, area).swapaxes(2, 3) * attn_scale).reshape(
            batch, attn_head, area, 1, attn_dim
        )  # batch x attn_head x (height*width) x 1 x attn_dim

        output = []

        for b in range(batch):
            output.append(self.get_columns(q, kv, b, clip_size, height, width, attn_size, attn_head, attn_dim))

        output = ms.ops.stack(output)

        output = output.reshape(batch, channels, height, width)

        return output


class RVRT_LIGHT(nn.Cell):
    """Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT).
        A PyTorch impl of : `Recurrent Video Restoration Transformer with Guided Deformable Attention`  -
          https://arxiv.org/pdf/2205.00000

    Args:
        upscale (int): Upscaling factor. Set as 1 for video deblurring, etc. Default: 4.
        clip_size (int): Size of clip in recurrent restoration transformer.
        img_size (int | tuple(int)): Size of input video. Default: [2, 64, 64].
        window_size (int | tuple(int)): Window size. Default: (2,8,8).
        num_blocks (list[int]): Number of RSTB blocks in each stage.
        depths (list[int]): Depths of each RSTB.
        embed_dims (list[int]): Number of linear projection output channels.
        num_heads (list[int]): Number of attention head of each stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (obj): Normalization layer. Default: nn.LayerNorm.
        inputconv_groups (int): Group of the first convolution layer in RSTBWithInputConv. Default: [1,1,1,1,1,1]
        deformable_groups (int): Number of deformable groups in deformable attention. Default: 12.
        attention_heads (int): Number of attention heads in deformable attention. Default: 12.
        attention_window (list[int]): Attention window size in aeformable attention. Default: [3, 3].
        nonblind_denoising (bool): If True, conduct experiments on non-blind denoising. Default: False.
    """

    def __init__(
        self,
        upscale=4,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[144, 144, 144],
        num_heads=[6, 6, 6],
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        inputconv_groups=[1, 1, 1, 1, 1, 1],
        max_residue_magnitude=10,
        attention_heads=12,
        attention_window=[3, 3],
        nonblind_denoising=False,
        to_float16=False,
        relative_position_encoding=True,
        recurrent_feature_refinement_steps=4,
    ):
        super().__init__()
        self.upscale = upscale
        self.clip_size = clip_size
        self.nonblind_denoising = nonblind_denoising
        self.recurrent_feature_refinement_steps = recurrent_feature_refinement_steps

        # optical flow
        blocks_class = RSTBWithInputConv

        # shallow feature extraction
        if self.upscale == 4:
            # video sr
            self.feat_extract = blocks_class(
                in_channels=3,
                kernel_size=(1, 3, 3),
                groups=inputconv_groups[0],
                num_blocks=num_blocks[0],
                dim=embed_dims[0],
                input_resolution=img_size,
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=[1, window_size[1], window_size[2]],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                relative_position_encoding=relative_position_encoding,
            )
        else:
            # video deblurring/denoising
            self.feat_extract = nn.SequentialCell(
                [
                    nn_Transpose("n d c h w -> n c d h w"),
                    nn.Conv3d(
                        has_bias=True,
                        in_channels=4 if self.nonblind_denoising else 3,
                        out_channels=embed_dims[0],
                        kernel_size=(1, 3, 3),
                        stride=(1, 2, 2),
                        pad_mode="pad",
                        padding=(0, 0, 1, 1, 1, 1),
                    ),
                    nn.LeakyReLU(alpha=0.1),
                    nn.Conv3d(
                        has_bias=True,
                        in_channels=embed_dims[0],
                        out_channels=embed_dims[0],
                        kernel_size=(1, 3, 3),
                        stride=(1, 2, 2),
                        pad_mode="pad",
                        padding=(0, 0, 1, 1, 1, 1),
                    ),
                    nn.LeakyReLU(alpha=0.1),
                    nn_Transpose("n c d h w -> n d c h w"),
                    blocks_class(
                        in_channels=embed_dims[0],
                        kernel_size=(1, 3, 3),
                        groups=inputconv_groups[0],
                        num_blocks=num_blocks[0],
                        dim=embed_dims[0],
                        input_resolution=img_size,
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=[1, window_size[1], window_size[2]],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        norm_layer=norm_layer,
                        relative_position_encoding=relative_position_encoding,
                    ),
                ]
            )

        # recurrent feature refinement
        self.backbone = dict()
        self.deform_align = dict()
        self.modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(self.modules):
            self.deform_align[module] = AttnPack(
                embed_dims[1],
                embed_dims[1],
                attention_window=attention_window,
                attention_heads=attention_heads,
                clip_size=clip_size,
                max_residue_magnitude=max_residue_magnitude,
            )

            # feature propagation
            self.backbone[module] = blocks_class(
                in_channels=(2 + i) * embed_dims[0],
                kernel_size=(1, 3, 3),
                groups=inputconv_groups[i + 1],
                num_blocks=num_blocks[1],
                dim=embed_dims[1],
                input_resolution=[self.clip_size, img_size[1], img_size[2]],
                depth=depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                relative_position_encoding=relative_position_encoding,
            )

        # reconstruction
        self.reconstruction = blocks_class(
            in_channels=5 * embed_dims[0],
            kernel_size=(1, 3, 3),
            groups=inputconv_groups[5],
            num_blocks=num_blocks[2],
            dim=embed_dims[2],
            input_resolution=[1, img_size[1], img_size[2]],
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=[1, window_size[1], window_size[2]],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
            relative_position_encoding=relative_position_encoding,
        )
        self.conv_before_upsampler = nn.SequentialCell(
            [
                nn.Conv3d(
                    has_bias=True,
                    in_channels=embed_dims[-1],
                    out_channels=64,
                    kernel_size=(1, 1, 1),
                    pad_mode="pad",
                    padding=(0, 0, 0, 0, 0, 0),
                ),
                nn.LeakyReLU(alpha=0.1),
            ]
        )
        self.backbone = CellDict(self.backbone)
        self.deform_align = CellDict(self.deform_align)
        self.backbone_dict = {k: getattr(self.backbone, k) for k in self.modules}
        self.deform_align_dict = {k: getattr(self.deform_align, k) for k in self.modules}
        self.upsampler = Upsample(4, 64)
        self.conv_last = nn.Conv3d(
            has_bias=True,
            in_channels=64,
            out_channels=3,
            kernel_size=(1, 3, 3),
            pad_mode="pad",
            padding=(0, 0, 1, 1, 1, 1),
        )

        self._init_weights()

        if to_float16:
            cast_module("float16", self)

    def _init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(
                cell,
                (
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.Dense,
                ),
            ):
                if hasattr(cell, "weight") and cell.weight is not None:
                    kaiming_init = HeUniform(negative_slope=math.sqrt(5.0), mode="fan_in", nonlinearity="leaky_relu")
                    cell.weight = initializer(kaiming_init, cell.weight.shape, cell.weight.dtype)
                if hasattr(cell, "bias") and cell.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                    bound = 1.0 / math.sqrt(fan_in)
                    cell.bias = initializer(Uniform(bound), cell.bias.shape, cell.bias.dtype)

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """

        feats["shallow"] = ms.ops.concat(feats["shallow"], 1)
        feats["backward_1"] = ms.ops.concat(feats["backward_1"], 1) if "backward_1" in feats else feats["shallow"]
        feats["forward_1"] = ms.ops.concat(feats["forward_1"], 1) if "forward_1" in feats else feats["backward_1"]
        feats["backward_2"] = ms.ops.concat(feats["backward_2"], 1) if "backward_2" in feats else feats["forward_1"]
        feats["forward_2"] = ms.ops.concat(feats["forward_2"], 1) if "forward_2" in feats else feats["backward_2"]

        hr = ms.ops.concat([feats[k] for k in feats], axis=2)
        hr = self.reconstruction(hr)
        hr = self.conv_last(self.upsampler(self.conv_before_upsampler(hr.swapaxes(1, 2)))).swapaxes(1, 2)

        dim0, dim1, dim2, dim3, dim4 = lqs.shape
        lqs = lqs.reshape(-1, dim2, dim3, dim4)
        resized = ms.ops.interpolate(
            lqs, sizes=(dim3 * 4, dim4 * 4), coordinate_transformation_mode="half_pixel", mode="bilinear"
        )
        resized = resized.reshape(dim0, dim1, dim2, resized.shape[-2], resized.shape[-1])

        return hr + resized

    def backward1(self, feats, t):
        direction = "backward"
        module_name = f"{direction}_1"
        feats[module_name] = []
        clip_idx_backward = range((t + 1) // self.clip_size - 1, -1, -1)

        feat_prop = ms.ops.Zeros()(feats["shallow"][0].shape, feats["shallow"][0].dtype)

        last_key = list(feats)[-2]

        idx_c = clip_idx_backward[0]
        feat = [feats[k][idx_c] for k in feats if k not in [module_name]]
        feat = [ms.ops.ReverseV2([1])(k) for k in feat] + [feat_prop]
        feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
        feats[module_name].append(feat_prop)

        for i in range(1, len(clip_idx_backward)):
            idx_c = clip_idx_backward[i]

            feat_q = feats[last_key][idx_c]
            feat_k = feats[last_key][clip_idx_backward[i - 1]]
            feat_q = ms.ops.ReverseV2([1])(feat_q)
            feat_k = ms.ops.ReverseV2([1])(feat_k)

            feat_prop = self.deform_align_dict[module_name](feat_q, feat_k, feat_prop)

            feat = [feats[k][idx_c] for k in feats if k not in [module_name]]
            feat = [ms.ops.ReverseV2([1])(k) for k in feat] + [feat_prop]
            feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
            feats[module_name].append(feat_prop)

        fi = len(feats[module_name]) - 1
        feats[module_name] = [
            ms.ops.ReverseV2([1])(feats[module_name][fi - i]) for i, _ in enumerate(feats[module_name])
        ]

        return feats

    def forward1(self, feats, t):
        direction = "forward"
        module_name = f"{direction}_1"
        feats[module_name] = []
        clip_idx_forward = range(0, (t + 1) // self.clip_size)

        feat_prop = ms.ops.Zeros()(feats["shallow"][0].shape, feats["shallow"][0].dtype)

        last_key = list(feats)[-2]

        idx_c = clip_idx_forward[0]
        feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
        feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
        feats[module_name].append(feat_prop)

        for i in range(1, len(clip_idx_forward)):
            idx_c = clip_idx_forward[i]

            feat_q = feats[last_key][idx_c]
            feat_k = feats[last_key][clip_idx_forward[i - 1]]

            feat_prop = self.deform_align_dict[module_name](feat_q, feat_k, feat_prop)

            feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
            feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
            feats[module_name].append(feat_prop)

        return feats

    def backward2(self, feats, t):
        direction = "backward"
        module_name = f"{direction}_2"
        feats[module_name] = []
        clip_idx_backward = range((t + 1) // self.clip_size - 1, -1, -1)

        feat_prop = ms.ops.Zeros()(feats["shallow"][0].shape, feats["shallow"][0].dtype)

        last_key = list(feats)[-2]

        idx_c = clip_idx_backward[0]
        feat = [feats[k][idx_c] for k in feats if k not in [module_name]]
        feat = [ms.ops.ReverseV2([1])(k) for k in feat] + [feat_prop]
        feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
        feats[module_name].append(feat_prop)

        for i in range(1, len(clip_idx_backward)):
            idx_c = clip_idx_backward[i]
            feat_q = feats[last_key][idx_c]
            feat_k = feats[last_key][clip_idx_backward[i - 1]]
            feat_q = ms.ops.ReverseV2([1])(feat_q)
            feat_k = ms.ops.ReverseV2([1])(feat_k)

            feat_prop = self.deform_align_dict[module_name](feat_q, feat_k, feat_prop)

            feat = [feats[k][idx_c] for k in feats if k not in [module_name]]
            feat = [ms.ops.ReverseV2([1])(k) for k in feat] + [feat_prop]
            feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
            feats[module_name].append(feat_prop)

        fi = len(feats[module_name]) - 1
        feats[module_name] = [
            ms.ops.ReverseV2([1])(feats[module_name][fi - i]) for i, _ in enumerate(feats[module_name])
        ]

        return feats

    def forward2(self, feats, t):
        direction = "forward"
        module_name = f"{direction}_{2}"
        feats[module_name] = []
        clip_idx_forward = range(0, (t + 1) // self.clip_size)

        feat_prop = ms.ops.Zeros()(feats["shallow"][0].shape, feats["shallow"][0].dtype)

        last_key = list(feats)[-2]

        idx_c = clip_idx_forward[0]
        feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
        feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
        feats[module_name].append(feat_prop)

        for i in range(1, len(clip_idx_forward)):
            idx_c = clip_idx_forward[i]
            feat_q = feats[last_key][idx_c]
            feat_k = feats[last_key][clip_idx_forward[i - 1]]

            feat_prop = self.deform_align_dict[module_name](feat_q, feat_k, feat_prop)

            feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]
            feat_prop = feat_prop + self.backbone_dict[module_name](ms.ops.concat(feat, axis=2))
            feats[module_name].append(feat_prop)

        return feats

    def construct(self, lqs):
        """Forward function for RVRT.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        t = lqs.shape[1]

        if self.upscale == 4:
            lqs_downsample = lqs.copy()
        else:
            raise NotImplementedError()
            # lqs_downsample = F.interpolate(lqs[:, :, :3, :, :].reshape(-1, 3, h, w), scale_factor=0.25, mode='bicubic')\
            #     .reshape(n, t, 3, h // 4, w // 4)

        # shallow feature extractions
        feats = {}
        feats["shallow"] = list(ms.ops.split(self.feat_extract(lqs), output_num=t // self.clip_size, axis=1))

        # recurrent feature refinement
        if self.recurrent_feature_refinement_steps >= 1:
            feats = self.backward1(feats, t)
        if self.recurrent_feature_refinement_steps >= 2:
            feats = self.forward1(feats, t)
        if self.recurrent_feature_refinement_steps >= 3:
            feats = self.backward2(feats, t)
        if self.recurrent_feature_refinement_steps >= 4:
            feats = self.forward2(feats, t)
        if self.recurrent_feature_refinement_steps >= 5:
            raise RuntimeError()

        # reconstruction
        return self.upsample(lqs_downsample, feats)

    def relative_position_bias_to_table(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, WindowAttention):
                cell.relative_position_bias_to_table()

    def relative_position_table_to_bias(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, WindowAttention):
                cell.relative_position_table_to_bias()
