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
import mindspore.ops as ops
import numpy as np
from mindspore.common.initializer import initializer

from mindediting.models.common.tunable_conv import TunableConv2d, TunableDense, TunableParameter, TunableSequentialCell
from mindediting.utils.utils import is_ascend


class DropPath(nn.Cell):
    """from timm.models.layers import DropPath
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def extend_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"

    def construct(self, x, *args, **kwargs):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = ops.bernoulli(x=ms.Tensor(shape=x.shape, dtype=x.dtype), p=keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor


class GELU(nn.Cell):
    """
    this implementation replicates exactly torch.nn.GELU
    mindspore.nn.GELU uses a different implementation which gives slightly different results
    """

    def __init__(self):
        super(GELU, self).__init__()

        self.erf = ops.Erf()
        self.sqrt2 = math.sqrt(2.0)

    def construct(self, x):
        cdf = 0.5 * (1.0 + self.erf(x / self.sqrt2))
        return x * cdf


class LeakyReLU(nn.LeakyReLU):
    """
    wrapper for mindspore.nn.LeakyReLU which use the same value of alpha of torch.nn.LeakyReLU
    """

    def __init__(self, alpha=0.01):
        super().__init__(alpha=alpha)

    def extend_repr(self):
        return f"alpha={self.alpha}"

    def construct(self, x, *args, **kwargs):
        return super().construct(x)


class LayerNorm(nn.LayerNorm):
    """
    wrapper for mindspore.nn.LayerNorm to use the same default
    initialization of epsilon of torch.nn.LayerNorm
    """

    def __init__(self, normalized_shape, begin_norm_axis=-1, begin_params_axis=-1):
        if type(normalized_shape) == int:
            normalized_shape = [normalized_shape]
        super(LayerNorm, self).__init__(
            normalized_shape=normalized_shape,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=begin_params_axis,
            epsilon=1e-5,
        )


class Dropout(nn.Dropout):
    """
    wrapper for mindspore.nn.Dropout which use the same interface as torch.nn.Dropout
    """

    def __init__(self, drop_prob):
        super().__init__(keep_prob=1 - drop_prob)


class Identity(nn.Cell):
    def __init__(self):
        super(Identity, self).__init__()

    def construct(self, x):
        return x


class PixelShuffle(nn.Cell):
    """
    mindspore.ops.DepthToSpace is not consistent with torch.nn.PixelShuffle
    """

    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.scale = scale

    def extend_repr(self):
        return f"scale={self.scale}"

    def construct(self, x, *args, **kwargs):
        s = self.scale
        b, c, h, w = x.shape

        y = x.reshape(b, c // s**2, s, s, h, w)
        y = ops.transpose(y, (0, 1, 4, 2, 5, 3))
        y = y.reshape(b, c // s**2, h * s, w * s)
        return y


class UpsampleNearest(nn.Cell):
    def __init__(self, scale_factor):
        super().__init__()
        assert scale_factor > 1 and type(scale_factor) == int
        self.scale_factor = scale_factor

    def extend_repr(self):
        return f"scale_factor={self.scale_factor}"

    def construct(self, x, *args, **kwargs):
        h, w = x.shape[2:]
        s = self.scale_factor
        resize = ops.ResizeNearestNeighbor((s * h, s * w))
        return resize(x)


# Equivalent to numpy.linspace, to avoid adding numpy dependency
def linspace(start, stop, num=50, endpoint=True):
    num = int(num)
    start = start * 1.0
    stop = stop * 1.0

    if num == 1:
        yield stop
        return
    if endpoint:
        step = (stop - start) / (num - 1)
    else:
        step = (stop - start) / num

    for i in range(num):
        yield start + step * i


class TunableMlp(nn.Cell):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.0, num_params=None, mode=None
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TunableDense(in_features, hidden_features, num_params=num_params, mode=mode)
        self.act = act_layer()
        self.fc2 = TunableDense(hidden_features, out_features, num_params=num_params, mode=mode)
        self.drop = Dropout(drop) if drop > 0.0 else Identity()

    def construct(self, x, px):
        x = self.fc1(x, px)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, px)
        x = self.drop(x)
        return x


@ms.ms_function
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = ops.transpose(x, (0, 1, 3, 2, 4, 5)).view(-1, window_size, window_size, C)
    return windows


@ms.ms_function
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5)).view(B, H, W, -1)
    return x


class TunableWindowAttention(nn.Cell):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim [int]: Number of input channels.
        window_size [tuple[int]]: The height and width of the window.
        num_heads [int]: Number of attention heads.
        qkv_bias [bool, optional]:  If True, add a learnable bias to query, key, value. Default: True
        qk_scale [float | None, optional]: Override default qk scale of head_dim ** -0.5 if set
        attn_drop [float, optional]: Dropout ratio of attention weight. Default: 0.0
        proj_drop [float, optional]: Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_params=None,
        mode=None,
    ):

        super().__init__()
        self.is_ascend = is_ascend()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.window_size = window_size  # Wh, Ww
        # define a parameter table of relative position bias
        self.relative_position_bias_table = TunableParameter(
            ms.Tensor(initializer("zeros", ((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))),
            num_params=num_params,
            mode=mode,
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        stack = ops.Stack()
        flatten = ops.Flatten()
        mesh_grid = ops.Meshgrid(indexing="ij")
        expand_dims = ops.ExpandDims()
        self.coords_h = ms.numpy.arange(self.window_size[0])
        self.coords_w = ms.numpy.arange(self.window_size[1])
        self.coords = stack(mesh_grid((self.coords_h, self.coords_w)))  # 2, Wh, Ww
        self.coords_flatten = flatten(self.coords)  # 2, Wh*Ww
        relative_coords = expand_dims(self.coords_flatten, 2) - expand_dims(self.coords_flatten, 1)  # 2, Wh*Ww, Wh*Ww
        relative_coords = ops.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        if self.is_ascend:
            relative_coords = ms.Tensor(relative_coords, ms.float16)
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        if self.is_ascend:
            relative_position_index = ms.Tensor(relative_position_index, ms.int32)
        self.relative_position_index = ms.Parameter(relative_position_index, requires_grad=False)

        self.qkv = TunableDense(dim, dim * 3, has_bias=qkv_bias, num_params=num_params, mode=mode)
        self.attn_drop = Dropout(attn_drop) if attn_drop > 0.0 else Identity()
        self.proj = TunableDense(dim, dim, num_params=num_params, mode=mode)

        self.proj_drop = Dropout(proj_drop) if proj_drop > 0.0 else Identity()

        self.softmax = nn.Softmax(axis=-1)

    def extend_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def construct(self, x, px, mask=None, batch_size=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        reshape = ops.Reshape()
        expand_dims = ops.ExpandDims()

        B_, N, C = x.shape
        qkv = self.qkv(x.view(batch_size, -1, N, C), px)
        qkv = reshape(qkv, (B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = ops.bmm(q, ops.transpose(k, (0, 1, 3, 2)))

        tuned_relative_position_bias_table = self.relative_position_bias_table(px)
        relative_position_bias = tuned_relative_position_bias_table[:, self.relative_position_index.view(-1)].view(
            batch_size, self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # B, Wh*Ww,Wh*Ww,nH
        relative_position_bias = ops.transpose(relative_position_bias, (0, 3, 1, 2))  # batch_size, nH, Wh*Ww, Wh*Ww
        attn = attn.view(batch_size, -1, self.num_heads, N, N) + expand_dims(
            relative_position_bias, 1
        )  # (batch_size, 1, self.num_heads, N, N)
        attn = reshape(attn, (B_, self.num_heads, N, N))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + expand_dims(expand_dims(mask, 1), 0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = reshape(ops.transpose(ops.bmm(attn, v), (0, 2, 1, 3)), (B_, N, C))
        x = reshape(self.proj(x.view(batch_size, -1, N, C), px), (B_, N, -1))
        x = self.proj_drop(x)
        return x


class TunableSwinTransformerBlock(nn.Cell):
    r"""Swin Transformer Block.

    Args:
        dim [int]: Number of input channels.
        num_heads [int]: Number of attention heads.
        window_size [int]: Window size.
        shift_size [int]: Shift size for SW-MSA.
        mlp_ratio [float]: Ratio of mlp hidden dim to embedding dim.
        qkv_bias [bool, optional]: If True, add a learnable bias to query, key, value. Default: True
        qk_scale [float | None, optional]: Override default qk scale of head_dim ** -0.5 if set.
        drop [float, optional]: Dropout rate. Default: 0.0
        attn_drop [float, optional]: Attention dropout rate. Default: 0.0
        drop_path [float, optional]: Stochastic depth rate. Default: 0.0
        act_layer [nn.Module, optional]: Activation layer. Default: nn.GELU
        norm_layer [nn.Module, optional]: Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=GELU,
        norm_layer=LayerNorm,
        num_params=None,
        mode=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim) if norm_layer is not None else Identity()
        self.attn = TunableWindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_params=num_params,
            mode=mode,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim) if norm_layer is not None else Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TunableMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            num_params=num_params,
            mode=mode,
        )

    def extend_repr(self):
        return (
            f"num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = ms.Tensor(
            initializer("zeros", (1, H, W, 1))
        )  # ms.Tensor(np.zeros(shape=(1, H, W, 1)), ms.float32)  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        expand_dims = ops.ExpandDims()
        masked_fill = ops.MaskedFill()
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = expand_dims(mask_windows, 1) - expand_dims(mask_windows, 2)
        attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
        attn_mask = masked_fill(attn_mask, attn_mask == 0, float(0.0))

        return attn_mask

    def construct(self, x, px, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = ms.numpy.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if min(x_size) <= self.window_size:
            attn_mask = None
        else:
            attn_mask = self.calculate_mask(x_size)
        attn_windows = self.attn(x_windows, px, mask=attn_mask, batch_size=B)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = ms.numpy.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), px))

        return x


class TunableBasicLayer(nn.Cell):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim [int]: Number of input channels.
        depth [int]: Number of blocks.
        num_heads [int]: Number of attention heads.
        window_size [int]: Local window size.
        mlp_ratio [float]: Ratio of mlp hidden dim to embedding dim.
        qkv_bias [bool, optional]: If True, add a learnable bias to query, key, value. Default: True
        qk_scale [float | None, optional]: Override default qk scale of head_dim ** -0.5 if set.
        drop [float, optional]: Dropout rate. Default: 0.0
        attn_drop [float, optional]: Attention dropout rate. Default: 0.0
        drop_path [float | tuple[float], optional]: Stochastic depth rate. Default: 0.0
        norm_layer [nn.Module, optional]: Normalization layer. Default: nn.LayerNorm
        use_checkpoint [bool]: Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNorm,
        use_checkpoint=False,
        num_params=None,
        mode=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        assert not use_checkpoint, "Not Implemented"

        # build blocks
        blocks = [
            TunableSwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                num_params=num_params,
                mode=mode,
            )
            for i in range(depth)
        ]
        self.blocks = nn.CellList(blocks)

    def extend_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def construct(self, x, px, x_size):
        for i in range(self.depth):
            x = self.blocks[i](x, px, x_size)
        return x


class PatchEmbed(nn.Cell):
    r"""Image to Patch Embedding

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.norm = norm_layer(embed_dim) if norm_layer is not None else Identity()

    def extend_repr(self) -> str:
        return f"embed_dim={self.embed_dim}"

    def construct(self, x):
        B, C, H, W = x.shape
        assert C == self.embed_dim
        x = ops.transpose(x.view(B, C, H * W), (0, 2, 1))  # B Ph*Pw C
        x = self.norm(x)
        return x


class PatchUnEmbed(nn.Cell):
    r"""Image to Patch Unembedding

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def extend_repr(self) -> str:
        return f"embed_dim={self.embed_dim}"

    def construct(self, x, x_size):
        B, HW, C = x.shape
        assert C == self.embed_dim
        x = ops.transpose(x, (0, 2, 1)).view(B, C, x_size[0], x_size[1])  # B C Ph Pw
        return x


class TunableRSTB(nn.Cell):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim [int]: Number of input channels.
        depth [int]: Number of blocks.
        num_heads [int]: Number of attention heads.
        window_size [int]: Local window size.
        mlp_ratio [float]: Ratio of mlp hidden dim to embedding dim.
        qkv_bias [bool, optional]: If True, add a learnable bias to query, key, value. Default: True
        qk_scale [float | None, optional]: Override default qk scale of head_dim ** -0.5 if set.
        drop [float, optional]: Dropout rate. Default: 0.0
        attn_drop [float, optional]: Attention dropout rate. Default: 0.0
        drop_path [float | tuple[float], optional]: Stochastic depth rate. Default: 0.0
        norm_layer [nn.Module, optional]: Normalization layer. Default: nn.LayerNorm
        use_checkpoint [bool]: Whether to use checkpointing to save memory. Default: False.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNorm,
        use_checkpoint=False,
        resi_connection="1conv",
        num_params=None,
        mode=None,
    ):
        super(TunableRSTB, self).__init__()

        self.dim = dim

        self.residual_group = TunableBasicLayer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            num_params=num_params,
            mode=mode,
        )

        if resi_connection == "1conv":
            self.conv = TunableConv2d(dim, dim, 3, has_bias=True, num_params=num_params, mode=mode)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = TunableSequentialCell(
                [
                    TunableConv2d(dim, dim // 4, 3, has_bias=True, num_params=num_params, mode=mode),
                    LeakyReLU(alpha=0.2),
                    TunableConv2d(dim // 4, dim // 4, 1, has_bias=True, num_params=num_params, mode=mode),
                    LeakyReLU(alpha=0.2),
                    TunableConv2d(dim // 4, dim, 3, has_bias=True, num_params=num_params, mode=mode),
                ]
            )

        self.patch_embed = PatchEmbed(in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(in_chans=0, embed_dim=dim, norm_layer=None)

    def construct(self, x, px, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, px, x_size), x_size), px)) + x


class TunableUpsample(TunableSequentialCell):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_params=None, mode=None):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(TunableConv2d(num_feat, 4 * num_feat, 3, has_bias=True, num_params=num_params, mode=mode))
                m.append(PixelShuffle(2))
        elif scale == 3:
            m.append(TunableConv2d(num_feat, 9 * num_feat, 3, has_bias=True, num_params=num_params, mode=mode))
            m.append(PixelShuffle(3))
        else:
            raise ValueError(f"scale {scale} is not supported. " "Supported scales: 2^n and 3.")
        super(TunableUpsample, self).__init__(m)


class TunableUpsampleOneStep(TunableSequentialCell):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale [int]: Scale factor. Supported scales: 2^n and 3.
        num_feat [int]: Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, num_params=None, mode=None):
        self.num_feat = num_feat
        m = []
        m.append(TunableConv2d(num_feat, (scale**2) * num_out_ch, 3, has_bias=True, num_params=num_params, mode=mode))
        m.append(PixelShuffle(scale))
        super(TunableUpsampleOneStep, self).__init__(m)


class TunableSwinIR(nn.Cell):
    r"""SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        in_chans [int]: Number of input image channels. Default: 3
        embed_dim [int]: Patch embedding dimension. Default: 96
        depths [tuple(int)]: Depth of each Swin Transformer layer.
        num_heads [tuple(int)]: Number of attention heads in different layers.
        window_size [int]: Window size. Default: 7
        mlp_ratio [float]: Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias [bool]: If True, add a learnable bias to query, key, value. Default: True
        qk_scale [float]: Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate [float]: Dropout rate. Default: 0
        attn_drop_rate [float]: Attention dropout rate. Default: 0
        drop_path_rate [float]: Stochastic depth rate. Default: 0.1
        norm_layer [nn.Module]: Normalization layer. Default: nn.LayerNorm.
        patch_norm [bool]: If True, add normalization after patch embedding. Default: True
        use_checkpoint [bool]: Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(
        self,
        img_channels=3,
        embed_dim=96,
        num_feat=64,
        depths=None,
        num_heads=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=LayerNorm,
        patch_norm=True,
        upscale=1,
        upsampler="",
        resi_connection="1conv",
        num_params=1,
        mode="mlp",
    ):
        super(TunableSwinIR, self).__init__()
        if depths is None:
            depths = [6, 6, 6, 6]
        if num_heads is None:
            num_heads = [6, 6, 6, 6]

        num_in_ch = img_channels
        num_out_ch = img_channels

        use_checkpoint = False

        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = ms.Tensor(rgb_mean, ms.float32).view(1, 3, 1, 1)
        else:
            self.mean = ms.Tensor(initializer("zeros", (1, 1, 1, 1)))

        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = TunableConv2d(num_in_ch, embed_dim, 3, has_bias=True, num_params=num_params, mode=mode)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )

        self.pos_drop = Dropout(drop_rate) if drop_rate > 0.0 else Identity()

        # stochastic depth
        dpr = [x for x in linspace(0.0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        layers = []
        for i_layer in range(self.num_layers):
            layer = TunableRSTB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection,
                num_params=num_params,
                mode=mode,
            )
            layers.append(layer)
        self.layers = nn.SequentialCell(layers)
        self.norm = norm_layer(self.num_features) if norm_layer is not None else Identity()

        # build the last conv layer in deep feature extraction
        if resi_connection == "1conv":
            self.conv_after_body = TunableConv2d(
                embed_dim, embed_dim, 3, has_bias=True, num_params=num_params, mode=mode
            )
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv_after_body = TunableSequentialCell(
                [
                    TunableConv2d(embed_dim, embed_dim // 4, 3, has_bias=True, num_params=num_params, mode=mode),
                    LeakyReLU(alpha=0.2, inplace=True),
                    TunableConv2d(embed_dim // 4, embed_dim // 4, 1, has_bias=True, num_params=num_params, mode=mode),
                    LeakyReLU(alpha=0.2, inplace=True),
                    TunableConv2d(embed_dim // 4, embed_dim, 3, has_bias=True, num_params=num_params, mode=mode),
                ]
            )

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = TunableSequentialCell(
                [
                    TunableConv2d(embed_dim, num_feat, 3, has_bias=True, num_params=num_params, mode=mode),
                    LeakyReLU(alpha=0.01),
                ]
            )
            self.upsample = TunableUpsample(self.upscale, num_feat, num_params=num_params, mode=mode)
            self.conv_last = TunableConv2d(num_feat, num_out_ch, 3, has_bias=True, num_params=num_params, mode=mode)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = TunableUpsampleOneStep(
                self.upscale, embed_dim, num_out_ch, num_params=num_params, mode=mode
            )
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            self.conv_before_upsample = TunableSequentialCell(
                [
                    TunableConv2d(embed_dim, num_feat, 3, has_bias=True, num_params=num_params, mode=mode),
                    LeakyReLU(alpha=0.01),
                ]
            )
            self.conv_up1 = TunableConv2d(num_feat, num_feat, 3, has_bias=True, num_params=num_params, mode=mode)
            if self.upscale == 4:
                self.conv_up2 = TunableConv2d(num_feat, num_feat, 3, has_bias=True, num_params=num_params, mode=mode)
            self.conv_hr = TunableConv2d(num_feat, num_feat, 3, has_bias=True, num_params=num_params, mode=mode)
            self.conv_last = TunableConv2d(num_feat, num_out_ch, 3, has_bias=True, num_params=num_params, mode=mode)
            self.lrelu = LeakyReLU(alpha=0.2)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = TunableConv2d(embed_dim, num_out_ch, 3, has_bias=True, num_params=num_params, mode=mode)

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        pad = nn.Pad(paddings=((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w)), mode="REFLECT")
        x = pad(x)
        return x

    def forward_features(self, x, px):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers.cell_list:
            x = layer(x, px, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def construct(self, x, px):

        h, w = x.shape[2:]
        x = self.check_image_size(x)

        x = x - self.mean

        x_first = 0.0
        fwd_fea = 0.0
        fwd_fea_ = 0.0
        res = 0.0

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x, px)
            x = self.conv_after_body(self.forward_features(x, px), px) + x
            x = self.conv_before_upsample(x, px)
            x = self.conv_last(self.upsample(x, px), px)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x, px)
            x = self.conv_after_body(self.forward_features(x, px), px) + x
            x = self.upsample(x, px)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            upsample_nearest = UpsampleNearest(2)
            x = self.conv_first(x, px)
            x = self.conv_after_body(self.forward_features(x, px), px) + x
            x = self.conv_before_upsample(x, px)
            x = self.lrelu(self.conv_up1(upsample_nearest(x), px))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(upsample_nearest(x), px))
            x = self.conv_last(self.lrelu(self.conv_hr(x, px)), px)
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x, px)
            res = self.conv_after_body(self.forward_features(x_first, px), px) + x_first
            x = x + self.conv_last(res, px)

        x = x + self.mean

        return x[:, :, : h * self.upscale, : w * self.upscale]


class TunableSwinIR_post(nn.Cell):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        px = np.expand_dims(params, axis=0).astype(np.float32)
        self.px = ms.Tensor(px)

    def construct(self, x):
        return self.model(x, self.px)
