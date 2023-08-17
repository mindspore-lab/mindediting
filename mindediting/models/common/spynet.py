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

import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import numpy as np
from mindspore.ops import operations as ops

from .grid_sample import GridSample
from .interpolation import interpolate_through_grid_sample


class ConvModule(nn.Cell):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, with_norm=None, bias="auto", activation=True
    ):
        super(ConvModule, self).__init__()
        self.with_activation = activation
        if bias == "auto":
            bias = not with_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            pad_mode="pad",
            has_bias=bias,
            weight_init="normal",
        )
        if activation:
            self.activate = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        if self.with_activation:
            x = self.activate(x)
        return x


class SPyNetBasicModule(nn.Cell):
    def __init__(self):
        super(SPyNetBasicModule, self).__init__()
        self.basic_module = nn.SequentialCell(
            ConvModule(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, bias=True),
            ConvModule(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, bias=True),
            ConvModule(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, bias=True),
            ConvModule(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, bias=True),
            ConvModule(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3, bias=True, activation=False),
        )

    def construct(self, x):
        return self.basic_module(x)


class FlowWarp(nn.Cell):
    def __init__(
        self,
        interpolation="bilinear",
        padding_mode="border",
        align_corners=True,
        precompute_grid=True,
        base_resolution=([64, 64], [64, 128]),
        levels=6,
        eliminate_gradient_for_gather=False,
    ):
        super(FlowWarp, self).__init__()
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.stack_2 = ops.Stack(axis=2)
        self.stack_3 = ops.Stack(axis=3)
        self.meshgrid = ops.Meshgrid(indexing="xy")
        self.max = ops.Maximum()
        self.grid_sample = GridSample(eliminate_gradient_for_gather=eliminate_gradient_for_gather)

        self.precompute_grid = precompute_grid
        if precompute_grid:
            base_resolution = [v for v in base_resolution]
            self.grids = [[], []]  # Grids for train and test step
            for _ in range(levels):
                for i in range(2):
                    h, w = base_resolution[i]
                    grid_x, grid_y = self.meshgrid(
                        (mnp.arange(0, int(w), dtype=ms.float32), mnp.arange(0, int(h), dtype=ms.float32))
                    )
                    grid = self.stack_2((grid_x, grid_y))
                    self.grids[i].append(grid)
                    base_resolution[i][0] /= 2
                    base_resolution[i][1] /= 2

    def construct(self, x, flow, level=None):
        dtype = x.dtype
        if x.shape[-2:] != flow.shape[1:3]:
            raise ValueError(
                f"The spatial sizes of input ({x.shape[-2:]}) and " f"flow ({flow.shape[1:3]}) are not the same."
            )
        _, _, h, w = x.shape
        # create mesh grid
        if self.precompute_grid and level is not None:
            grid = ms.ops.cast(self.grids[0][-1 - level] if self.training else self.grids[1][-1 - level], dtype)
        else:
            grid_x, grid_y = self.meshgrid((mnp.arange(0, w, dtype=dtype), mnp.arange(0, h, dtype=dtype)))
            grid = self.stack_2((grid_x, grid_y))  # h, w, 2
        ms.ops.stop_gradient(grid)

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        w_ = self.max(ms.Tensor(w - 1), ms.Tensor(1))
        h_ = self.max(ms.Tensor(h - 1), ms.Tensor(1))
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / w_ - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / h_ - 1.0
        grid_flow = self.stack_3((grid_flow_x, grid_flow_y))
        # Due to non-deterministic results of ms.ops.grid_sample function on GPU use the custom one
        output = self.grid_sample(x, grid_flow)
        return output


class SPyNet(nn.Cell):
    def __init__(
        self,
        pretrained=None,
        precompute_grid=True,
        base_resolution=([64, 64], [64, 128]),
        levels=6,
        eliminate_gradient_for_gather=False,
    ):
        super(SPyNet, self).__init__()

        self.basic_module = nn.CellList([SPyNetBasicModule() for _ in range(6)])
        self.interpolate = nn.ResizeBilinear()
        self.avg_pool2d = ops.AvgPool(kernel_size=2, strides=2, pad_mode="valid")
        self.zeros = ops.Zeros()
        self.concat = ops.Concat(axis=1)
        self.flow_warp = FlowWarp(
            precompute_grid=precompute_grid,
            base_resolution=base_resolution,
            levels=levels,
            eliminate_gradient_for_gather=eliminate_gradient_for_gather,
        )
        self.transpose = ops.Transpose()

        if isinstance(pretrained, str):
            mindspore.load_checkpoint(pretrained, self, strict_load=True)

        self.mean = ms.Tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1)
        self.std = ms.Tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1)

        device = ms.get_context("device_target")
        self.interpolate_func = ms.ops.interpolate if device in ["GPU", "Ascend"] else interpolate_through_grid_sample

    def normalize(self, inp):
        return (inp - self.mean) / self.std

    def construct(self, ref, supp):
        dtype = ref.dtype
        n, _, h, w = ref.shape
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = self.interpolate_func(
            ref, sizes=(h_up, w_up), coordinate_transformation_mode="half_pixel", mode="bilinear"
        )
        supp = self.interpolate_func(
            supp, sizes=(h_up, w_up), coordinate_transformation_mode="half_pixel", mode="bilinear"
        )

        # normalize the input images
        ref = [self.normalize(ref)]
        supp = [self.normalize(supp)]

        # generate downsampled frames
        for level in range(5):
            ref.append(self.avg_pool2d(ref[-1]))
            supp.append(self.avg_pool2d(supp[-1]))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = self.zeros((n, 2, h_up // 32, w_up // 32), dtype)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = self.interpolate(flow, scale_factor=2, align_corners=True) * 2.0
            # add the residual to the upsampled flow
            flow_warp_feat = self.flow_warp(supp[level], self.transpose(flow_up, (0, 2, 3, 1)), level)
            flow_feat = self.concat([ref[level], flow_warp_feat, flow_up])
            flow_feat = self.basic_module[level](flow_feat)
            flow = flow_up + flow_feat

        flow = self.interpolate_func(flow, sizes=(h, w), coordinate_transformation_mode="half_pixel", mode="bilinear")
        # adjust the flow values
        kw = float(w) / float(w_up)
        kh = float(h) / float(h_up)
        flow_w = flow[:, 0, :, :]
        flow_h = flow[:, 1, :, :]
        flow_w *= kw
        flow_h *= kh
        flow = ms.ops.stack([flow_w, flow_h], axis=1)
        return flow


if __name__ == "__main__":
    f = SPyNet()
    x1 = ms.Tensor(np.random.rand(1, 3, 128, 128), ms.float32)
    x2 = ms.Tensor(np.random.rand(1, 3, 128, 128), ms.float32)
    y = f(x1, x2)
