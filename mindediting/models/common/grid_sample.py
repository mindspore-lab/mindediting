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
import mindspore.ops as ops
import mindspore.ops.operations.nn_ops as nn_ops


def _grid_sampler_unnormalize(coord, side, align_corners):
    if align_corners:
        return (0.5 * (coord + 1.0)) * (side - 1.0)
    else:
        return 0.5 * ((coord + 1.0) * side - 1.0)


def _index_processing(x0, x1, y0, y1, num_cols, pad):
    x0_int32 = x0.astype(ms.int32)
    x1_int32 = x1.astype(ms.int32)
    y0_int32 = y0.astype(ms.int32)
    y1_int32 = y1.astype(ms.int32)

    grid_y0_x0 = (y0_int32 + pad) * (num_cols + 2 * pad) + x0_int32 + pad
    grid_y1_x0 = (y1_int32 + pad) * (num_cols + 2 * pad) + x0_int32 + pad
    grid_y0_x1 = (y0_int32 + pad) * (num_cols + 2 * pad) + x1_int32 + pad
    grid_y1_x1 = (y1_int32 + pad) * (num_cols + 2 * pad) + x1_int32 + pad

    return grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1


class _UnnormalizeAlignCorners(nn.Cell):
    def construct(self, coord, sizes):
        sizes = sizes.reshape(1, 1, 1, 2)
        out = (0.5 * (coord + 1.0)) * (sizes - 1.0)

        return out


class _UnnormalizeHalfPixel(nn.Cell):
    def construct(self, coord, sizes):
        sizes = sizes.reshape(1, 1, 1, 2)
        out = 0.5 * ((coord + 1.0) * sizes - 1.0)

        return out


class _ZeroPadLimits(nn.Cell):
    def construct(self, H, W):
        min_limit = ms.Tensor([-2, -2])
        max_limit = ms.Tensor([W + 1, H + 1])

        return min_limit, max_limit


class _BorderPadLimits(nn.Cell):
    def construct(self, H, W):
        min_limit = ms.Tensor([0, 0])
        max_limit = ms.Tensor([W - 1, H - 1])

        return min_limit, max_limit


class _NearestInterpolation(nn.Cell):
    def __init__(self, pad):
        super().__init__()

        self.pad = pad

    def construct(self, input_pad, grid, N, C, W):
        nearest = ops.round(grid).astype(ms.int32)
        ix_nearest = nearest[:, :, :, 0]
        iy_nearest = nearest[:, :, :, 1]

        grid_nearest = (iy_nearest + self.pad) * (W + 2 * self.pad) + ix_nearest + self.pad
        grid_nearest = grid_nearest.reshape(N, 1, -1)
        grid_nearest = grid_nearest.broadcast_to((-1, C, -1))

        input_pad = input_pad.reshape(N, C, -1)
        out = ops.gather_elements(input_pad, 2, grid_nearest)
        out = out.reshape(N, C, *grid.shape[1:3])

        return out


class _BilinearInterpolation(nn.Cell):
    def __init__(self, pad):
        super().__init__()

        self.pad = pad
        self.mode = "default"

    @staticmethod
    def _index_processing(x0, x1, y0, y1, num_cols, pad, input_pad_shape):
        x0_int32 = x0.astype(ms.int32)
        x1_int32 = x1.astype(ms.int32)
        y0_int32 = y0.astype(ms.int32)
        y1_int32 = y1.astype(ms.int32)

        n, c, h, w = input_pad_shape
        assert w == num_cols + 2 * pad
        batch_stride = h * w
        neighbour_stride = n * batch_stride
        batch_offsets = ms.numpy.arange(0, n * batch_stride, batch_stride, dtype=ms.int32).reshape(-1, 1, 1, 1)

        grid_y0_x0 = (y0_int32 + pad) * (num_cols + 2 * pad) + x0_int32 + pad + batch_offsets
        grid_y1_x0 = (y1_int32 + pad) * (num_cols + 2 * pad) + x0_int32 + pad + batch_offsets + neighbour_stride
        grid_y0_x1 = (y0_int32 + pad) * (num_cols + 2 * pad) + x1_int32 + pad + batch_offsets + 2 * neighbour_stride
        grid_y1_x1 = (y1_int32 + pad) * (num_cols + 2 * pad) + x1_int32 + pad + batch_offsets + 3 * neighbour_stride

        return grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1

    def construct(self, input_pad, grid, N, C, W):
        if self.mode == "default":
            return self.construct_x(input_pad, grid, N, C, W)
        return self.construct_shift_burst(input_pad, grid, N, C, W)

    def construct_x(self, input_pad, grid, N, C, W):
        # ops.Print()("BilinearGS")
        # ops.Print()(input_pad.shape)
        # ops.Print()(grid.shape)
        C_orig = C
        if C % 16 != 0:
            input_pad = ms.numpy.pad(input_pad, ((0, 0), (0, 16 - (C % 16)), (0, 0), (0, 0)))
            C = input_pad.shape[1]
            # return self.construct_opt(input_pad, grid, N, C, W)

        ix = grid[:, :, :, 0:1]
        iy = grid[:, :, :, 1:2]

        x0 = ops.floor(ix)
        x1 = x0 + 1.0
        y0 = ops.floor(iy)
        y1 = y0 + 1.0

        w_a = ((x1 - ix) * (y1 - iy)).reshape(N, 1, -1)
        w_b = ((x1 - ix) * (iy - y0)).reshape(N, 1, -1)
        w_c = ((ix - x0) * (y1 - iy)).reshape(N, 1, -1)
        w_d = ((ix - x0) * (iy - y0)).reshape(N, 1, -1)
        w = ops.concat([w_a, w_b, w_c, w_d], axis=0).reshape(1, 4, -1, 1)

        grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1 = self._index_processing(
            x0, x1, y0, y1, W, self.pad, input_pad.shape
        )

        grids = ops.concat([grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1], axis=0)
        grids = grids.reshape(-1)

        # input_pad [N, C, Hpad, Wpad]
        assert C % 16 == 0
        C1 = C // 16
        input_pad = input_pad.reshape(N, C1, 16, -1)
        input_pad = ops.transpose(input_pad, (1, 0, 3, 2))  # [C // 16, N, spatial, 16]
        input_pad = ops.tile(input_pad, (1, 4, 1, 1))
        input_pad = input_pad.reshape(C1, -1, 16)  # [C // 16, 4 * N * Hpad * Wpad, 16]

        values = ops.gather(input_pad, grids, axis=1)

        values = values.reshape(C1, 4, -1, 16)
        out = w * values
        out = ops.reduce_sum(out, 1)  # [C // 16, 1, N * Hpad * Wpad, 16]

        out = out.reshape(C1, N, -1, 16)
        out = ops.transpose(out, (1, 0, 3, 2))
        out = out.reshape(N, C, -1)

        if C_orig != C:
            out = out[:, :C_orig]

        return out

    def construct_opt(self, input_pad, grid, N, C, W):
        ix = grid[:, :, :, 0:1]
        iy = grid[:, :, :, 1:2]

        x0 = ops.floor(ix)
        x1 = x0 + 1.0
        y0 = ops.floor(iy)
        y1 = y0 + 1.0

        w_a = ((x1 - ix) * (y1 - iy)).reshape(N, 1, -1)
        w_b = ((x1 - ix) * (iy - y0)).reshape(N, 1, -1)
        w_c = ((ix - x0) * (y1 - iy)).reshape(N, 1, -1)
        w_d = ((ix - x0) * (iy - y0)).reshape(N, 1, -1)
        w = ops.concat([w_a, w_b, w_c, w_d], axis=0).reshape(1, 4, -1)

        grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1 = self._index_processing(
            x0, x1, y0, y1, W, self.pad, input_pad.shape
        )

        grids = ops.concat([grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1], axis=0)
        grids = grids.reshape(-1)

        # input_pad [N, C, Hpad, Wpad]
        input_pad = ops.transpose(input_pad, (1, 0, 2, 3))
        input_pad = input_pad.reshape(C, 1, -1)
        input_pad = input_pad.broadcast_to((-1, 4, -1))
        input_pad = input_pad.reshape(C, -1)  # [C, 4 * N * Hpad * Wpad]

        values = ops.gather(input_pad, grids, axis=1)
        values = values.reshape(C, 4, -1)

        out = w * values
        out = ops.reduce_sum(out, 1)
        out = out.reshape(C, N, -1)
        out = ops.transpose(out, (1, 0, 2))

        return out

    def construct_shift_batch(self, input_pad, grid, N, C, W):
        ix = grid[:, :, :, 0:1]
        iy = grid[:, :, :, 1:2]

        x0 = ops.floor(ix)
        x1 = x0 + 1.0
        y0 = ops.floor(iy)
        y1 = y0 + 1.0

        w_a = ((x1 - ix) * (y1 - iy)).reshape(N, 1, -1)
        w_b = ((x1 - ix) * (iy - y0)).reshape(N, 1, -1)
        w_c = ((ix - x0) * (y1 - iy)).reshape(N, 1, -1)
        w_d = ((ix - x0) * (iy - y0)).reshape(N, 1, -1)
        w = ops.concat([w_a, w_b, w_c, w_d], axis=0).reshape(4, 1, -1)

        grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1 = self._index_processing(
            x0, x1, y0, y1, W, self.pad, input_pad.shape
        )

        # grids = ops.concat([grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1], axis=0)
        grids = grid_y0_x0
        grids = grids.reshape(-1)
        # grids = grids.broadcast_to((-1, C, -1))

        # input_pad [N, C, Hpad, Wpad]
        input_pad = ops.transpose(input_pad, (1, 0, 2, 3))  # [C, N, Hpad, Wpad]
        input_pad_pad = ops.pad(input_pad, ((0, 0), (0, 0), (0, 1), (0, 1)))
        input_pad = ops.concat(
            [
                input_pad,
                input_pad_pad[..., 1:, :-1],
                input_pad_pad[..., :-1, 1:],
                input_pad_pad[..., 1:, 1:],
            ],
            axis=0,
        )  # [4 * C, N, Hpad, Wpad]

        # input_pad = input_pad.reshape(C, 1, -1)
        # input_pad = input_pad.broadcast_to((-1, 4, -1))
        input_pad = input_pad.reshape(4 * C, -1)  # [4 * C, N * Hpad * Wpad]

        values = ops.gather(input_pad, grids, axis=1)
        values = values.reshape(4, C, -1)

        out = w * values
        out = ops.reduce_sum(out, 0)
        out = out.reshape(C, N, -1)
        out = ops.transpose(out, (1, 0, 2))

        return out

    @staticmethod
    def _index_processing_shift_burst(x0, y0, num_cols, pad, input_pad_shape):
        x0_int32 = x0.astype(ms.int32)
        y0_int32 = y0.astype(ms.int32)

        n, c, h, w = input_pad_shape
        assert w == num_cols + 2 * pad
        batch_stride = h * w
        batch_offsets = ms.numpy.arange(0, n * batch_stride, batch_stride, dtype=ms.int32).reshape(-1, 1, 1, 1)

        grid_y0_x0 = (y0_int32 + pad) * w + x0_int32 + pad + batch_offsets

        return grid_y0_x0

    def construct_shift_burst(self, input_pad, grid, N, C, W):
        ix = grid[:, :, :, 0:1]
        iy = grid[:, :, :, 1:2]

        x0 = ops.floor(ix)
        x1 = x0 + 1.0
        y0 = ops.floor(iy)
        y1 = y0 + 1.0

        w_a = ((x1 - ix) * (y1 - iy)).reshape(N, 1, -1)
        w_b = ((x1 - ix) * (iy - y0)).reshape(N, 1, -1)
        w_c = ((ix - x0) * (y1 - iy)).reshape(N, 1, -1)
        w_d = ((ix - x0) * (iy - y0)).reshape(N, 1, -1)
        w = ops.stack([w_a, w_b, w_c, w_d], axis=-1)
        w = w.reshape(-1, 4, 1)  # [N * Hpad * Wpad, 4, 1]

        grids = self._index_processing_shift_burst(x0, y0, W, self.pad, input_pad.shape)
        grids = grids.reshape(-1)

        # input_pad [N, C, Hpad, Wpad]
        input_pad = ops.transpose(input_pad, (0, 2, 3, 1))  # [N, Hpad, Wpad, C]
        input_pad_pad = ms.numpy.pad(input_pad, ((0, 0), (0, 1), (0, 1), (0, 0)))
        input_pad = ops.concat(
            [
                input_pad,
                input_pad_pad[:, 1:, :-1],
                input_pad_pad[:, :-1, 1:],
                input_pad_pad[:, 1:, 1:],
            ],
            axis=-1,
        )  # [N, Hpad, Wpad, 4 * C]

        input_pad = input_pad.reshape(-1, 4 * C)  # [N * Hpad * Wpad, 4 * C]

        values = ops.gather(input_pad, grids, axis=0)  # [N * Hgrid * Wgrid, 4 * C]
        values = values.reshape(-1, 4, C)  # [N * Hgrid * Wgrid, 4, C]

        out = w * values
        out = ops.reduce_sum(out, 1)
        # out = ops.BatchMatMul(True, False)(values, w)
        out = out.reshape(N, -1, C)  # [N, Hgrid * Wgrid, C]

        # FIXME
        # out = ops.transpose(out, (0, 2, 1))  # [N, C, Hgrid * Wgrid]

        return out


class GridSample2D(nn.Cell):
    def __init__(self, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False, do_reshape=True):
        super().__init__()

        assert interpolation_mode in ["bilinear", "nearest"]
        assert padding_mode in ["zeros", "border"]

        self.pad_value = 1 if padding_mode == "border" else 3

        self.unnormalize = _UnnormalizeAlignCorners() if align_corners else _UnnormalizeHalfPixel()
        self.interpolate = (
            _NearestInterpolation(self.pad_value)
            if interpolation_mode == "nearest"
            else _BilinearInterpolation(self.pad_value)
        )
        self.pad_limits = _BorderPadLimits() if padding_mode == "border" else _ZeroPadLimits()
        self.do_reshape = do_reshape

    def construct(self, input_x, grid):
        dtype = input_x.dtype
        N, C, H, W = input_x.shape
        grid_H, grid_W = grid.shape[1:3]

        sizes = ms.Tensor([W, H], dtype)
        unnorm_grid = self.unnormalize(ops.stop_gradient(grid), sizes)

        min_limit, max_limit = self.pad_limits(H, W)
        unnorm_grid = ops.clip_by_value(unnorm_grid, min_limit.astype(dtype), max_limit.astype(dtype))

        input_pad = ms.numpy.pad(
            input_x, ((0, 0), (0, 0), (self.pad_value, self.pad_value), (self.pad_value, self.pad_value))
        )

        out = self.interpolate(input_pad, unnorm_grid, N, C, W)
        if self.do_reshape:
            out = out.reshape(N, C, grid_H, grid_W)

        return out


def build_grid_sample(interpolation_mode="bilinear", padding_mode="zeros", align_corners=False, fast_grid_sample=True):
    if fast_grid_sample:
        return GridSample2D(interpolation_mode, padding_mode, align_corners)
    else:
        return nn_ops.GridSampler2D(interpolation_mode, padding_mode, align_corners)


class GridSample(nn.Cell):
    def __init__(
        self,
        interpolation_mode="bilinear",
        padding_mode="border",
        align_corners=True,
        unnormalize=True,
        eliminate_gradient_for_gather=True,
    ):
        super(GridSample, self).__init__()
        assert interpolation_mode in ["bilinear", "nearest"]
        assert padding_mode in ["zeros", "border"]
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.unnormalize = unnormalize
        if padding_mode == "border":
            self.padding = 1
            self.pad = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        else:
            self.padding = 3
            self.pad = nn.Pad(((0, 0), (0, 0), (3, 3), (3, 3)))
        self.eliminate_gradient_for_gather = eliminate_gradient_for_gather
        self.gather_elements = ms.ops.GatherD()

    def _grid_sampler_unnormalize(self, coord, side):
        if self.align_corners:
            return (0.5 * (coord + 1.0)) * (side - 1.0)
        else:
            return 0.5 * ((coord + 1.0) * side - 1.0)

    def _pad(self, x):
        # This padding is used for export to .om
        dtype = x.dtype
        n, c, h, w = x.shape
        offset = int(self.padding * 2)
        th, tw = h + offset, w + offset
        w0z = ms.ops.zeros((n, c, h, self.padding), dtype)
        w1z = ms.ops.zeros((n, c, h, self.padding), dtype)
        h0z = ms.ops.zeros((n, c, self.padding, tw), dtype)
        h1z = ms.ops.zeros((n, c, self.padding, tw), dtype)
        x_out = ms.ops.concat([w0z, x, w1z], axis=-1)
        x_out = ms.ops.concat([h0z, x_out, h1z], axis=-2)
        return x_out

    def _index_processing(self, x0, x1, y0, y1, num_cols):
        pad = self.padding
        x0 = x0.astype(ms.int32)
        x1 = x1.astype(ms.int32)
        y0 = y0.astype(ms.int32)
        y1 = y1.astype(ms.int32)

        grid_y0_x0 = (y0 + pad) * (num_cols + 2 * pad) + x0 + pad
        grid_y1_x0 = (y1 + pad) * (num_cols + 2 * pad) + x0 + pad
        grid_y0_x1 = (y0 + pad) * (num_cols + 2 * pad) + x1 + pad
        grid_y1_x1 = (y1 + pad) * (num_cols + 2 * pad) + x1 + pad

        return grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1

    def construct(self, input_x, grid):
        dtype = input_x.dtype

        N, C, H, W = input_x.shape
        grid_H, grid_W = grid.shape[1:3]

        x = grid[:, :, :, 0].expand_dims(-1)
        y = grid[:, :, :, 1].expand_dims(-1)

        if self.unnormalize:
            ix = self._grid_sampler_unnormalize(x, W)
            iy = self._grid_sampler_unnormalize(y, H)
        else:
            ix = x
            iy = y

        # apply border padding
        min_limit = ms.Tensor(-2.0, dtype)
        max_w_limit = ms.Tensor(W + 1, dtype)
        max_h_limit = ms.Tensor(H + 1, dtype)
        if self.padding_mode == "border":
            min_limit = ms.Tensor(0.0, dtype)
            max_w_limit = ms.Tensor(W - 1, dtype)
            max_h_limit = ms.Tensor(H - 1, dtype)
        ix = ms.ops.clip_by_value(ix, min_limit, max_w_limit)
        iy = ms.ops.clip_by_value(iy, min_limit, max_h_limit)

        input_pad = self._pad(input_x)
        input_pad = input_pad.reshape(N, C, -1)
        out = None

        if self.interpolation_mode == "nearest":
            ix_nearest = ops.round(ix).astype(ms.int32)
            iy_nearest = ops.round(iy).astype(ms.int32)
            grid_nearest = (iy_nearest + self.padding) * (W + 2 * self.padding) + ix_nearest + self.padding
            out = ops.gather_elements(input_pad, 2, grid_nearest.reshape(N, 1, -1).broadcast_to((-1, C, -1)))
        else:
            x0 = ms.ops.floor(ix)
            x1 = x0 + 1.0
            y0 = ms.ops.floor(iy)
            y1 = y0 + 1.0

            w_a = ((x1 - ix) * (y1 - iy)).reshape(N, 1, -1)
            w_b = ((x1 - ix) * (iy - y0)).reshape(N, 1, -1)
            w_c = ((ix - x0) * (y1 - iy)).reshape(N, 1, -1)
            w_d = ((ix - x0) * (iy - y0)).reshape(N, 1, -1)

            grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1 = self._index_processing(x0, x1, y0, y1, W)

            grids = ms.ops.concat(
                [
                    grid_y0_x0.reshape(N, 1, -1),
                    grid_y1_x0.reshape(N, 1, -1),
                    grid_y0_x1.reshape(N, 1, -1),
                    grid_y1_x1.reshape(N, 1, -1),
                ],
                axis=2,
            )
            if self.eliminate_gradient_for_gather:
                abcd = ms.ops.stop_gradient(self.gather_elements(input_pad, 2, grids.broadcast_to((-1, C, -1))))
            else:
                grids = ms.ops.concat([grids for _ in range(C)], axis=1)
                abcd = self.gather_elements(input_pad, 2, grids)
            offset = grid_W * grid_H
            a = abcd[:, :, 0:offset]
            b = abcd[:, :, offset : offset * 2]
            c = abcd[:, :, offset * 2 : offset * 3]
            d = abcd[:, :, offset * 3 :]

            out = a * w_a + b * w_b + c * w_c + d * w_d

        return out.reshape(N, C, grid_H, grid_W)


def grid_sample_2d(
    input_x, grid, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False, dtype=ms.float32
):
    """
    input_x shape = [N, C, H, W]
    grid shape  = [N, H, W, 2]
    output shape = [N, C, H, W]
    """

    assert interpolation_mode in ["bilinear", "nearest"]
    assert padding_mode in ["zeros", "border"]

    N, C, H, W = input_x.shape
    grid_H, grid_W = grid.shape[1:3]

    x = grid[:, :, :, 0].expand_dims(-1)
    y = grid[:, :, :, 1].expand_dims(-1)

    ix = _grid_sampler_unnormalize(x, W, align_corners).astype(dtype)
    iy = _grid_sampler_unnormalize(y, H, align_corners).astype(dtype)

    min_limit = None
    max_w_limit = None
    max_h_limit = None
    pad = None
    # apply border padding
    if padding_mode == "zeros":
        pad = 3
        min_limit = ms.Tensor(-2.0, dtype)
        max_w_limit = ms.Tensor(W + 1, dtype)
        max_h_limit = ms.Tensor(H + 1, dtype)
    elif padding_mode == "border":
        pad = 1
        min_limit = ms.Tensor(0.0, dtype)
        max_w_limit = ms.Tensor(W - 1, dtype)
        max_h_limit = ms.Tensor(H - 1, dtype)
    ix = ops.clip_by_value(ix, min_limit, max_w_limit)
    iy = ops.clip_by_value(iy, min_limit, max_h_limit)

    input_pad = ms.numpy.pad(input_x, ((0, 0), (0, 0), (pad, pad), (pad, pad))).reshape(N, C, -1)

    out = None

    if interpolation_mode == "nearest":
        ix_nearest = ops.round(ix).astype(ms.int32)
        iy_nearest = ops.round(iy).astype(ms.int32)
        grid_nearest = (iy_nearest + pad) * (W + 2 * pad) + ix_nearest + pad

        out = ops.gather_elements(input_pad, 2, grid_nearest.reshape(N, 1, -1).broadcast_to((-1, C, -1)))
    elif interpolation_mode == "bilinear":
        x0 = ops.floor(ix)
        x1 = x0 + 1.0
        y0 = ops.floor(iy)
        y1 = y0 + 1.0

        w_a = ((x1 - ix) * (y1 - iy)).reshape(N, 1, -1)
        w_b = ((x1 - ix) * (iy - y0)).reshape(N, 1, -1)
        w_c = ((ix - x0) * (y1 - iy)).reshape(N, 1, -1)
        w_d = ((ix - x0) * (iy - y0)).reshape(N, 1, -1)

        grid_y0_x0, grid_y1_x0, grid_y0_x1, grid_y1_x1 = _index_processing(x0, x1, y0, y1, W, pad=pad)

        grid_y0_x0 = grid_y0_x0.reshape(N, 1, -1)
        grid_y1_x0 = grid_y1_x0.reshape(N, 1, -1)
        grid_y0_x1 = grid_y0_x1.reshape(N, 1, -1)
        grid_y1_x1 = grid_y1_x1.reshape(N, 1, -1)

        # grid_y0_x0 = grid_y0_x0.broadcast_to((-1, C, -1))
        # grid_y1_x0 = grid_y1_x0.broadcast_to((-1, C, -1))
        # grid_y0_x1 = grid_y0_x1.broadcast_to((-1, C, -1))
        # grid_y1_x1 = grid_y1_x1.broadcast_to((-1, C, -1))

        grid_y0_x0 = ms.ops.concat([grid_y0_x0 for _ in range(C)], axis=1)
        grid_y1_x0 = ms.ops.concat([grid_y1_x0 for _ in range(C)], axis=1)
        grid_y0_x1 = ms.ops.concat([grid_y0_x1 for _ in range(C)], axis=1)
        grid_y1_x1 = ms.ops.concat([grid_y1_x1 for _ in range(C)], axis=1)

        a = ops.gather_elements(input_pad, 2, grid_y0_x0)
        b = ops.gather_elements(input_pad, 2, grid_y1_x0)
        c = ops.gather_elements(input_pad, 2, grid_y0_x1)
        d = ops.gather_elements(input_pad, 2, grid_y1_x1)

        out = a * w_a + b * w_b + c * w_c + d * w_d

    return out.reshape(N, C, grid_H, grid_W)
