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

import mindspore.numpy as mnp

from ..common.grid_sample import GridSample


def interpolate_through_grid_sample(
    x, scales=None, sizes=None, coordinate_transformation_mode="half_pixel", mode="bilinear"
):
    """Reproduces bilinear interpolation with option 'half_pixel' on CPU.
    ops.interpolalate function with 'half_pixel' has the same results as Pytorch's
    interpolate function with align_corners = False. This implementation uses grid_sample
    function to get the same result and works correctly not in all cases and connot be used everywhere.
    To use it firstly should check the result with a reference one.
    """
    assert coordinate_transformation_mode == "half_pixel"
    assert mode == "bilinear"
    if (scales is None and sizes is None) or (scales is not None and sizes is not None):
        raise ValueError(
            f"Should be specify exactly one of the following arguments: "
            f"'size' or 'scale' but got size = {sizes}, scale = {scales}"
        )
    n, c, h, w = x.shape
    if scales is not None:
        if isinstance(scales, (tuple, list)) and len(scales) == 4:
            n_s, c_s, h_scale, w_scale = scales
        elif isinstance(scales, (tuple, list)) and len(scales) == 2:
            h_scale, w_scale = scales
        elif isinstance(scales, (int, float)):
            h_scale = w_scale = scales
        else:
            raise ValueError("Unsupported scale value for the interpolation function.")
        h1, w1 = int(h * h_scale), int(w * w_scale)
    else:
        h1, w1 = sizes
    w_scale = float(w1) / float(w)
    h_scale = float(h1) / float(h)
    step_w = 1.0 / w_scale
    step_h = 1.0 / h_scale
    offset_w = float(w1 - w) / float(w1) * 0.5
    offset_h = float(h1 - h) / float(h1) * 0.5
    grid_h = mnp.arange(0, h, step_h)[:h1]
    grid_w = mnp.arange(0, w, step_w)[:w1]
    grid_h[1:-1] -= offset_h
    grid_w[1:-1] -= offset_w
    grid = mnp.stack(mnp.meshgrid(grid_w, grid_h), axis=-1)
    grid = mnp.stack([grid] * n, axis=0)
    return GridSample(unnormalize=False)(x, grid)
