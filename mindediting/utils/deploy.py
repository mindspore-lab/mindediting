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
import numpy as np
from mindspore import Tensor

from mindediting.models import create_model_by_name


def create_export_helper(export_helper_name):
    available_export_helpers = {
        "default": default,
        "basicvsr": basicvsr,
        "ipt": ipt,
        "rrdb": default,
        "srdiff": srdiff,
    }
    return available_export_helpers[export_helper_name]


def default(cfg, input_shape):
    cfg.mode = "val"
    net, _ = create_model_by_name(model_name=cfg.model.name, cfg=cfg)
    input_array = Tensor(np.random.rand(*input_shape), mindspore.float32)
    if hasattr(net, "input_resolution"):
        net.input_resolution = input_shape

    return net, input_array


def basicvsr(cfg, input_shape):
    cfg.mode = "val"
    b, t, c, height, width = input_shape
    # Adjust image's resolution for SPyNet
    width_up = width if (width % 32) == 0 else 32 * (width // 32 + 1)
    height_up = height if (height % 32) == 0 else 32 * (height // 32 + 1)

    cfg.optimization.spynet.base_resolution = [[height_up, width_up], [height_up, width_up]]
    cfg.optimization.basicvsr.base_resolution = [[height, width], [height, width]]
    cfg.optimization.precompute_grid = True

    net, _ = create_model_by_name(model_name=cfg.model.name, cfg=cfg)

    # fix shape just for test
    input_array = Tensor(np.ones(input_shape).astype(np.float32))

    return net, input_array


def srdiff(cfg, input_shape):
    n, c, h, w = input_shape
    cfg.model.input_shape = input_shape
    net, _ = create_model_by_name(model_name=cfg.model.name, cfg=cfg)
    input_0 = Tensor(np.ones(input_shape).astype(np.float32))
    input_1 = Tensor(np.ones((n, c, h * cfg.model.scale, w * cfg.model.scale)).astype(np.float32))
    return net, (input_0, input_1)


def ipt(cfg, input_shape):
    cfg.mode = "val"
    # frame_seq, 3, height, width == input_shape
    net, _ = create_model_by_name(model_name=cfg.model.name, cfg=cfg)

    x = Tensor(np.ones(input_shape).astype(np.float32))
    h, w = x.shape[-2:]
    padsize = int(cfg.dataset.patch_size)
    x_hw_cut = x[:, :, (h - padsize) :, (w - padsize) :]
    input_array = x_hw_cut
    idx = Tensor(np.ones(cfg.task.alltask.task_id), mindspore.int32)

    return net, (input_array, idx)
