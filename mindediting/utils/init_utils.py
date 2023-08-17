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

import random

import mindspore
import numpy as np
from mindspore import context, set_seed
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode

from mindediting.utils.device_adapter import get_device_id, get_device_num
from mindediting.utils.utils import check_paths


def init_dist():
    """initialization for distributed training"""
    init("nccl")
    mindspore.context.set_context(device_id=get_rank())
    mindspore.set_auto_parallel_context(
        device_num=get_group_size(), parallel_mode=mindspore.ParallelMode.DATA_PARALLEL, gradients_mean=True
    )


def get_current_device():
    return context.get_context("device_target").lower()


def init_env(cfg):
    """
    init env for mindspore
    """
    cfg.system.device_target = context.get_context("device_target")
    # seed
    if cfg.system.random_seed is not None:
        random.seed(cfg.system.random_seed)
    if cfg.system.mindspore_seed is not None:
        set_seed(cfg.system.mindspore_seed)
    if cfg.system.numpy_seed is not None:
        np.random.seed(cfg.system.numpy_seed)

    # mode
    mode = cfg.system.context_mode.lower()
    context_mode = None
    if mode == "pynative_mode":
        context_mode = context.PYNATIVE_MODE
    elif mode == "graph_mode":
        context_mode = context.GRAPH_MODE
    context.set_context(mode=context_mode, device_target=cfg.system.device_target)

    context.set_context(max_call_depth=10000)

    device_num = get_device_num()

    if cfg.system.graph_kernel_flags is not None:
        context.set_context(graph_kernel_flags=cfg.system.graph_kernel_flags)

    if cfg.system.device_target == "Ascend":
        context.set_context(device_id=get_device_id())
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                parameter_broadcast=True,
                gradients_mean=True,
            )
    elif cfg.system.device_target == "GPU":
        context.set_context(enable_graph_kernel=getattr(cfg.system, "enable_graph_kernel", True))
        if device_num > 1:
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                parameter_broadcast=True,
                gradients_mean=True,
            )
    elif cfg.system.device_target == "CPU":
        pass
    else:
        raise ValueError("Unsupported platform.")
    check_paths(cfg)
