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
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.train.amp import _OutputTo32

from mindediting.models.common.with_eval_cell_out_fix import WithEvalCellOutFix

AMP_WHITE_LIST = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Dense,
    nn.LSTMCell,
    nn.RNNCell,
    nn.GRUCell,
    P.Conv2D,
    P.Conv3D,
    P.MatMul,
    P.BatchMatMul,
    P.GeLU,
    P.PReLU,
    P.ReLU,
    P.Ger,
]


def _auto_cast_with_white_list(network, white_list=None):
    if white_list is None:
        white_list = AMP_WHITE_LIST
    white_list = tuple(white_list)

    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, white_list):
            network._cells[name] = _OutputTo32(subcell.to_float(mindspore.float16))
            change = True
        else:
            _auto_cast_with_white_list(subcell, white_list)

    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


def cast_net_and_loss(config, net, loss):
    """
    Cast net (params and weights) and loss to new type (float16 or float32).
    """
    if config.train.cast_net:
        if config.train.cast_net == "float16":
            net.to_float(mindspore.float16)
        elif config.train.cast_net == "float32":
            net.to_float(mindspore.float32)
        elif config.train.cast_net == "base":
            pass
        elif config.train.cast_net == "white_list":
            _auto_cast_with_white_list(net)
        else:
            print(
                f"The network does not support conversion to type {config.train.cast_net}. The default value will be used."
            )

    if config.train.cast_loss:
        if config.train.cast_loss == "float16":
            loss.to_float(mindspore.float16)
        elif config.train.cast_loss == "float32":
            loss.to_float(mindspore.float32)
        elif config.train.cast_loss == "base":
            pass
        else:
            print(
                f"The loss does not support conversion to type {config.train.cast_loss}. The default value will be used."
            )


eval_network_dict = {"cast_fp32": WithEvalCellOutFix}
eval_index_dict = {"cast_fp32": [0, 1, 2]}


def prepare_model_kwargs(config, net, loss):
    """
    Create kwargs to init a Model object from config args.
    """
    model_kwargs = {}
    if config.train.is_use_dynamic_loss_scale:
        model_kwargs["loss_scale_manager"] = mindspore.DynamicLossScaleManager()
    if config.loss.amp_level:
        model_kwargs["amp_level"] = config.loss.amp_level
    if config.train.val_monitor and config.train.eval_network is not None:
        is_add_cast_fp32 = model_kwargs["amp_level"] in ["O2", "O3", "auto"]
        model_kwargs["eval_network"] = eval_network_dict[config.train.eval_network](net, loss, is_add_cast_fp32)
        model_kwargs["eval_indexes"] = eval_index_dict[config.train.eval_network]
    return model_kwargs
