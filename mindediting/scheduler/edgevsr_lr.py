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
from mindspore import nn


def get_lr_schedule_for_ifn(min_lr, max_lr, steps, freeze_steps):
    lr = nn.CosineDecayLR(min_lr, max_lr, steps)
    schedule = []
    for step in range(steps):
        schedule.append(lr(mindspore.Tensor(step)).asnumpy().item())
    schedule = np.array(schedule)
    schedule[:freeze_steps] *= 0.0
    return schedule.astype(np.float32)


def init_opt(opt, net, total_steps):
    """
    init opt to train
    """
    loss_scale = 1.0

    if opt["lr_scheme"] == "CosineAnnealingLR_Restart":
        net_lr = nn.CosineDecayLR(float(opt["eta_min"]), float(opt["lr_G"]), total_steps)
        ifn_lr = get_lr_schedule_for_ifn(
            float(opt["eta_min"]),
            float(opt["lr_G"]) * float(opt["ifn_lr_mul"]),
            total_steps,
            int(opt["ifn_freeze_steps"]),
        )
    else:
        raise ValueError("Unsupported learning rate policy.")

    net_params = []
    ifn_params = []
    for p in net.get_parameters():
        if p.requires_grad:
            if p.name.startswith("aggregation.alignment_net"):
                ifn_params.append(p)
            else:
                net_params.append(p)

    params = [
        {"params": net_params, "lr": net_lr, "weight_decay": float(opt["weight_decay"])},
        {"params": ifn_params, "lr": ifn_lr, "weight_decay": float(opt["weight_decay"])},
    ]

    optimizer = nn.Adam(params=params, loss_scale=loss_scale, beta1=opt["beta1"], beta2=opt["beta2"])
    return optimizer
