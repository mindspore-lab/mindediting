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

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from tt_common import do_val_train_val_test

cases = [
    [
        "configs/vrt/train.yaml",
        "configs/vrt/val.yaml",
        0,
        0,
        1,
        {},
    ],
    [
        "configs/rvrt/config.yaml",
        "configs/rvrt/config.yaml",
        100,
        1000,
        1,
        {},
    ],
    [
        "configs/rvrt_light/config.yaml",
        "configs/rvrt_light/config.yaml",
        100,
        1000,
        1,
        {},
    ],
    [
        "configs/basicvsr/train.yaml",
        "configs/basicvsr/val.yaml",
        10,
        10,
        1,
        {},
    ],
    [
        "configs/basicvsr_plus_plus_light/video_deblur/train.yaml",
        "configs/basicvsr_plus_plus_light/video_deblur/val.yaml",
        0,
        0,
        2,
        {
            "dataset": {
                "num_parallel_workers": 1,
                "max_rowsize": 32,
            },
        },
    ],
    [
        "configs/basicvsr_plus_plus_light/video_denoising/train_ascend.yaml",
        "configs/basicvsr_plus_plus_light/video_denoising/val_ascend.yaml",
        0,
        0,
        30,
        {
            "dataset": {
                "dataset_sink_mode": False,
            },
            "scheduler": {
                "base_lr": 8 * 1.0e-4,
                "min_lr": 8 * 1.0e-3,
                "warmup_epochs": 0,
            },
        },
    ],
    [
        "configs/basicvsr_plus_plus_light/video_super_resolution/train.yaml",
        "configs/basicvsr_plus_plus_light/video_super_resolution/val.yaml",
        10,
        10,
        1,
        {},
    ],
    [
        "configs/ttvsr/train.yaml",
        "configs/ttvsr/val.yaml",
        0,
        0,
        20,
        {
            "scheduler": {
                "warmup_epochs": 0,
            },
            "extra_scheduler": {
                "warmup_epochs": 0,
            },
        },
    ],
    [
        "configs/noahtcv/train.yaml",
        "configs/noahtcv/val.yaml",
        0,
        9,
        1,
        {
            "dataset": {
                "num_parallel_workers": 2,
                "batch_size": 1,
                "dataset_sink_mode": False,
            },
        },
    ],
    [
        "configs/rrdb/train.yaml",
        "configs/rrdb/val.yaml",
        0,
        13,
        1,
        {
            "dataset": {
                "num_parallel_workers": 1,
                "batch_size": 1,
                "dataset_sink_mode": False,
            },
        },
    ],
    [
        "configs/mimo_unet/train_deblur_gopro_ascend.yaml",
        "configs/mimo_unet/val_deblur_gopro_gpu.yaml",
        0,
        0,
        1,
        {
            "dataset": {
                "batch_size": 2,
                "dataset_sink_mode": False,
            },
        },
    ],
    [
        "configs/nafnet/train.yaml",
        "configs/nafnet/val.yaml",
        0,
        0,
        1,
        {},
    ],
    [
        "configs/fsrcnn/train_sr_x4_t91image_Ascend.yaml",
        "configs/fsrcnn/val_sr_x4_Set5_gpu.yaml",
        0,
        0,
        1,
        {
            "dataset": {
                "batch_size": 1,
                "num_parallel_workers": 1,
                "dataset_sink_mode": False,
            },
        },
    ],
    [
        "configs/srdiff/train.yaml",
        "configs/srdiff/val.yaml",
        0,
        13,
        1,
        {},
    ],
]


@pytest.mark.parametrize("train_cfg,val_cfg,train_every_nth,val_every_nth,epochs,train_update_cfg", cases)
def test_training(train_cfg, val_cfg, train_every_nth, val_every_nth, epochs, train_update_cfg):
    do_val_train_val_test(train_cfg, val_cfg, train_every_nth, val_every_nth, epochs, train_update_cfg)
