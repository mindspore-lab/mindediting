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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from common import do_train_test


def test_can_train_with_tiling_validator():
    update_cfg = {
        "dataset": {
            "every_nth": 10,
            "batch_size": 2,
            "input_path": "/data/LLVT/Basic_VSR/data/mini_vimeo_super_resolution_test",
        },
        "train_params": {
            "epoch_size": 1,
            "eval_frequency": 1,
            "need_val": True,
        },
        "validator": {
            "name": "tiling",
            "temporal_overlap": 0,
            "spatial_overlap": 0,
            "temporal_size": 0,
            "spatial_size": [0, 0],
            "scale": 4,
        },
    }
    do_train_test("basicvsr_plus_plus_light", update_cfg=update_cfg, task="video_super_resolution")
