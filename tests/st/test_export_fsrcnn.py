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
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "mindediting"))

from common import FUSION_SWITCH_PATH, do_export_test


def test_fsrcnn_sr4x():
    update_cfg = {
        "dataset": {
            "every_nth": 10,
        }
    }
    do_export_test(
        "fsrcnn",
        val_cfg_name="val_sr_x4_Set5_gpu.yaml",
        shape=[1, 1, 32, 32],
        output_type="FP32",
        precision_mode="allow_mix_precision",
        target_format="om",
        soc_version="Ascend910A",
        fusion_switch_file=FUSION_SWITCH_PATH,
        log_level="info",
        eps=3e-3,
        update_cfg=update_cfg,
        task="",
    )
