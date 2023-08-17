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

from common import do_val_test


def test_basicvsr_plus_plus_light_video_denoising():
    update_cfg = {
        "dataset": {
            "input_path": "/data/LLVT/ci_test_data/test_vdn/",
        },
        "val_params": {
            "img_save_directory": None,
        },
    }
    do_val_test(
        "basicvsr_plus_plus_light", val_cfg_name="val_ascend.yaml", update_cfg=update_cfg, task="video_denoising"
    )
