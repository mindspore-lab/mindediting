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
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from common import do_train_test


def test_can_train():
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(tmpdirname)
        update_cfg = {
            "train": {
                "train_iter": 10,
            },
            "train_params": {
                "log_frequency_step": 1,
                "ckpt_save_dir": tmpdirname,
            },
        }
        do_train_test("ctsdg", train_cfg_name="train_inpainting_celeba_ascend.yaml", update_cfg=update_cfg)
