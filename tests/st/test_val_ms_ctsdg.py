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
import os.path as osp
import shutil
import sys
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from common import do_val_test


def test_ctsdg():
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(tmpdirname)
        src_data_dir = "/data/LLVT/CTSDG/data/CelebA"
        val_imgs_list = [
            "182638.jpg",
            "182639.jpg",
            "182640.jpg",
            "182641.jpg",
            "182642.jpg",
            "182643.jpg",
            "182644.jpg",
            "182645.jpg",
            "182646.jpg",
            "182647.jpg",
        ]
        for img_name in val_imgs_list:
            shutil.copy(osp.join(src_data_dir, "Img", "img_align_celeba", img_name), tmpdirname)
        train_val_partition_file_path = osp.join(tmpdirname, "list_eval_partition.txt")
        with open(train_val_partition_file_path, "wt") as f:
            f.writelines([f"{img_name} 2\n" for img_name in val_imgs_list])

        update_cfg = {
            "dataset": {
                "data_root": tmpdirname,
                "anno_path": train_val_partition_file_path,
            }
        }
        do_val_test("ctsdg", val_cfg_name="val_inpainting_celeba_ascend.yaml", update_cfg=update_cfg)
