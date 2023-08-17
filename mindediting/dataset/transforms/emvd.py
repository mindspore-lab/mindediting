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

import numpy as np


def transforms_emvd_train(**kwargs):
    input_columns = ["image", "label", "noisy_level"]
    output_columns = ["in_tensor", "gt_tensor", "coeff_a", "coeff_b"]
    frame_num = kwargs.get("frame_num", 25)

    def operation(images, labels, noisy_levels, batchInfo=1):
        in_tensors, coeff_as, coeff_bs, gt_tensors = [], [], [], []
        for i in range(len(images)):
            in_data = images[i]
            gt_raw_data = labels[i]
            noisy_level = noisy_levels[i]

            coeff_a = noisy_level[0] / (2**12 - 1 - 240)
            coeff_b = noisy_level[1] / (2**12 - 1 - 240) ** 2

            in_tensor = in_data[: (frame_num + 1) * 4, :, :]
            gt_tensor = gt_raw_data[: (frame_num + 1) * 4, :, :]
            in_tensors.append(in_tensor)
            gt_tensors.append(gt_tensor)
            coeff_as.append(coeff_a)
            coeff_bs.append(coeff_b)

        coeff_as = np.array(coeff_as, dtype=np.float32)[:, None, None, None]
        coeff_bs = np.array(coeff_bs, dtype=np.float32)[:, None, None, None]
        return in_tensors, gt_tensors, coeff_as, coeff_bs

    return operation, input_columns, output_columns


def transforms_emvd_val(**kwargs):
    input_columns = ["HR", "LR", "coeff_a", "coeff_b", "idx"]
    output_columns = ["LR", "HR", "coeff_a", "coeff_b", "idx"]

    def operation(HR, LR, coeff_a, coeff_b, idx, batchInfo=1):
        return LR, HR, coeff_a, coeff_b, idx

    return operation, input_columns, output_columns
