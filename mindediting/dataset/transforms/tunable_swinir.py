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

from mindediting.dataset.transforms.utils import add_gaussian_noise


def transforms_tunable_swinir_train(**kwargs):
    input_columns = ["HR", "LR"]
    output_columns = ["HR", "LR"]
    return [], input_columns, output_columns


def transforms_tunable_swinir_val(**kwargs):
    input_columns = ["HR", "LR"]
    output_columns = ["LR", "HR"]
    noise_stddev = kwargs.get("noise_stddev", None)
    sr_factor = kwargs.get("scale", None)
    use_seed = kwargs.get("use_seed", True)

    assert noise_stddev in [None, 15.0, 25.0, 50.0]
    assert sr_factor in [None, 4]
    assert not (noise_stddev is None and sr_factor is None)
    assert not (noise_stddev is not None and sr_factor is not None)

    def operation(HR, LR, batchInfo=1):
        scale = 255.0
        HR = [item / scale for item in HR]
        if sr_factor:
            LR = [item / scale for item in LR]
        elif noise_stddev:
            tmp_LR = []
            for index, item in enumerate(HR):
                seed = index if use_seed else None
                tmp_item = add_gaussian_noise(np.transpose(item, (1, 2, 0)), noise_stddev / scale, seed=seed)
                tmp_LR.append(np.transpose(tmp_item, (2, 0, 1)))
            LR = tmp_LR
        return LR, HR

    return operation, input_columns, output_columns
