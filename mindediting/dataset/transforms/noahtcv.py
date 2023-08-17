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
from mindspore.dataset import vision
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import Inter


def transform_noahtcv_train(**kwargs):
    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["image", "label"]

    def operation(HR, LR, idx, filename, batchInfo=1):
        new_image, new_label = [], []
        for i in range(len(HR)):
            label, image = HR[i], LR[i]

            noise = np.random.randn(*label.shape) * 50
            image = np.clip(noise + label, 0, 255)

            image = image.astype(np.float32) / 255
            label = label.astype(np.float32) / 255
            new_image.append(image)
            new_label.append(label)
        return new_image, new_label

    return operation, input_columns, output_columns


def transform_noahtcv_val(**kwargs):
    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["image", "label"]

    def operation(HR, LR, idx, filename, batchInfo=1):
        new_image, new_label = [], []
        for i in range(len(HR)):
            label, image = HR[i], LR[i]
            image = image.astype(np.float32) / 255
            label = label.astype(np.float32) / 255
            # resize image to be dividable by 8 to suit model architecture
            image = image[:, 0 : image.shape[1] - 1, 0 : image.shape[2] - 1]
            label = label[:, 0 : label.shape[1] - 1, 0 : label.shape[2] - 1]
            new_image.append(image)
            new_label.append(label)
        return new_image, new_label

    return operation, input_columns, output_columns
