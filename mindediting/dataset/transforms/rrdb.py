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

from mindediting.dataset.transforms.basicvsr import create_pipeline


def transform_rrdb_train(**kwargs):
    input_columns = ["HR", "LR"]
    output_columns = ["image", "label"]
    pipeline = create_pipeline(kwargs.get("pipeline", []))
    scale = kwargs.get("scale", 4)

    def operation(hr, lr, batchInfo=1):
        images, labels, = (
            [],
            [],
        )
        for i in range(len(hr)):
            data = {
                "lq": lr[i],
                "gt": hr[i],
                "scale": scale,
            }
            img_lr, img_hr = pipeline(data)
            images.append(img_lr)
            labels.append(img_hr.copy())
        return images, labels

    return operation, input_columns, output_columns


def transform_rrdb_val(**kwargs):
    input_columns = ["HR", "LR"]
    output_columns = ["image", "label"]
    pipeline = create_pipeline(kwargs.get("pipeline", []))
    scale = kwargs.get("scale", 4)

    def operation(hr, lr, batchInfo=1):
        new_image, new_label = [], []
        for i in range(len(hr)):
            data = {"lq": lr[i], "gt": hr[i], "scale": scale}
            image, label = pipeline(data)
            new_image.append(image)
            new_label.append(label)
        return new_image, new_label

    return operation, input_columns, output_columns
