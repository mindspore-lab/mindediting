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


def transform_fsrcnn(**kwargs):
    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["lr", "hr"]

    def operation(HR, LR, idx, filename, batchInfo=1):
        new_image, new_label = [], []
        for i in range(len(HR)):
            label, image = HR[i], LR[i]
            image = image / 255.0
            label = label / 255.0
            new_image.append(image)
            new_label.append(label)
        return new_image, new_label

    return operation, input_columns, output_columns
