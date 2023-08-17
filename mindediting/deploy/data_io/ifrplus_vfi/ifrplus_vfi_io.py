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
from deploy.data_io.data_metaclass import DataIO


def _img_to_tensor(img):
    out = np.transpose(img, (2, 0, 1))  # HWC to CHW
    out = out.astype(np.float32) / 255.0
    out = np.expand_dims(out, axis=0)

    return out


def _tensor_to_img(tensor):
    out = tensor.squeeze(axis=0).astype(np.float32).clip(0, 1)
    out = (out * 255.0).round().astype(np.uint8)

    return out


class IFRPlusVFIDataIO(DataIO):
    def __init__(self):
        super().__init__()

        self._inputs = None

    def preprocess(self, input_data):
        assert len(input_data) == 2
        self._prev_frame = input_data[0]

        out_data = [_img_to_tensor(x) for x in input_data]
        out_data = np.concatenate(out_data)
        out_data = np.expand_dims(out_data, axis=0)  # [1, T, C, H, W]

        return out_data

    def postprocess(self, model_output):
        out_data = [self._prev_frame] if self._prev_frame is not None else []

        images = _tensor_to_img(model_output)

        if len(images.shape) == 4:
            images = np.transpose(images, (0, 2, 3, 1))  # TCHW to THWC
            out_data.extend(np.split(images, images.shape[0]))
        elif len(images.shape) == 3:
            image = np.transpose(images, (1, 2, 0))  # CHW to HWC
            out_data.append(image)

        return out_data
