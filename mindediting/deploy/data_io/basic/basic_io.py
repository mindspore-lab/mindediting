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


class BasicDataIO(DataIO):
    def preprocess(self, input_data):
        # return numpy array of shape [1, T, C, H, W]
        outdata = []
        for image in input_data:
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            outdata.append(image)
        outdata = np.concatenate(outdata)
        outdata = np.expand_dims(outdata, axis=0)
        return outdata

    def postprocess(self, input_data):
        outdata = []
        img = input_data.squeeze(axis=0).astype(np.float32).clip(0, 1)
        img = (img * 255.0).round().astype(np.uint8)
        if len(img.shape) == 4:
            img = np.transpose(img, (0, 2, 3, 1))  # TCHW to THWC
            for i in range(img.shape[0]):
                outdata.append(img[i, ...].squeeze())
        elif len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))  # CHW to HWC
            outdata.append(img)
        return outdata
