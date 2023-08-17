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
from deploy.data_io.srdiff_sr.pipeline import create_pipeline
from deploy.data_io.srdiff_sr.utils_preprocess import tensor2img
from deploy.utils.config import Config


class SRDiffDataIO(DataIO):
    def set_scale(self, scale):
        self.scale = scale

    def preprocess(self, input_data):
        if isinstance(input_data, list):
            assert len(input_data) == 1
            input_data = input_data[0]
        pipeline = [
            {"Resize": {"in_key": "lq", "out_key": "lq_up", "scale": self.scale}},
            {"RescaleToZeroOne": {"keys": ["lq", "lq_up"]}},
            {"Normalize": {"keys": ["lq", "lq_up"]}},
            {"Collect": {"keys": ["lq", "lq_up"]}},
        ]
        pipeline = [Config(transform) for transform in pipeline]
        pipeline = create_pipeline(pipeline)
        result = pipeline({"lq": input_data.transpose(2, 0, 1)})
        return [[np.expand_dims(x, axis=0) for x in result]]

    def postprocess(self, input_data):
        while isinstance(input_data, list):
            input_data = input_data[0]
        return tensor2img(input_data, swap_red_blue=False)[0]
