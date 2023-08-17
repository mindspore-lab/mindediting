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

import mindspore as ms
from deploy.backend.backend_metaclass import Backend
from deploy.utils.config import Config, parse_yaml
from models import create_model_by_name
from munch import DefaultMunch


class MsBackend(Backend):
    """mindspore wrapper for inference.

    Args:
        model_file (str): Input mindspore model file.
        online_infer_config_file (str): Configuration file path for online inference.
        input_shape (list[int]): The model input tensor shape.
        output_shape (list[int]): The model output tensor shape.
    """

    def __init__(self, model_path: str, online_infer_config_file, input_shape, output_shape):
        super().__init__(model_path)
        self.model_path = model_path
        self.device_id = int(os.environ.get("DEVICE_ID", "0"))
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.get_model(model_path, online_infer_config_file)

    def get_model(self, model_path, online_infer_config_file):
        default, helper, choices = parse_yaml(online_infer_config_file)
        online_cfg = DefaultMunch.fromDict(default)
        online_cfg = Config(online_cfg)
        net, eval_network = create_model_by_name(model_name=online_cfg.model.name, cfg=online_cfg)
        return eval_network

    def run(self, input_list, **kwargs):
        """Run inference with mindspore session."""
        results = []
        for input_data in input_list:
            if isinstance(input_data, list):
                input_data = [ms.Tensor(i) for i in input_data]
                tmp_res = self.model(*input_data)
            else:
                input_data = ms.Tensor(input_data)
                tmp_res = self.model(input_data)
            if isinstance(tmp_res, ms.Tensor):
                tmp_res = tmp_res.asnumpy()
            results.append(tmp_res)
        return results

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.output_shape
