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

from mindediting.deploy.data_io.ipt_sr.ipt_sr_io import IPTSRDataIO
from mindediting.deploy.utils.tiler import DefaultTiler

from .tiling_val import TilingValidator


class Model:
    def __init__(self, net, scale):
        self.model = net
        self.ipt_data_io = IPTSRDataIO()
        self.ipt_data_io.set_scale(scale)

    def preprocess(self, input_tensor):
        input_tensor = input_tensor.transpose((0, 2, 3, 1))
        input_tensor = self.ipt_data_io.preprocess(input_tensor)
        return input_tensor

    def postprocess(self, output_data):
        output_data = self.ipt_data_io.postprocess(output_data)
        output_data = output_data.transpose((2, 0, 1))[np.newaxis, :] / 255.0
        return output_data

    def run(self, input_tensor):
        input_tensor = self.preprocess(input_tensor)
        output_data = self.model(input_tensor)
        if not isinstance(output_data, list) and not isinstance(output_data, np.ndarray):
            output_data = output_data.asnumpy()
        output_data = self.postprocess(output_data)
        return output_data


class IPTValidator(TilingValidator):
    def __init__(
        self,
        net,
        eval_network,
        loss_fn,
        metrics,
        scale=1,
        temporal_size=0,
        temporal_overlap=0,
        spatial_size=(0, 0),
        spatial_overlap=0,
        dtype="float32",
        input_tensor_type="mindspore",
    ) -> None:
        super().__init__(
            net,
            eval_network,
            loss_fn,
            metrics,
            scale,
            temporal_size,
            temporal_overlap,
            spatial_size,
            spatial_overlap,
            dtype,
            input_tensor_type,
        )
        self.model = Model(eval_network, scale)
        self.tiler = DefaultTiler(
            self.model,
            frame_window_size=temporal_size,
            frame_overlap=temporal_overlap,
            patch_size=spatial_size,
            patch_overlap=spatial_overlap,
            sf=scale,
            dtype=dtype,
        )
