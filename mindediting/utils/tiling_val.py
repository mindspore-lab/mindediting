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

from mindediting.deploy.utils.tiler import DefaultTiler


class TilingValidator:
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
        self.metrics = metrics
        if dtype not in {"float32", "float16"}:
            raise ValueError(f"Invalid dtype: {dtype}")

        if input_tensor_type not in {"mindspore", "numpy"}:
            raise ValueError(f"Invalid input_tensor_type: {input_tensor_type}")
        self.input_tensor_type = input_tensor_type

        class Model:
            def __init__(self, net):
                self.model = net

            def run(self, input_tensor):
                output_data = self.model(input_tensor)
                if not isinstance(output_data, list) and not isinstance(output_data, np.ndarray):
                    output_data = output_data.asnumpy()
                return output_data

        self.model = Model(eval_network)
        self.tiler = DefaultTiler(
            self.model,
            frame_window_size=temporal_size,
            frame_overlap=temporal_overlap,
            patch_size=spatial_size,
            patch_overlap=spatial_overlap,
            sf=scale,
            dtype=dtype,
        )

    def eval(self, loader_val, dataset_sink_mode, callbacks):
        for i in range(len(callbacks)):
            callbacks[i].on_eval_begin(None)

        for lq, hq in loader_val:
            for i in range(len(callbacks)):
                callbacks[i].on_eval_step_begin(None)

            if self.input_tensor_type == "numpy" and not isinstance(lq, np.ndarray):
                lq = lq.asnumpy()
                hq = hq.asnumpy()

            hr = self.tiler(lq)

            for i in range(len(callbacks)):
                callbacks[i].on_eval_step_end(None)
            for k in self.metrics:
                self.metrics[k].update(hr, hq)

        for i in range(len(callbacks)):
            callbacks[i].on_eval_end(None)
