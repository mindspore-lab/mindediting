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
from importlib import import_module

from deploy.utils.tiler import DefaultTiler, SimpleImageTiler
from munch import DefaultMunch

BACKEND = {"ascend": "AscendBackend", "onnx": "OnnxBackend", "bolt": "BoltBackend", "ms": "MsBackend"}

DATA_IO = {
    "basic": "BasicDataIO",
    "ipt_sr": "IPTSRDataIO",
    "srdiff_sr": "SRDiffDataIO",
    "ifrplus_vfi": "IFRPlusVFIDataIO",
    "rgb_to_bgr": "RGB2BGRDataIO",
    "bgr_to_rgb": "BGR2RGBDataIO",
    "srdiff_sr_mindspore": "SRDiffDataIO",
}


class Task:
    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = DefaultMunch.fromDict(cfg)

        self.task_name = cfg.task_name
        self.once_process_frames = cfg.once_process_frames
        self.frame_overlap = cfg.get("frame_overlap", 0)
        self.patch_overlap = cfg.get("patch_overlap", 0)
        self.dtype = cfg.get("dtype", "float32")
        self.up_scale = cfg.get("up_scale", 1)
        self.tiling = cfg.get("tiling", None)
        self.tiler = None

        assert os.path.exists(cfg.model_file), "The model file does not exist!"

        if cfg.backend and cfg.model_file:
            backend_name = cfg.backend.lower()
            backend_module = import_module(f"deploy.backend.{backend_name}.{backend_name}_backend")
            backend_kwargs = cfg.get("backend_args", {})
            self.backend = getattr(backend_module, BACKEND[backend_name])(cfg.model_file, **backend_kwargs)
        else:
            self.backend = None

        data_io_name = cfg.data_io.lower()
        data_io_module = import_module(f"deploy.data_io.{data_io_name}.{data_io_name}_io")
        self.data_io = getattr(data_io_module, DATA_IO[data_io_name])()
        self.data_io.set_scale(self.up_scale)
        if self.backend is not None:
            input_shape = self.backend.get_input_shape()
            if len(input_shape) == 4:
                tlen, height, width = input_shape[0], input_shape[-2], input_shape[-1]
            elif len(input_shape) == 5:
                _, tlen, _, height, width = self.backend.get_input_shape()
            if self.tiling is not None:
                if self.tiling == "default":
                    self.tiler = DefaultTiler(
                        self.backend,
                        frame_window_size=tlen,
                        frame_overlap=self.frame_overlap,
                        patch_size=(height, width),
                        patch_overlap=self.patch_overlap,
                        sf=self.up_scale,
                        dtype=self.dtype,
                    )
                elif self.tiling == "simple_image":
                    self.tiler = SimpleImageTiler(
                        self.backend,
                        self.up_scale,
                    )
                else:
                    raise ValueError(f"Unsupported type of tiling {self.tiling_func}")

    # @func_time('one task total')
    def run(self, input_data=None, **kwargs):
        input = self.data_io.preprocess(input_data=input_data)
        if self.tiler is not None:
            output = self.tiler(input, **kwargs)
        elif self.backend is not None:
            output = self.backend.run(input_list=input, **kwargs)
        else:
            print("Warning. Please check if backend is set correctly.")
            # Can be case where task is RGB2BGR or similar.
            output = input
        result = self.data_io.postprocess(output)
        return result
