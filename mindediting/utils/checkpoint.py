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


import mindspore as ms


def load_param_into_net(model, checkpoint_path, strict_load=True, ignore_list=None, verbose=False):
    loaded_param_dict = ms.load_checkpoint(checkpoint_path)

    ms.load_param_into_net(model, loaded_param_dict, strict_load=True)
    model_params = model.parameters_and_names()

    loaded_param_names = [k for k in loaded_param_dict.keys()]
    models_param_names = [param.name for _, param in model_params]

    unused_loaded_params = [n for n in loaded_param_names if n not in models_param_names]
    if ignore_list is not None and len(ignore_list) > 0:
        unused_loaded_params = [n for n in unused_loaded_params if n not in ignore_list]

    unused_model_params = [n for n in models_param_names if n not in loaded_param_names]
    if ignore_list is not None and len(ignore_list) > 0:
        unused_model_params = [n for n in unused_model_params if n not in ignore_list]

    if verbose and len(unused_loaded_params) > 0:
        print(f"\nFound {len(unused_loaded_params)} unused loaded params:")
        for name in unused_loaded_params:
            shape = loaded_param_dict[name].shape
            print(f"   * {name}: {shape}")

    if verbose and len(unused_model_params) > 0:
        print(f"\nFound {len(unused_model_params)} unused model params:")
        for name in unused_model_params:
            print(f"   * {name}")

    if strict_load and (len(unused_loaded_params) > 0 or len(unused_model_params) > 0):
        raise RuntimeError("Loaded checkpoint doesn't match the model parameters.")


def load_vrt(model, checkpoint_path):
    param_dict = ms.load_checkpoint(checkpoint_path)
    for name, parameter in model.parameters_and_names():
        if name.endswith(".attn.relative_position_bias"):
            if name not in param_dict:
                param_dict[name] = parameter
    ms.load_param_into_net(model, param_dict)
    model.relative_position_table_to_bias()
