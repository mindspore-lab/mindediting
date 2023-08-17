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

import glob

import pytest
from common import init_test_environment

init_test_environment()

from mindediting.deploy.utils.config import Config, parse_yaml


def get_configs(config_type: str):
    supported_types = ["train", "val", "all"]
    if config_type not in supported_types:
        raise ValueError(f"The 'config_type' should be one of {supported_types}.")

    if config_type == "all":
        configs = glob.glob("configs/**/*.yaml", recursive=True)
    else:
        configs = [filename for filename in glob.glob(f"configs/**/{config_type}*.yaml", recursive=True)]
    return configs


@pytest.mark.parametrize("config_path", get_configs("all"))
def test_configs_without_extra_keys(config_path):
    cfg, _, _ = parse_yaml(config_path)
    cfg = Config(cfg)

    keys_to_remove_from_config = [
        "enable_modelarts",
        "data_url",
        "train_url",
        "checkpoint_url",
        "output_path",
        "ckpt_save_dir",
        "need_val",
        "callback",
    ]
    for key in cfg.cfg_dict.keys():
        assert not key in keys_to_remove_from_config, f"Key '{key}' should be removed from config in '{config_path}'."

    common_keys = [
        "epoch_size",
        "output_path",
        "need_unzip_in_modelarts",
    ]
    keys_to_remove_from_dict = {
        "model": common_keys
        + [
            "initial_epoch",
            "save_path",
        ],
        "dataset": common_keys,
        "val_dataset": common_keys,
    }
    for dict_name in keys_to_remove_from_dict.keys():
        if dict_name in cfg.cfg_dict.keys():
            for key in cfg.get(dict_name).keys():
                assert (
                    not key in keys_to_remove_from_dict[dict_name]
                ), f"Key '{key}' should be removed from '{dict_name}' in config '{config_path}'."


@pytest.mark.parametrize("config_path", get_configs("train"))
def test_train_configs_train_params(config_path):
    cfg, _, _ = parse_yaml(config_path)
    cfg = Config(cfg)

    assert "train_params" in cfg.cfg_dict, f"Train config {config_path} does not have 'train_params' key."

    keys_for_train_params = [
        "epoch_size",
        "need_val",
    ]
    assert set(keys_for_train_params).issubset(
        set(cfg.train_params.cfg_dict.keys())
    ), f"Train config {config_path}:train_params does not have one of the required keys: {keys_for_train_params}."

    keys_to_remove_from_train_params = [
        "workers_num",
    ]
    for key in cfg.train_params.cfg_dict.keys():
        assert (
            not key in keys_to_remove_from_train_params
        ), f"Key '{key}' should be removed from 'model' in config '{config_path}'."


@pytest.mark.parametrize("config_path", get_configs("val"))
def test_val_configs_without_train_params(config_path):
    cfg, _, _ = parse_yaml(config_path)
    cfg = Config(cfg)

    assert not "train_params" in cfg.cfg_dict, f"Key 'train_params' should be removed from config '{config_path}'."

    keys_to_remove_from_model = [
        "need_val",
    ]
    for key in cfg.model.cfg_dict.keys():
        assert (
            not key in keys_to_remove_from_model
        ), f"Key '{key}' should be removed from 'model' in config '{config_path}'."
