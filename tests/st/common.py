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

import json
import os
import sys
import tempfile
import time
from importlib import import_module
from subprocess import run

import mindspore
import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mindediting.deploy.deploy import deploy
from mindediting.deploy.utils.config import Config, merge_dicts, parse_yaml
from train import train
from val import val

FUSION_SWITCH_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mindediting", "deploy", "utils", "fusion_switch.cfg")
)


def set_seed(seed=1):
    if seed:
        mindspore.set_seed(seed)
        np.random.seed(seed)


def init_test_environment():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    mindspore.set_seed(1)
    np.random.seed(1)

    device_id = int(os.getenv("DEVICE_ID", "0"))
    mindspore.set_context(device_id=device_id)


def do_val_test(model_name, val_cfg_name="val.yaml", eps=3e-3, update_cfg={}, task="", seed=1, estimate_cfg_suffix=""):
    set_seed(seed)
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", model_name, task, val_cfg_name)
    cfg = parse_yaml(cfg_path)[0]
    cfg = merge_dicts(cfg, update_cfg)

    cfg = Config(cfg)

    tmp_metrics_file = tempfile.NamedTemporaryFile(suffix=f"_{model_name}.txt", delete=False)
    actual_metrics_values_file = tmp_metrics_file.name
    val(cfg, actual_metrics_values_file)

    expected_metrics_file_name = (
        f"{model_name}_{task}{estimate_cfg_suffix}.json" if task else f"{model_name}{estimate_cfg_suffix}.json"
    )
    expected_metrics_values_file = os.path.join(
        os.path.dirname(__file__),
        "expected_values",
        "models_val",
        expected_metrics_file_name,
    )

    with open(expected_metrics_values_file) as read_file:
        expected_metrics = json.load(read_file)

    with open(actual_metrics_values_file) as read_file:
        actual_metrics = json.load(read_file)

    for k in expected_metrics:
        assert abs(expected_metrics[k] - actual_metrics[k]) / expected_metrics[k] < eps


def do_train_test(model_name, train_cfg_name="train.yaml", eps=3e-3, update_cfg={}, task="", seed=1):
    set_seed(seed)
    cfg_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "configs",
        model_name,
        task,
        train_cfg_name,
    )
    cfg = parse_yaml(cfg_path)[0]
    cfg = merge_dicts(cfg, update_cfg)
    cfg = Config(cfg)

    train(cfg, profile=False)


def do_export_test(
    model_name,
    shape,
    output_type,
    precision_mode,
    target_format,
    val_cfg_name="val.yaml",
    deploy_cfg_name="task.yaml",
    soc_version="Ascend910A",
    fusion_switch_file="",
    log_level="info",
    eps=3e-3,
    update_cfg={},
    task="",
    validator="tiling",
    seed=1,
    estimate_cfg_suffix="",
):
    set_seed(seed)
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", model_name, task, val_cfg_name)
    cfg = parse_yaml(cfg_path)[0]
    cfg = merge_dicts(cfg, update_cfg)

    temp_folder = tempfile.mkdtemp()
    print("temp_folder", temp_folder)
    output_name = os.path.join(temp_folder, "model")
    temp_cfg_path = os.path.join(temp_folder, "export_config.yaml")
    with open(temp_cfg_path, "w") as f:
        yaml.dump(cfg, f)

    cmd = [
        "python3",
        "export.py",
        "--config_path",
        temp_cfg_path,
        "--shape",
        *[str(s) for s in shape],
        "--output_name",
        output_name,
        "--output_type",
        output_type,
        "--precision_mode",
        precision_mode,
        "--target_format",
        target_format,
        "--soc_version",
        soc_version,
        "--log_level",
        log_level,
    ]
    if fusion_switch_file:
        cmd.extend(["--fusion_switch_file", fusion_switch_file])

    # export is run as a subprocess, since it is unknown how to deallocate memory on Ascend before running evaluation.
    print("RUNNING:", " ".join(cmd))
    env = os.environ.copy()
    run(cmd, check=True, env=env)

    # sleep some time to make sure that Ascend has been released by subprocess
    time.sleep(10)
    scale = cfg["dataset"].get("scale", 1)
    if scale is None:
        scale = 1
    if isinstance(scale, list):
        scale = scale[0]

    if validator == "tiling":
        validator_cfg = {
            "name": "tiling",
            "temporal_overlap": 0,
            "spatial_overlap": 0,
            "temporal_size": shape[-4],
            "spatial_size": shape[-2:],
            "input_tensor_type": "numpy",
            "scale": scale,
        }
    elif validator == "om_default":
        validator_cfg = {
            "name": "om_default",
        }
    elif validator == "ipt_validator":
        validator_cfg = {
            "name": "ipt_validator",
            "temporal_overlap": 0,
            "spatial_overlap": 0,
            "temporal_size": shape[-4],
            "spatial_size": shape[-2:],
            "input_tensor_type": "numpy",
            "scale": scale,
        }
    else:
        raise ValueError(f'Supported only "tiling" and "om_default" validators but got {validator}')

    merge_dicts(
        update_cfg,
        {
            "model": {
                "name": "om",
                "load_path": output_name + ".om",
            },
            "validator": validator_cfg,
        },
    )

    do_val_test(model_name, val_cfg_name, eps, update_cfg, task=task, estimate_cfg_suffix=estimate_cfg_suffix)

    cfg_deploy_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", model_name, task, deploy_cfg_name)
    cfg_deploy = parse_yaml(cfg_deploy_path)[0]
    out_dir = os.path.join(temp_folder, "deploy_out")
    cfg_deploy = merge_dicts(
        cfg_deploy,
        {
            "output_file": out_dir,
        },
    )
    tasks = cfg_deploy.get("tasks")
    cfg_task = merge_dicts(
        tasks[0],
        {
            "model_file": output_name + ".om",
        },
    )
    cfg_deploy = merge_dicts(
        cfg_deploy,
        {
            "tasks": [cfg_task],
        },
    )

    deploy(cfg_deploy)

    assert len(os.listdir(out_dir)) > 0
