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
import json
import os
import tempfile
from subprocess import run

import yaml

from mindediting.deploy.utils.config import merge_dicts, parse_yaml
from mindediting.utils.tests import get_npu_names
from mindediting.utils.utils import get_rank_mentioned_filename


def save_cfg(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def mpirun(cmd, nproc):
    run(["mpirun", "-n", f"{nproc}", "--allow-run-as-root"] + cmd, check=True)


def update_config_and_eval(config_path, load_path, save_cfg_path, save_metrics, nproc, every_nth):
    cfg = parse_yaml(config_path)[0]
    update_cfg = {
        "model": {
            "load_path": load_path,
        },
    }
    if every_nth:
        if "val_dataset" in cfg:
            update_cfg["val_dataset"] = {"every_nth": every_nth}
        else:
            update_cfg["dataset"] = {"every_nth": every_nth}
    cfg = merge_dicts(cfg, update_cfg)
    save_cfg(cfg, save_cfg_path)

    mpirun(["python3", "val.py", "--config_path", save_cfg_path, "--save_metrics", save_metrics], nproc)


def update_config_and_train(config_path, work_dir, save_cfg_path, nproc, every_nth, epochs, train_update_cfg={}):
    cfg = parse_yaml(config_path)[0]
    cfg = merge_dicts(cfg, train_update_cfg)
    update_cfg = {
        "model": {
            "load_path": "",
        },
        "train_params": {
            "epoch_size": epochs,
            "need_val": False,
            "keep_checkpoint_max": 1,
            "save_epoch_frq": epochs,
            "ckpt_save_dir": work_dir,
        },
    }
    if every_nth:
        update_cfg["dataset"] = {"every_nth": every_nth}
    cfg = merge_dicts(cfg, update_cfg)
    save_cfg(cfg, save_cfg_path)

    mpirun(["python3", "train.py", "--config_path", save_cfg_path], nproc)


def do_val_train_val_test(train_cfg_path, val_cfg_path, train_every_nth, val_every_nth, epochs, train_update_cfg={}):
    nproc = len(get_npu_names())
    work_dir = tempfile.mkdtemp()
    patched_train_cfg_path = os.path.join(work_dir, "train_cfg_patched.yaml")
    val_cfg_before_train_path = os.path.join(work_dir, "val_cfg_before_train.yaml")
    metrics_before_train_path = os.path.join(work_dir, "metrics_before_train.yaml")
    val_cfg_after_train_path = os.path.join(work_dir, "val_cfg_after_train.yaml")
    metrics_after_train_path = os.path.join(work_dir, "metrics_after_train.yaml")

    update_config_and_eval(val_cfg_path, "", val_cfg_before_train_path, metrics_before_train_path, nproc, val_every_nth)
    if nproc > 1:
        metrics_before_train_path = get_rank_mentioned_filename(metrics_before_train_path)
    with open(metrics_before_train_path) as f:
        metrics_before = json.load(f)
    print("metrics_before", metrics_before)

    update_config_and_train(
        train_cfg_path, work_dir, patched_train_cfg_path, nproc, train_every_nth, epochs, train_update_cfg
    )
    checkpoint = glob.glob(work_dir + "/*.ckpt")[0]
    print("checkpoint", checkpoint)

    update_config_and_eval(
        val_cfg_path, checkpoint, val_cfg_after_train_path, metrics_after_train_path, nproc, val_every_nth
    )
    if nproc > 1:
        metrics_after_train_path = get_rank_mentioned_filename(metrics_after_train_path)
    with open(metrics_after_train_path) as f:
        metrics_after = json.load(f)
    print("metrics_after", metrics_after)

    for k in metrics_before:
        print(f"{k}, before={metrics_before[k]}, after={metrics_after[k]}")
        assert metrics_before[k] < metrics_after[k], f"{k}, before={metrics_before[k]}, after={metrics_after[k]}"
