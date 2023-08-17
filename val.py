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

import argparse
import json

from mindediting.deploy.utils.config import Config, merge, parse_cli_to_yaml, parse_yaml
from mindediting.models import create_model_by_name
from mindediting.utils.callbacks import val_py_eval
from mindediting.utils.init_utils import init_env


def save_metrics(metrics, path):
    metrics_values = {k: float(v.eval()) for k, v in metrics.items()}
    with open(path, "w") as write_file:
        json.dump(metrics_values, write_file)


def val(cfg, save_metrics_path):
    init_env(cfg)

    cfg.mode = "val"
    net, eval_network = create_model_by_name(model_name=cfg.model.name, cfg=cfg)

    val_py_eval(cfg, net, eval_network, save_metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", type=str, help="Config file path", required=True)
    parser.add_argument(
        "--save_metrics", type=str, help="Path to a file where computed quality metrics will be saved as an json file."
    )
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    final_config = Config(final_config)
    val(final_config, path_args.save_metrics)
