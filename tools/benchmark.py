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
import datetime
import os.path as osp
import time

import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Profiler
from tqdm import trange

from mindediting.deploy.utils.config import Config, merge, parse_cli_to_yaml, parse_yaml
from mindediting.models import create_model_by_name
from mindediting.utils.init_utils import init_env


def _prepare_input(shape, dtype, is_om_model):
    data = np.random.randn(*shape)

    if not is_om_model:
        data = ms.Tensor(data, dtype)

    return data


def _create_profiler(output_path):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = osp.join(output_path, now)
    profiler = Profiler(start_profile=False, output_path=output_path)

    return profiler


def benchmark(cfg, shape, ntimes, dtype, profile, profiler_path="../profile"):
    assert ntimes > 1

    init_env(cfg)

    cfg.mode = "val"
    net, eval_network = create_model_by_name(model_name=cfg.model.name, cfg=cfg)
    net = eval_network or net

    is_om_model = not isinstance(net, nn.Cell)

    if not is_om_model and dtype == ms.float16:
        net = net.to_float(dtype)

    if profile:
        profiler = _create_profiler(profiler_path)
        print("Created profiler.")

    times = []
    for iteration in trange(ntimes, leave=False):
        net_input = _prepare_input(shape, dtype, is_om_model)

        if profile and iteration == 1:
            profiler.start()
            print("Profiler has been started.")

        start_time = time.time()
        _ = net(net_input)
        end_time = time.time()

        times.append(end_time - start_time)

    times = times[1:]

    if profile:
        profiler.stop()
        print("Profiler has been stopped.")

        profiler.analyse()
        print("Profiler has finished the analysis.")

    print("Execution time:")
    print(f"   * min: {1e3 * np.min(times)} ms")
    print(f"   * median: {1e3 * np.median(times)} ms")
    print(f"   * max: {1e3 * np.max(times)} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", type=str, help="Config file path", required=True)
    parser.add_argument("--shape", type=int, nargs="+", help="Network input shape")
    parser.add_argument("--ntimes", type=int, default=11, help="Number of network runs")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16"], help="Target precision")
    parser.add_argument("--profile", action="store_true", help="Enables profile callback.")
    path_args, _ = parser.parse_known_args()

    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    final_config = Config(final_config)

    if path_args.dtype == "fp32":
        dtype = ms.float32
    elif path_args.dtype == "fp16":
        dtype = ms.float16
    else:
        raise ValueError(f"Invalid dtype: {path_args.dtype}")
    print(f"Run on {dtype} precision.")

    benchmark(final_config, path_args.shape, path_args.ntimes, dtype, path_args.profile)
