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
import os
import sys
from subprocess import run

from mindspore import context, export

from mindediting.deploy.utils.config import Config, parse_yaml
from mindediting.utils.deploy import create_export_helper
from mindediting.utils.init_utils import init_env

context.set_context(mode=context.GRAPH_MODE)
context.set_context(pynative_synchronize=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Config file path", required=True)
    parser.add_argument("--target_format", choices=["om", "air"], help="Format of output model", required=True)
    parser.add_argument("--shape", nargs="+", type=int, required=True)
    parser.add_argument("--output_name", type=str, help="Name of output file without extension", required=True)
    parser.add_argument("--soc_version", type=str, default="Ascend910A", help="SoC version for ATC")
    parser.add_argument("--fusion_switch_file", type=str, default="", help="Fusion switch file for ATC")
    parser.add_argument("--precision_mode", type=str, default="allow_mix_precision", help="Precision mode for ATC")
    parser.add_argument("--log_level", type=str, default="info", help="Log level for ATC")
    parser.add_argument(
        "--output_type",
        choices=["FP32", "FP16", "UINT8"],
        required=False,
        help="Set data type of output to FP32, FP16 or UINT8 (recommended to use UINT8)",
    )
    return parser.parse_args()


def export_to_om(
    ir_file,
    output_file,
    soc_version="Ascend910A",
    fusion_switch_file="",
    output_type="UINT8",
    log_level="info",
    precision_mode="allow_mix_precision",
):
    if ir_file.endswith(".air"):
        framework = 1
    else:
        raise NotImplemented("Supported only AIR format")
    assert output_type in ["FP32", "FP16", "UINT8"]

    if os.getenv("ASCEND_TOOLKIT_HOME") is None:
        toolkit_path = "/usr/local/Ascend/ascend-toolkit/latest/"
        if not os.path.isdir(toolkit_path):
            print("Please, set ASCEND_TOOLKIT_HOME environment variable. Exit.")
            return
        os.environ["ASCEND_TOOLKIT_HOME"] = toolkit_path
        print(f"Set ASCEND_TOOLKIT_HOME={toolkit_path}")

    my_env = os.environ
    my_env["PATH"] = ":".join(
        [
            "/usr/bin/",
            "/usr/local/bin",
            os.path.join(my_env["ASCEND_TOOLKIT_HOME"], "aarch64-linux/bin/"),
            os.path.join(my_env["ASCEND_TOOLKIT_HOME"], "aarch64-linux/ccec_compiler/bin/"),
        ]
    )

    command = [
        "atc",
        "--model",
        ir_file,
        "--framework",
        str(framework),
        "--output",
        output_file,
        "--precision_mode",
        precision_mode,
        "--log",
        log_level,
        "--soc_version",
        soc_version,
        "--output_type",
        output_type,
    ]
    if fusion_switch_file:
        command.extend(["--fusion_switch_file", fusion_switch_file])

    print(f"Running:\n", " ".join(command), "\n")
    run(command, env=my_env)


def do_export(
    cfg, shape, output_name, output_type, precision_mode, target_format, soc_version, fusion_switch_file, log_level
):

    init_env(cfg)
    export_helper = create_export_helper(getattr(cfg, "export_helper", "default"))
    net, inputs = export_helper(cfg, shape)
    inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
    export(net, *inputs, file_name=output_name, file_format="AIR")

    if target_format == "om":
        export_to_om(
            output_name + ".air",
            output_name,
            soc_version=soc_version,
            fusion_switch_file=fusion_switch_file,
            output_type=output_type,
            log_level=log_level,
            precision_mode=precision_mode,
        )


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(parse_yaml(args.config_path)[0])
    do_export(
        cfg,
        args.shape,
        args.output_name,
        args.output_type,
        args.precision_mode,
        args.target_format,
        args.soc_version,
        args.fusion_switch_file,
        args.log_level,
    )
