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

from subprocess import run


def check_hbm_available(npu_name):
    cmd = "npu-smi info -t usages -i " + f"{npu_name}" + " | grep \"HBM Usage\" | awk '{ print $5}' "
    cp = run(cmd, shell=True, capture_output=True, text=True)
    return cp.stdout.rstrip() == "0"


def get_npu_names():
    cmd = "npu-smi info -l | grep ID | awk '{ print $4}' "
    cp = run(cmd, shell=True, capture_output=True, text=True)
    return [int(id) for id in cp.stdout.splitlines()]
