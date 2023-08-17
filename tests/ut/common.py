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
import sys

import mindspore
import numpy as np
from mindspore import context


def init_test_environment():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

    mindspore.set_seed(1)
    np.random.seed(1)

    device_id = int(os.getenv("DEVICE_ID", "0"))
    mindspore.set_context(device_id=device_id)
    mindspore.context.set_context(mode=context.GRAPH_MODE)
