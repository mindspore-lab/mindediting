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
import pytest
from common import init_test_environment

init_test_environment()

from mindediting.dataset.create_datasets import create_dataset


@pytest.mark.parametrize("name", ["sidd"])
@pytest.mark.parametrize("root", ["/data/LLVT/Tunable_Conv/data/SIDD/"])
@pytest.mark.parametrize("split", ["val"])
def test_create_dataset(name, root, split):
    dataset = create_dataset(name=name, root=root, split=split)
    assert dataset.source_len > 0
