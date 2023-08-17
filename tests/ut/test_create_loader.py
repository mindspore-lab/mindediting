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
from mindediting.dataset.create_loaders import create_loader
from mindediting.dataset.create_transforms import create_transform


@pytest.mark.parametrize("data_root", ["/data/LLVT/NoahTCV/data/CBSD68"])
@pytest.mark.parametrize("split", ["val"])
@pytest.mark.parametrize("batch_size", [1, 64])
def test_create_loader(data_root, split, batch_size):
    dataset_val = create_dataset(name="cbsd68", root=data_root, split="val", shuffle=False, nframes=None)
    val_operations, val_input_columns, val_output_columns = create_transform(
        model_name="noahtcv",
        split="val",
        pipeline=None,
    )
    loader_val = create_loader(
        dataset=dataset_val,
        batch_size=batch_size,
        operations=val_operations,
        input_columns=val_input_columns,
        output_columns=val_output_columns,
        split=split,
    )

    out_batch_size = loader_val.get_batch_size()
    assert out_batch_size == batch_size
