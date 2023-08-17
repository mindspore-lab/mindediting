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

import mindspore.dataset as ds

from .df2k import DF2K


class DIV2K(DF2K):
    def __init__(self, dataset_dir, usage=None, scale=4, divisor=8, **kwargs):
        self.dataset_dir = dataset_dir
        self.usage = usage
        self.scale = scale
        self.divisor = divisor
        self._data = None
        self._data = self.get_data(dataset_dir)


def create_div2k_dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    every_nth=1,
    **kwargs,
):
    source = DIV2K(dataset_dir, usage=usage, **kwargs)
    if every_nth > 1:
        source._data = source._data[::every_nth]
    dataset = ds.GeneratorDataset(
        source=source,
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        column_names=["HR", "LR"],
        shuffle=shuffle,
    )
    return dataset
