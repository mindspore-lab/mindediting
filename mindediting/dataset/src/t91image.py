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

import h5py
import mindspore.dataset as ds
import numpy as np


class H5T91Image(object):
    def __init__(self, dataset_dir, usage=None, **kwargs):
        self.h5_file = dataset_dir
        self.f = h5py.File(self.h5_file, "r")

    def __getitem__(self, index):
        HR = np.expand_dims(self.f["hr"][index], 0)
        LR = np.expand_dims(self.f["lr"][index], 0)
        return HR, LR, index, self.h5_file

    def __len__(self):
        return len(self.f["lr"])

    def __del__(self):
        self.f.close()


def create_h5_t91image_dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    **kwargs
):
    if h5py.is_hdf5(dataset_dir):
        source = H5T91Image(dataset_dir, usage=usage, **kwargs)
    else:
        raise LookupError("This dataset format is not supported currently.")
    dataset = ds.GeneratorDataset(
        source=source,
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        column_names=["HR", "LR", "idx", "filename"],
        shuffle=shuffle,
    )
    return dataset
