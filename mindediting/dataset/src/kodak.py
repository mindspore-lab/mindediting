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
import os

import mindspore.dataset as ds

from mindediting.dataset.src.dataset_base import DatasetBase


class KodakDataset(DatasetBase):
    """
    Expects dataset organized as follows.

    └── Kodak dataset_dir
         ├── HR
         |    ├── <filename>.png
         |    ├── ...
         |
         ├── LR_<dataset_compression>
         |    ├── X<scaling_factor>
         |    |    ├── <filename>x<scaling_factor>.png
         |    |    ├── ...
         |    ├── ...
         |

    """

    def __init__(self, dataset_dir, usage="train", **kwargs):
        super(KodakDataset, self).__init__()

        assert usage in ["train", "val"]

        self.dataset_dir = dataset_dir
        self.training = usage == "train"
        self.usage = usage
        if self.training:
            raise NotImplementedError()
        self._data = self.get_data(self.dataset_dir, self.usage)

    def get_data(self, dataset_dir, usage):
        hr_fnames = sorted(glob.glob(os.path.join(dataset_dir, "HR", "*png")))
        lr_fnames = sorted(glob.glob(os.path.join(dataset_dir, "LR_bicubic", "X4", "*png")))
        assert len(hr_fnames) == len(lr_fnames)
        fnames = []
        for hr_fname, lr_fname in zip(hr_fnames, lr_fnames):
            fnames.append({"HR": hr_fname, "LR": lr_fname})
        return fnames

    def __getitem__(self, index: int):
        HR = self._load_image(self._data[index]["HR"])
        LR = self._load_image(self._data[index]["LR"])

        return HR, LR


def create_kodak_dataset(
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
    if usage == "train":
        source = KodakDataset(dataset_dir, usage=usage, **kwargs)
        column_names = ["HR", "LR"]
    else:
        source = KodakDataset(dataset_dir, usage=usage, **kwargs)
        column_names = ["HR", "LR"]
    dataset = ds.GeneratorDataset(
        source=source,
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        column_names=column_names,
        shuffle=shuffle,
    )
    return dataset
