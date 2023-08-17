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


class SIDDDataset(DatasetBase):
    """

    The dataset contains 160 pairs of noisy/ground-truth images taken by
    the following smartphones under different lightening conditions:
        GP: Google Pixel
        IP: iPhone 7
        S6: Samsung Galaxy S6 Edge
        N6: Motorola Nexus 6
        G4: LG G4

    NOTE: train dataset is not currently available, only use this with usage='valid'

    └── SIDD dataset_dir
         ├── train
         |    ├── 0001_001_S6_00100_00060_3200_L
         |    |    ├── 0001_GT_SRGB_010_0000.png
         |    |    ├── 0001_GT_SRGB_010_0001.png
         |    |    ├── ...
         |    ├── 0002_001_S6_00100_00020_3200_N
         |    |    ├── 0002_GT_SRGB_010_0000.png
         |    |    ├── 0002_GT_SRGB_010_0001.png
         |    |    ├── ...
         |    ├── ...
         |
         ├── val
         |    ├── 0009_001_S6_00800_00350_3200_L
         |    |    ├── 0009_GT_SRGB_010_0000.png
         |    |    ├── 0009_GT_SRGB_010_0001.png
         |    |    ├── ...
         |    ├── 0021_001_GP_10000_05000_5500_N
         |    |    ├── 0021_GT_SRGB_010_0000.png
         |    |    ├── 0021_GT_SRGB_010_0001.png
         |    |    ├── ...
         |    ├── ...
         |

    """

    def __init__(self, dataset_dir, usage="train", **kwargs):
        super(SIDDDataset, self).__init__()

        assert usage in ["train", "val"]

        self.dataset_dir = dataset_dir
        self.training = usage == "train"
        self.usage = usage
        if self.training:
            raise NotImplementedError()
        self._data = self.get_data(self.dataset_dir, self.usage)

    def get_data(self, dataset_dir, usage):
        hr_fnames = sorted(glob.glob(os.path.join(dataset_dir, usage, "**", "*_GT_SRGB_*.png"), recursive=True))

        fnames = []
        for hr_fname in hr_fnames:
            fnames.append(
                {
                    "hr_fname": hr_fname,
                    "lr_fname": hr_fname.replace("_GT_SRGB_", "_NOISY_SRGB_"),
                }
            )
        return fnames

    def __getitem__(self, index: int):

        HR = self._load_image(self._data[index]["hr_fname"])
        LR = self._load_image(self._data[index]["lr_fname"])

        return HR, LR


def create_sidd_dataset(
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
        source = SIDDDataset(dataset_dir, usage=usage, **kwargs)
        column_names = ["HR", "LR"]
    else:
        source = SIDDDataset(dataset_dir, usage=usage, **kwargs)
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
