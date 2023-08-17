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
import warnings

import mindspore.dataset as ds

from .dataset_base import DatasetBase


class Cbsd68(DatasetBase):
    """
    Denoising dataset
    Incoming Path: dataset_dir = ".../CBSD68"
    The data set directory structure is as follows:
    CBSD68
      ├── noisy5
      ├── noisy10
      ├── noisy15
      ├── noisy25
      ├── noisy35
      ├── noisy50
      ├── original
      └── original_png
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        DatasetBase.__init__(self)
        self.dataset_dir = dataset_dir
        self.usage = usage
        self.noise_folder = "noisy" + str(kwargs.get("noise_level", 50))
        self._index = 0
        self._data = self.get_data(self.dataset_dir, self.usage)

    def get_data(self, dataset_dir, usage):
        data = []
        data_dir = dataset_dir
        if not os.path.exists(data_dir):
            data_dir = dataset_dir
        for img_name in sorted(os.listdir(os.path.join(data_dir, "original_png"))):
            img_path_original = os.path.join(data_dir, "original_png", img_name)
            img_path_noisy = os.path.join(data_dir, self.noise_folder, img_name)
            if not all([os.path.exists(img_path_original), os.path.exists(img_path_noisy)]):
                warnings.warn(f"not path {img_path_original} or {img_path_noisy}")
                continue
            img_data = {
                "HR": img_path_original,
                "LR": img_path_noisy,
                "filename": img_name,
            }
            data.append(img_data)
        return data

    def __getitem__(self, index):
        HR = self._load_image(self._data[index]["HR"])
        LR = self._load_image(self._data[index]["LR"])
        filename = self._data[index]["filename"]
        return HR, LR, index, filename


def create_cbsd68_dataset(
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
    source = Cbsd68(dataset_dir, usage=usage, **kwargs)
    if every_nth > 1:
        source._data = source._data[::every_nth]
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
