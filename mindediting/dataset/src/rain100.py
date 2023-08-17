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


class Rain100(DatasetBase):
    """
    Rain removal dataset
    Incoming Path: dataset_dir = ".../Rain100H"
    The data set directory structure is as follows:
    Rain100H
      ├── rainy
      |     ├── rain-001.png
      |     ├── rain-002.png
      |     ...
      ├── norain-001.png
      ├── norain-002.png
      ...
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        DatasetBase.__init__(self)
        self.dataset_dir = dataset_dir
        self.usage = usage
        self._index = 0
        self._data = self.get_data(self.dataset_dir, self.usage)

    def get_data(self, dataset_dir, usage):
        data = []
        data_dir = os.path.join(dataset_dir, usage)
        if not os.path.exists(data_dir):
            data_dir = dataset_dir
        data_dir_rainy = os.path.join(data_dir, "rainy")
        for rain_img_name in sorted(os.listdir(data_dir_rainy)):
            rain_img_path = os.path.join(data_dir_rainy, rain_img_name)
            norain_img_name = f"no{rain_img_name}"
            norain_img_path = os.path.join(dataset_dir, norain_img_name)
            if not all([os.path.exists(norain_img_path), os.path.exists(rain_img_path)]):
                warnings.warn(f"not path {norain_img_path} or {rain_img_path}")
                continue
            img_data = {
                "norain": norain_img_path,
                "rain": rain_img_path,
                "filename": norain_img_name,
            }
            data.append(img_data)
        return data

    def __getitem__(self, index):
        HR = self._load_image(self._data[index]["norain"])
        LR = self._load_image(self._data[index]["rain"])
        filename = self._data[index]["filename"]
        return HR, LR, index, filename


def create_rain100_dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    **kwargs,
):
    source = Rain100(dataset_dir, usage=usage, **kwargs)
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
