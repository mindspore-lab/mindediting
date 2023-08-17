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


class DeepVideoDeblurring(DatasetBase):
    """
    Deblurring dataset
    Incoming Path: dataset_dir = ".../DeepVideoDeblurring_Dataset/qualitative_datasets"
    The data set directory structure is as follows:
    qualitative_datasets
      ├── alley
      |     ├── ae_stabilize
      |     └── input
      ├── anita
      |     ├── ae_stabilize
      |     └── input
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
        for class_ in os.listdir(data_dir):
            if class_ == ".DS_Store":
                continue
            blur_data_input_path = os.path.join(data_dir, class_, "input")
            for blur_img_name in sorted(os.listdir(blur_data_input_path)):
                if blur_img_name == ".DS_Store":
                    continue
                img_blur_path = os.path.join(blur_data_input_path, blur_img_name)
                img_sharp_path = os.path.join(data_dir, class_, "GT", blur_img_name)
                if not os.path.exists(img_sharp_path):
                    img_sharp_path = os.path.join(data_dir, class_, "ae_stabilize", blur_img_name)
                if not all([os.path.exists(img_sharp_path), os.path.exists(img_blur_path)]):
                    warnings.warn(f"not path {img_sharp_path} or {img_blur_path}")
                    continue
                img_data = {
                    "sharp": img_sharp_path,
                    "blur": img_blur_path,
                    "file_name": blur_img_name,
                }
                data.append(img_data)
        return data

    def __getitem__(self, index):
        HR = self._load_image(self._data[index]["sharp"])
        LR = self._load_image(self._data[index]["blur"])
        file_name = self._data[index]["file_name"]
        return HR, LR, index, file_name


def create_dvd_dataset(
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
    source = DeepVideoDeblurring(dataset_dir, usage=usage, **kwargs)
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
