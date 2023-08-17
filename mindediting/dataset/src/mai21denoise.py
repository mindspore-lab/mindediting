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


class MAI21Denoise(DatasetBase):
    """
    Denoising dataset

    Training dataset:
        Incoming Path: dataset_dir = ".../MAI2021_denoising_train_jpeg/train"
        The data set directory structure is as follows:
        train
        ├── denoised
        └── original
    Validating dataset:
        Incoming Path: dataset_dir = ".../MAI2021_denoising_valid_cropped_noi"
        The data set directory structure is as follows:
        MAI2021_denoising_valid_cropped_noi
        ├── 12.png
        ├── 20.png
        ...
    Test Dataset:
        Incoming Path: dataset_dir = ".../MAI2021_denoising_test_cropped_nois"
        The data set directory structure is as follows:
        MAI2021_denoising_test_cropped_nois
        ├── 7.png
        ├── 31.png
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
        data_dir = dataset_dir
        if not os.path.exists(data_dir):
            data_dir = dataset_dir
        data_dir_original = os.path.join(data_dir, "noisy")
        for img_name in sorted(os.listdir(data_dir_original)):
            img_original_path = os.path.join(data_dir_original, img_name)
            img_denoised_path = os.path.join(data_dir, "sharp", img_name)
            if not all([os.path.exists(img_original_path), os.path.exists(img_denoised_path)]):
                warnings.warn(f"not path {img_original_path} or {img_denoised_path}")
                continue
            img_data = {
                "noisy": img_original_path,
                "sharp": img_denoised_path,
                "filename": img_name,
            }
            data.append(img_data)
        return data

    def __getitem__(self, index):
        LR = self._load_image(self._data[index]["noisy"])
        HR = self._load_image(self._data[index]["sharp"])
        filename = self._data[index]["filename"]
        return HR, LR, index, filename


def create_mai21denoise_dataset(
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
    source = MAI21Denoise(dataset_dir, usage=usage, **kwargs)
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
