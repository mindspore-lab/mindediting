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

import mindspore.dataset as ds

from .dataset_base import DatasetBase


class DF2K(DatasetBase):
    def __init__(self, dataset_dir, usage=None, scale=4, divisor=1, **kwargs):
        DatasetBase.__init__(self)
        self.dataset_dir = dataset_dir
        self.usage = usage
        self.scale = scale
        self.divisor = divisor
        self._data = None
        assert isinstance(dataset_dir, list) and len(dataset_dir) == 2, (
            f"DF2K dataset consists of two datasets: DIV2K and Flickr2K, should be specified "
            f"exactly two directories as list, but got {dataset_dir}"
        )
        for dataset_dir_ in dataset_dir:
            if self._data is None:
                self._data = self.get_data(dataset_dir_)
            else:
                self._data += self.get_data(dataset_dir_)

    def get_data(self, dataset_dir):
        data = []
        subdir = "train" if self.usage == "train" else "val"
        hr_path = os.path.join(dataset_dir, subdir, "X1")
        lr_path = os.path.join(dataset_dir, subdir, f"X{self.scale}")
        files = sorted([img for img in os.listdir(hr_path) if img.endswith(".png")])
        for img_name in files:
            hr_image = os.path.join(hr_path, img_name)
            lr_image = os.path.join(lr_path, img_name)
            if not os.path.exists(hr_image) or not os.path.exists(lr_image):
                continue
            img_data = [hr_image, lr_image]
            data.append(img_data)
        return data

    def __getitem__(self, index):
        index = index % len(self._data)
        img_hr = self._load_image(self._data[index][0], to_chw=True, to_float=False)
        img_lr = self._load_image(self._data[index][1], to_chw=True, to_float=False)
        h, w = img_hr.shape[1:]
        h1 = h // self.divisor * self.divisor
        w1 = w // self.divisor * self.divisor
        h1l = h1 // self.scale
        w1l = w1 // self.scale
        return img_hr[:, :h1, :w1], img_lr[:, :h1l, :w1l]

    def __len__(self):
        return len(self._data)


def create_df2k_dataset(
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
    source = DF2K(dataset_dir, usage=usage, **kwargs)
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
