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
import random
import warnings

import mindspore.dataset as ds

from .dataset_base import DatasetBase


class Vid4(DatasetBase):
    """
    Deblurring dataset
    Incoming Path: dataset_dir = ".../Vid4"
    The data set directory structure is as follows:
    Vid4
      ├── BDx4
      └── GT
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        DatasetBase.__init__(self)
        self.dataset_dir = dataset_dir
        self.usage = usage
        self._index = 0
        self.one_video_frames = 32
        self.num_input_frames = kwargs.get("nframes", 0)
        self.interval = kwargs.get("interval", 1)
        self.random_reverse = kwargs.get("random_reverse", True if self.usage == "train" else False)
        self._data = self.get_data(self.dataset_dir, self.usage)

    def get_data(self, dataset_dir, usage):
        data = []
        data_dir = os.path.join(dataset_dir, usage)
        if not os.path.exists(data_dir):
            data_dir = dataset_dir
        hr_data_path = os.path.join(data_dir, "GT")
        for class_ in os.listdir(hr_data_path):
            data_dir_sharp = os.path.join(hr_data_path, class_)
            img_name_list = sorted(os.listdir(data_dir_sharp))[: self.one_video_frames]
            per_video_frames = []
            for img_name in img_name_list:
                hr_img_path = os.path.join(data_dir_sharp, img_name)
                lr_img_path = os.path.join(data_dir, "BDx4", class_, img_name)
                if not all([os.path.exists(hr_img_path), os.path.exists(lr_img_path)]):
                    warnings.warn(f"not path {hr_img_path} or {lr_img_path}")
                    continue
                img_data = {
                    "HR": hr_img_path,
                    "LR": lr_img_path,
                    "filename": img_name,
                }
                per_video_frames.append(img_data)
            data.append(per_video_frames)
        return data

    def __getitem__(self, index):
        begin, end = 0, len(self._data[index])
        if self.num_input_frames > 0:
            begin = random.randint(0, len(self._data[index]) - self.num_input_frames)
            end = begin + self.num_input_frames

        HRs = [self._load_image(img_data["HR"]) for img_data in self._data[index][begin:end]]
        LRs = [self._load_image(img_data["LR"]) for img_data in self._data[index][begin:end]]
        filenames = [img_data["filename"] for img_data in self._data[index][begin:end]]

        return HRs, LRs, index, filenames

    def __len__(self):
        return len(self._data)


def create_vid4_dataset(
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
    source = Vid4(dataset_dir, usage=usage, **kwargs)
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
