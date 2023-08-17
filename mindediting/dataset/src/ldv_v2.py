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
import os.path as osp
import random
import warnings
from time import time

import mindspore.dataset as ds
import numpy as np

from .dataset_base import DatasetBase


class LDV_v2(DatasetBase):
    def __init__(self, dataset_dir, usage=None, return_paths=False, transforms=None, **kwargs):
        DatasetBase.__init__(self)
        self.data_root = kwargs.get("input_path", "")
        self.gt_subdir = kwargs.get("gt_subdir", "train_gt")
        self.lr_subdir = kwargs.get("lr_subdir", "train_lq")
        self.num_input_frames = kwargs.get("nframes", 0)
        self.usage = usage
        self.return_paths = return_paths
        self.transforms = transforms
        self._data = self.get_data()

    def get_data(self):
        data = []

        sequences = sorted(glob.glob(osp.join(self.data_root, self.lr_subdir, "*")))
        for sequence in sequences:
            per_video_frames = []
            for img_name in sorted(glob.glob(osp.join(sequence, "*"))):
                lr_img_path = os.path.join(sequence, img_name)
                path = os.path.join(self.data_root, self.lr_subdir)
                key = img_name.replace(f"{path}/", "")
                hr_img_path = os.path.join(self.data_root, self.gt_subdir, key)
                img_data = {
                    "HR": hr_img_path,
                    "LR": lr_img_path,
                    "filename": img_name,
                }
                per_video_frames.append(img_data)
            data.append(per_video_frames)
        assert len(data) > 0
        return data

    def __getitem__(self, index):
        begin, end = 0, len(self._data[index])
        if self.num_input_frames > 0:
            begin = random.randint(0, len(self._data[index]) - self.num_input_frames)
            end = begin + self.num_input_frames
        HRs = [img_data["HR"] for img_data in self._data[index][begin:end]]
        LRs = [img_data["LR"] for img_data in self._data[index][begin:end]]
        filenames = [img_data["filename"] for img_data in self._data[index][begin:end]]

        if not self.return_paths:
            HRs = list(self._load_image(frame_path) for frame_path in HRs)
            LRs = list(self._load_image(frame_path) for frame_path in LRs)
        if self.transforms is not None:
            return self.transforms(HRs, LRs, index, filenames)
        return HRs, LRs, index, filenames

    def __len__(self):
        return len(self._data)


def create_ldv_video_enhancement_dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    column_names=["HR", "LR", "idx", "filename"],
    max_rowsize=32,
    **kwargs,
):
    source = LDV_v2(dataset_dir, usage=usage, **kwargs)
    dataset = ds.GeneratorDataset(
        source=source,
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        max_rowsize=max_rowsize,
        num_shards=num_shards,
        shard_id=shard_id,
        column_names=column_names,
        shuffle=shuffle,
    )
    return dataset


if __name__ == "__main__":
    source = LDV_v2(dataset_dir="/data/LLVT/VRT/data/", usage="train")
    dataset = ds.GeneratorDataset(source=source, column_names=["HR", "LR", "idx", "filename"], shuffle=False)
    print(dataset)
