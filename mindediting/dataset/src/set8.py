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
import warnings

import mindspore.dataset as ds

from .dataset_base import DatasetBase


class Set8(DatasetBase):
    """
    Set8 dataset.
    Incoming Path: dataset_dir = ".../Set8"
    The data set directory structure is as follows:
    Set8
    ├── hypersmooth
    ├── motorbike
    ├── park_joy
    ├── rafting
    ├── snowboard
    ├── sunflower
    ├── touchdown
    └── tractor
    """

    def __init__(self, dataset_dir, usage=None, return_paths=False, transforms=None, **kwargs):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.return_paths = return_paths
        self.transforms = transforms
        self._data = self.get_data()

    def get_data(self):
        data = []

        sequences = sorted(glob.glob(osp.join(self.dataset_dir, "*")))

        for img_dir_name in sequences:
            img_dir_path = os.path.join(self.dataset_dir, img_dir_name)
            video_frames_names = list(sorted(os.listdir(img_dir_path)))
            video_data = []
            for img_name in video_frames_names:
                img_origin_path = os.path.join(img_dir_path, img_name)
                if not os.path.exists(img_origin_path):
                    warnings.warn(f"Invalid image path {img_origin_path}")
                    continue
                img_data = {"origin": img_origin_path, "file_name": img_name}
                video_data.append(img_data)

            data.append(video_data)

        return data

    def __getitem__(self, index):
        video_data = self._data[index]
        frames = [v["origin"] for v in video_data]
        annotation = ["none" for v in video_data]
        file_names = [v["file_name"] for v in video_data]
        if not self.return_paths:
            frames = list(self._load_image(frame_path) for frame_path in frames)
        if self.transforms is not None:
            out = self.transforms(frames, annotation, index, file_names)
            return out
        return frames, annotation, index, file_names


def create_set8_dataset(
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
    source = Set8(dataset_dir, usage=usage, **kwargs)
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
