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


class Gopro(DatasetBase):
    """
    Deblurring dataset
    Incoming Path: dataset_dir = "../GOPRO_Large"
    The data set directory structure is as follows:
    GOPRO_Large
    ├── train
    |   ├── GOPR0372_07_00
    |   |   ├── blur
    |   |   ├── blur_gamma
    |   |   └── sharp
    |   ├── GOPR0372_07_01
    |   |   ├── blur
    |   |   ├── blur_gamma
    |   |   └── sharp
    |   ...
    └── test
        ├── GOPR0384_11_00
        |   ├── blur
        |   ├── blur_gamma
        |   └── sharp
        ├── GOPR0384_11_05
        |    ├── blur
        |    ├── blur_gamma
        |    └── sharp
        ...
    """

    def __init__(self, dataset_dir, usage=None, video_mode=True, transforms=None, **kwargs):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.video_mode = video_mode
        if usage not in ["train", "val"]:
            raise ValueError(f'Incorrect value of parameter "usage": {usage}. Valid values: train, val.')
        self.usage = "test" if usage == "val" else usage
        self.transforms = transforms
        self._data = self.get_data()

    def get_data(self):
        data = []
        data_dir = os.path.join(self.dataset_dir, self.usage)
        if not os.path.exists(data_dir):
            data_dir = self.dataset_dir
        data_dir_sharp = os.path.join(data_dir, "sharp")
        data_dir_sharp_list = []
        if os.path.exists(data_dir_sharp):
            data_dir_sharp_list.append(data_dir_sharp)
        else:
            for gopro_name in os.listdir(data_dir):
                data_dir_sharp_list.append(os.path.join(data_dir, gopro_name, "sharp"))
        for dds in data_dir_sharp_list:
            image_list = os.listdir(dds)
            image_list.sort()
            video_frames = []
            for img_name in image_list:
                img_sharp_path = os.path.join(dds, img_name)
                img_blur_path = img_sharp_path.replace("sharp", "blur")
                if not all([os.path.exists(img_sharp_path), os.path.exists(img_blur_path)]):
                    warnings.warn(f"not path {img_sharp_path} or {img_blur_path}")
                    continue
                img_data = {
                    "sharp": img_sharp_path,
                    "blur": img_blur_path,
                    "filename": img_name,
                }
                video_frames.append(img_data)
            if self.video_mode:
                data.append(video_frames)
            else:
                data.extend(video_frames)
        return data

    def __getitem__(self, index):
        if self.video_mode:
            HRs, LRs, filenames = [], [], []
            for img_data in self._data[index]:
                HRs.append(img_data["sharp"])
                LRs.append(img_data["blur"])
                filenames.append(img_data["filename"])
            return HRs, LRs, index, filenames
        else:
            HR = self._load_image(self._data[index]["sharp"])
            LR = self._load_image(self._data[index]["blur"])
            file_name = os.path.basename(self._data[index]["filename"])
            if self.transforms is not None:
                return self.transforms(HR, LR, index, file_name)
            return HR, LR, index, file_name


def create_gopro_dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    column_names=["HR", "LR", "idx", "filename"],
    every_nth=1,
    **kwargs,
):
    source = Gopro(dataset_dir, usage=usage, **kwargs)
    if every_nth > 1:
        source._data = source._data[::every_nth]
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
