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


class RedsDeblur(DatasetBase):
    """
    REDS deblur dataset.
    Incoming Path: dataset_dir = ".../REDS"
    The data set directory structure is as follows:
    REDS
      ├── train
      |     └── train_blur
      |         ├── 000
      |         ├── 001
      |         ...
      |     └── train_sharp
      |         ├── 000
      |         ├── 001
      |         ...
      ├── val
      |     └── val_blur
      |         ├── 000
      |         ├── 001
      |         ...
      |     └── val_sharp
      |         ├── 000
      |         ├── 001
      |         ...
      ├── REDS_train.txt
      └── REDS_test.txt
    """

    def __init__(self, dataset_dir, usage=None, video_mode=True, transforms=None, **kwargs):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.video_mode = video_mode
        self.usage = usage
        if self.usage not in ["train", "val"]:
            raise ValueError(f'Incorrect value of parameter "usage": {usage}. Valid values: train, val.')
        gt_subdir = f"{self.usage}/{self.usage}_sharp/"
        self.dataroot_gt = os.path.join(dataset_dir, gt_subdir)
        lq_subdir = f"{self.usage}/{self.usage}_blur/"
        self.dataroot_lq = os.path.join(dataset_dir, lq_subdir)
        self.sep_file = "REDS_train.txt" if self.usage == "train" else "REDS_test.txt"
        self._data = self.get_data()

    def get_data(self):
        data = []
        with open(os.path.join(self.dataset_dir, self.sep_file), "r") as read_file:
            for data_row in read_file:
                video_id = data_row.split()[0].strip()
                hr_dir_path = os.path.join(self.dataroot_gt, video_id)
                per_video_frames = []
                for img_name in sorted(os.listdir(hr_dir_path)):
                    hr_img_path = os.path.join(hr_dir_path, img_name)
                    lr_img_path = os.path.join(self.dataroot_lq, video_id, img_name)
                    if not all([os.path.exists(hr_img_path), os.path.exists(lr_img_path)]):
                        warnings.warn(f"not path {hr_img_path} or {lr_img_path}")
                        continue
                    img_data = {
                        "HR": hr_img_path,
                        "LR": lr_img_path,
                        "filename": img_name,
                    }
                    per_video_frames.append(img_data)
                if self.video_mode:
                    data.append(per_video_frames)
                else:
                    data.extend(per_video_frames)
        assert len(data) > 0, "No data found. Check paths and annotation file."
        return data

    def __getitem__(self, index):
        if self.video_mode:
            HRs, LRs, filenames = [], [], []
            for img_data in self._data[index]:
                HRs.append(img_data["HR"])
                LRs.append(img_data["LR"])
                filenames.append(img_data["filename"])
            if self.transforms is not None:
                out = self.transforms(HRs, LRs, index, filenames)
                return out
            return HRs, LRs, index, filenames
        else:
            HR = self._load_image(self._data[index]["HR"])
            LR = self._load_image(self._data[index]["LR"])
            file_name = os.path.basename(self._data[index]["filename"])
            if self.transforms is not None:
                return self.transforms(HR, LR, index, file_name)
            return HR, LR, index, file_name

    def __len__(self):
        return len(self._data)


def create_reds_deblur_dataset(
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
    source = RedsDeblur(dataset_dir, usage=usage, **kwargs)
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
