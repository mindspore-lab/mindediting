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

import cv2
import lmdb
import mindspore.dataset as ds
import numpy as np

from .dataset_base import DatasetBase


class Reds(DatasetBase):
    """
    4X Super-Resolution dataset
    Incoming Path: dataset_dir = ".../REDS"
    The data set directory structure is as follows:
    REDS
      ├── trainval_sharp_bicubic
      |     └── X4
      |         ├── 000
      |         ├── 001
      |         ...
      ├── trainval_sharp_HR
      |     ├── 000
      |     ├── 001
      |     ...
      ├── REDS4.txt
      └── REDS266.txt
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        DatasetBase.__init__(self)
        self.dataset_dir = dataset_dir
        self.dataroot_gt = os.path.join(dataset_dir, kwargs.get("gt_subdir", "trainval_sharp_HR"))
        self.dataroot_lq = os.path.join(dataset_dir, kwargs.get("lq_subdir", f"trainval_sharp_bicubic{os.sep}X4"))
        self.usage = usage
        self.sep_file = (
            kwargs.get("train_annotation", "REDS266.txt")
            if usage == "train"
            else kwargs.get("test_annotation", "REDS4.txt")
        )
        self.num_input_frames = kwargs.get("nframes", 0)
        self.random_subsequence = kwargs.get("random_subsequence", True)
        self._data = self.get_data(self.dataset_dir)

    def get_data(self, dataset_dir):
        data = []
        with open(os.path.join(dataset_dir, self.sep_file), "r") as read_file:
            for video_id in read_file:
                video_id = video_id.strip()
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
                data.append(per_video_frames)
        return data

    def __getitem__(self, index):
        begin, end = 0, len(self._data[index])
        if self.num_input_frames > 0:
            if self.random_subsequence:
                begin = random.randint(0, len(self._data[index]) - self.num_input_frames)
            end = begin + self.num_input_frames

        HRs = [self._load_image(img_data["HR"]) for img_data in self._data[index][begin:end]]
        LRs = [self._load_image(img_data["LR"]) for img_data in self._data[index][begin:end]]
        filenames = [img_data["filename"] for img_data in self._data[index][begin:end]]

        return HRs, LRs, index, filenames

    def __len__(self):
        return len(self._data)


class LmdbReds(DatasetBase):
    def __init__(self, dataset_dir, usage=None, **kwargs):
        DatasetBase.__init__(self)
        self.dataroot_gt = kwargs.get("dataroot_gt", os.path.join(dataset_dir, "train_sharp_with_val.lmdb"))
        self.dataroot_lq = kwargs.get("dataroot_lq", os.path.join(dataset_dir, "train_sharp_bicubic_with_val.lmdb"))
        self.meta_info_file = kwargs.get("meta_info_file", os.path.join(dataset_dir, "meta_info_REDS_GT.txt"))
        self.num_input_frames = kwargs.get("nframes", 0)
        self.each_frame_sum = 100
        if self.is_lmdb_dataset():
            self._data = self.get_data()
        else:
            self._data = []

    def is_lmdb_dataset(self):
        return all(
            [
                os.path.exists(self.dataroot_gt),
                self.dataroot_gt.endswith(".lmdb"),
                os.path.exists(self.dataroot_lq),
                self.dataroot_lq.endswith(".lmdb"),
            ]
        )

    def get_folders(self):
        folders = []
        with open(self.meta_info_file, "r") as f:
            for i in f.readlines():
                if i:
                    folder, each_frame_sum, _, start_frame = i.split()
                    self.each_frame_sum = int(each_frame_sum)
                    folders.append(folder)
        return folders

    def get_lmdb_data(self, lmdb_path):
        folders = self.get_folders()
        data = {}
        env = lmdb.open(lmdb_path)
        txn = env.begin()
        max_frame = (
            self.each_frame_sum // self.num_input_frames * self.num_input_frames
            if self.num_input_frames > 0
            else self.each_frame_sum
        )
        count = 0
        for key, value in txn.cursor():
            key = key.decode("utf-8")
            folder = key.split("/")[-2]
            if folder in folders and count % self.each_frame_sum < max_frame:
                image_buf = np.frombuffer(value, dtype=np.uint8)
                img = cv2.cvtColor(cv2.imdecode(image_buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
                name = key.split("/")[-1]
                data[f"{folder}/{name}"] = img
            count += 1
        env.close()
        return data

    def get_data(self):
        data = []
        data_gt = self.get_lmdb_data(self.dataroot_gt)
        data_lq = self.get_lmdb_data(self.dataroot_lq)
        for filename in data_gt:
            img_data = {
                "HR": data_gt[filename],
                "LR": data_lq[filename],
                "filename": filename,
            }
            data.append(img_data)
        return data

    def __getitem__(self, index):
        if self.num_input_frames:
            start = index
            end = index + self.num_input_frames
            HRs = list(map(lambda x: x["HR"], self._data[start:end]))
            LRs = list(map(lambda x: x["LR"], self._data[start:end]))
            indexs = list(range(start, end))
            filenames = list(map(lambda x: x["filename"], self._data[start:end]))
            return HRs, LRs, indexs, filenames
        else:
            HR = self._data[index]["HR"]
            LR = self._data[index]["LR"]
            filename = self._data[index]["filename"]
            return HR, LR, index, filename

    def __len__(self):
        if self.num_input_frames:
            return len(self._data) - self.num_input_frames + 1
        return len(self._data)


def create_reds_dataset(
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
    source = LmdbReds(dataset_dir, usage=usage, **kwargs)
    if not source._data:
        source = Reds(dataset_dir, usage=usage, **kwargs)
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
