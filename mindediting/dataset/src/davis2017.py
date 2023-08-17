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


class Davis2017(DatasetBase):
    """
    Deblurring datasets
    Incoming Path: dataset_dir = ".../DAVIS"
    The data set directory structure is as follows:
    DAVIS
      ├── Annotations_unsupervised
      |    ├── 480p
      |    |    ├── bear
      |    |    ├── bike-packing
      |    |    ...
      |    └── Full-Resolution
      |         ├── bear
      |         ├── bike-packing
      |         ...
      ├── ImageSets
      |     └── 2017
      |          ├── train.txt
      |          └── val.txt
      └── JPEGImages
           ├── Full-Resolution
           |    ├── bear
           |    ├── bike-packing
           |    ...
           └── 480p
                ├── bear
                ├── bike-packing
                ...

    """

    def __init__(
        self,
        dataset_dir,
        usage="train",
        resolution="Full-Resolution",
        video_mode=False,
        return_paths=False,
        transforms=None,
        use_val_for_training=False,
        **kwargs,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        assert usage in {"train", "val"}
        self.usage = usage
        assert resolution in {"Full-Resolution", "480p"}
        self.resolution = resolution
        self.video_mode = video_mode
        self.return_paths = return_paths
        self.transforms = transforms
        self.use_val_for_training = use_val_for_training
        self._index = 0
        self._data = self.get_data()

    def get_data(self):
        data = []

        test_split = "test-dev" if self.use_val_for_training else "val"
        train_split = "train"

        split = test_split if self.usage == "val" else train_split

        txt_file = os.path.join(self.dataset_dir, "ImageSets", "2017", f"{split}.txt")
        with open(txt_file, "r") as f:
            readlines = f.readlines()

        if split == train_split and self.use_val_for_training:
            txt_file = os.path.join(self.dataset_dir, "ImageSets", "2017", "val.txt")
            with open(txt_file, "r") as f:
                readlines += f.readlines()

        for i in readlines:
            img_dir_name = i.rstrip("\n")
            img_dir_name_path = os.path.join(self.dataset_dir, "JPEGImages", self.resolution, img_dir_name)
            video_frames_names = list(sorted(os.listdir(img_dir_name_path)))
            video_data = []
            for img_name in video_frames_names:
                img_origin_path = os.path.join(img_dir_name_path, img_name)
                img_unsupervised_path = os.path.join(
                    self.dataset_dir,
                    "Annotations_unsupervised",
                    self.resolution,
                    img_dir_name,
                    img_name.replace(".jpg", ".png"),
                )
                if not all([os.path.exists(img_origin_path), os.path.exists(img_unsupervised_path)]):
                    warnings.warn(f"not path {img_origin_path} or {img_unsupervised_path}")
                img_data = {
                    "origin": img_origin_path,
                    "unsupervised": img_unsupervised_path,
                    "file_name": img_name,
                }
                video_data.append(img_data)

            if self.video_mode:
                data.append(video_data)
            else:
                data.extend(video_data)

        return data

    def __getitem__(self, index):
        if self.video_mode:
            video_data = self._data[index]
            frames = [v["origin"] for v in video_data]
            segments = [v["unsupervised"] for v in video_data]
            file_names = [v["file_name"] for v in video_data]
            if not self.return_paths:
                frames = list(self._load_image(frame_path) for frame_path in frames)
                segments = list(self._load_image(annotation_path) for annotation_path in segments)
            if self.transforms is not None:
                return self.transforms(frames, segments, index, file_names)
            return frames, segments, index, file_names
        else:
            HR = self._load_image(self._data[index]["origin"])
            LR = self._load_image(self._data[index]["unsupervised"])
            file_name = os.path.basename(self._data[index]["file_name"])
            if self.transforms is not None:
                return self.transforms(HR, LR, index, file_name)
            return HR, LR, index, file_name


def create_davis2017_dataset(
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
    source = Davis2017(dataset_dir, usage=usage, **kwargs)
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
