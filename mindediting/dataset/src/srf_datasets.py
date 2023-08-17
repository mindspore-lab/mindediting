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

import h5py
import mindspore.dataset as ds
import numpy as np

from .dataset_base import DatasetBase


class SrfDataset(DatasetBase):
    """
    2X and 4X super-resolution dataset
    Includes Set5, Set14, BSD100, and Urban100 datasets.
    Set5:
        Incoming Path: dataset_dir = ".../Set5/image_SRF_2"
        The data set directory structure is as follows:
        Set5
        ├── img_001_SRF_2_HR.png
        ├── img_001_SRF_2_LR.png
        ...
    Set14:
        Incoming Path: dataset_dir = ".../Set14/image_SRF_2"
        The data set directory structure is as follows:
        Set14
        ├── img_001_SRF_2_HR.png
        ├── img_001_SRF_2_LR.png
        ...
    BSD100:
        Incoming Path: dataset_dir = ".../BSD100/image_SRF_2"
        The data set directory structure is as follows:
        BSD100
        ├── img_001_SRF_2_HR.png
        ├── img_001_SRF_2_LR.png
        ...
    Urban100:
        Incoming Path: dataset_dir = ".../Urban100/image_SRF_2"
        The data set directory structure is as follows:
        Urban100
        ├── img_001_SRF_2_HR.png
        ├── img_001_SRF_2_LR.png
        ...
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        DatasetBase.__init__(self)
        self.scale = kwargs.get("scale", 2)
        if isinstance(self.scale, list):
            self.scale = self.scale[0]
        self.dataset_dir = dataset_dir
        self.usage = usage
        self._index = 0
        self._data = self.get_data_bicubic(self.dataset_dir, self.usage)

    def get_data(self, dataset_dir, usage):
        data = []
        for img_name in os.listdir(dataset_dir):
            if "HR.png" in img_name:
                hr_img_path = os.path.join(dataset_dir, img_name)
                lr_img_path = hr_img_path.replace("HR", "LR")
                if not all([os.path.exists(hr_img_path), os.path.exists(lr_img_path)]):
                    warnings.warn(f"not path {hr_img_path} or {lr_img_path}")
                    continue
                img_data = {
                    "HR": hr_img_path,
                    "LR": lr_img_path,
                    "filename": img_name,
                }
                data.append(img_data)
        return data

    def get_data_bicubic(self, dataset_dir, usage):
        data = []
        hr_path = os.path.join(dataset_dir, "HR")
        lr_path = os.path.join(dataset_dir, "LR_bicubic", f"X{self.scale}")

        for img_name in sorted(os.listdir(hr_path)):
            hr_img_path = os.path.join(hr_path, img_name)
            lr_img_path = os.path.join(lr_path, img_name.replace(".png", f"x{self.scale}.png"))
            if not all([os.path.exists(hr_img_path), os.path.exists(lr_img_path)]):
                print(f"The image path {hr_img_path} or {lr_img_path} is not found.")
                continue
            img_data = {
                "HR": hr_img_path,
                "LR": lr_img_path,
                "filename": img_name,
            }
            data.append(img_data)
        return data

    def get_patch(self, hr, lr):
        """
        高分辨率和低分辨率图像保持比例
        """
        c, h, w = lr.shape
        hr = hr[:, 0 : h * self.scale, 0 : w * self.scale]
        return hr, lr

    def __getitem__(self, index):
        HR = self._load_image(self._data[index]["HR"])
        LR = self._load_image(self._data[index]["LR"])
        HR, LR = self.get_patch(HR, LR)
        filename = self._data[index]["filename"]
        return HR, LR, index, filename


class H5SrfDataset(object):
    def __init__(self, dataset_dir, usage=None, **kwargs):
        self.h5_file = dataset_dir

    def __getitem__(self, index):
        with h5py.File(self.h5_file, "r") as f:
            HR = np.expand_dims(f["hr"][str(index)][:, :], 0)
            LR = np.expand_dims(f["lr"][str(index)][:, :], 0)
            return HR, LR, index, self.h5_file

    def __len__(self):
        with h5py.File(self.h5_file, "r") as f:
            return len(f["lr"])


def create_srf_dataset(
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
    if h5py.is_hdf5(dataset_dir):
        source = H5SrfDataset(dataset_dir, usage=usage, **kwargs)
    else:
        source = SrfDataset(dataset_dir, usage=usage, **kwargs)
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
