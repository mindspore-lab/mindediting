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
import numpy as np

from .dataset_base import DatasetBase


class Vimeo90k(DatasetBase):
    """
    Denoising, deblocking, deblurring, and 4x super-resolution dataset
    Incoming Path: dataset_dir = ".../Vimeo90K"
    The data set directory structure is as follows:
    Vimeo90K
      ├── vimeo_septuplet
      |     ├── sequences
      |     |     ├── 00001
      |     |     |    ├── 00001
      |     |     |    ├── 00002
      |     |     |    ...
      |     |     ├── 00002
      |     |     |    ├── 00001
      |     |     |    ├── 00002
      |     |     |    ...
      |     |     ...
      |     ├── sep_testlist.txt
      |     └── sep_trainlist.txt
      ├── vimeo_deblocking_test
      |     ├── input
      |     |     ├── 00001
      |     |     |    ├── 0266
      |     |     |    ├── 0268
      |     |     |    ...
      |     |     ├── 00002
      |     |     |    ├── 0004
      |     |     |    ├── 0007
      |     |     |    ...
      |     |     ...
      |     ├── target
      |     |     ├── 00001
      |     |     |    ├── 0266
      |     |     |    ├── 0268
      |     |     |    ...
      |     |     ├── 00002
      |     |     |    ├── 0004
      |     |     |    ├── 0007
      |     |     |    ...
      |     |     ...
      |     └── sep_testlist.txt
      ├── vimeo_sep_noisy_correct
      |     ├── 00001
      |     |    ├── 0266
      |     |    ├── 0268
      |     |    ...
      |     ├── 00002
      |     |    ├── 0004
      |     |    ├── 0007
      |     |    ...
      |     ...
      └── vimeo_super_resolution_test
      |     ├── input
      |     |     ├── 00001
      |     |     |    ├── 0266
      |     |     |    ├── 0268
      |     |     |    ...
      |     |     ├── 00002
      |     |     |    ├── 0004
      |     |     |    ├── 0007
      |     |     |    ...
      |     |     ...
      |     ├── target
      |     |     ├── 00001
      |     |     |    ├── 0266
      |     |     |    ├── 0268
      |     |     |    ...
      |     |     ├── 00002
      |     |     |    ├── 0004
      |     |     |    ├── 0007
      |     |     |    ...
      |     |     ...
      |     ├── low_resolution
      |     |     ├── 00001
      |     |     |    ├── 0266
      |     |     |    ├── 0268
      |     |     |    ...
      |     |     ├── 00002
      |     |     |    ├── 0004
      |     |     |    ├── 0007
      |     |     |    ...
      |     |     ...
      |     └── sep_testlist.txt
    """

    def __init__(
        self, dataset_dir, usage=None, nframes=0, random_reverse=None, return_paths=False, transforms=None, **kwargs
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.usage = usage
        self.num_input_frames = nframes
        assert self.num_input_frames <= 7
        self.random_reverse = random_reverse
        if random_reverse is None:
            self.random_reverse = usage == "train"
        self.return_paths = return_paths
        self.transforms = transforms

    def get_data(self, dataset_dir, usage):
        data = []
        data_dir = os.path.join(dataset_dir, usage)
        if not os.path.exists(data_dir):
            data_dir = dataset_dir
        input_dir = os.path.join(data_dir, self._dataset_name)
        for dir1 in os.listdir(input_dir):
            dir1_path = os.path.join(input_dir, dir1)
            for dir2 in os.listdir(dir1_path):
                folder_path = os.path.join(dir1_path, dir2)
                if not os.path.exists(folder_path):
                    warnings.warn(f"not folder_path {folder_path}")
                    continue
                img_name_list = [
                    f"im{i}.png" for i in range(4 - self.num_input_frames // 2, 4 + self.num_input_frames // 2 + 1)
                ]
                img_data_list = []
                for img_name in img_name_list:
                    loss_img_path = os.path.join(folder_path, img_name)
                    full_img_path = os.path.join(dataset_dir, "vimeo_septuplet", "sequences", dir1, dir2, img_name)
                    if not all([os.path.exists(loss_img_path), os.path.exists(full_img_path)]):
                        warnings.warn(f"not path {full_img_path} or {loss_img_path}")
                        img_data_list = []
                        break
                    img_data = {
                        "full_img": full_img_path,
                        "loss_img": loss_img_path,
                        "filename": img_name,
                    }
                    img_data_list.append(img_data)
                data.extend(img_data_list)
        assert len(data) > 0
        return data

    def __getitem__(self, index):
        if self.num_input_frames:
            start = index * self.num_input_frames
            end = (index + 1) * self.num_input_frames
            data_list = self._data[start:end]
            if self.random_reverse and random.random() < 0.5:
                data_list.reverse()
                indexs = list(range(end - 1, start - 1, -1))
            else:
                indexs = list(range(start, end))
            HRs = list(x["full_img"] for x in data_list)
            LRs = list(x["loss_img"] for x in data_list)
            filenames = list(x["filename"] for x in data_list)
            if not self.return_paths:
                HRs = np.array(list(self._load_image(image_path) for image_path in HRs))
                LRs = np.array(list(self._load_image(image_path) for image_path in LRs))
            if self.transforms is not None:
                return self.transforms(HRs, LRs, indexs, filenames)
            return HRs, LRs, indexs, filenames
        else:
            HR = self._load_image(self._data[index]["full_img"])
            LR = self._load_image(self._data[index]["loss_img"])
            filename = self._data[index]["filename"]
            return HR, LR, index, filename

    def __len__(self):
        if self.num_input_frames:
            return len(self._data) // self.num_input_frames
        return len(self._data)


class VimeoDenoising(Vimeo90k):
    """
    Denoising dataset
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        super().__init__(dataset_dir, usage, **kwargs)
        self._dataset_name = "vimeo_sep_noisy_correct"
        self._data = self.get_data(self.dataset_dir, self.usage)


class VimeoDeblocking(Vimeo90k):
    """
    Deblocked dataset
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        super().__init__(dataset_dir, usage, **kwargs)
        self._dataset_name = "vimeo_deblocking_test"
        self._data = self.get_data(self.dataset_dir, self.usage)


class VimeoBlur(Vimeo90k):
    """
    Deblurring dataset
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        super().__init__(dataset_dir, usage, **kwargs)
        self._dataset_name = os.path.join("vimeo_super_resolution_test", "input")
        self._data = self.get_data(self.dataset_dir, self.usage)


class VimeoSuperResolutionTest(Vimeo90k):
    """
    4X Super-Resolution Test Dataset
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        super().__init__(dataset_dir, usage, **kwargs)
        self._dataset_name = os.path.join("vimeo_super_resolution_test", "low_resolution")
        self._data = self.get_data(self.dataset_dir, self.usage)


class VimeoSuperResolution(DatasetBase):
    """
    4X super-resolution training dataset
    .
    └─ data
    └─ vimeo90k
        ├─ sequences
        │  ├─ 00001
        │  │  ├─ 0001
        │  │  │  ├─ im1.png
        │  │  │   ...
        │  │  │  └─ im7.png
        │  │  ├─ ...
        │  │  ...
        │  ├─ 00002
        │  │  ├─ 0001
        │  │  │  ├─ im1.png
        │  │  │   ...
        │  │  │  └─ im7.png
        │  │  ├─ ...
        │  │  ...
        │  ...
        └─ BIx4
        ├─ 00001
        │  ├─ 0001
        │  │  ├─ im1.png
        │  │   ...
        │  │  └─ im7.png
        │  ├─ ...
        │  ...
        ├─ 00002
        │  ├─ 0001
        │  │  ├─ im1.png
        │  │   ...
        │  │  └─ im7.png
        │  ├─ ...
        │  ...
        ...
    """

    def __init__(
        self,
        dataset_dir,
        usage=None,
        random_reverse=None,
        train_annotation=None,
        test_annotation=None,
        return_paths=False,
        transforms=None,
        **kwargs,
    ):
        super().__init__()
        self.gt_subdir = kwargs.get("gt_subdir", "sequences")
        self.lr_subdir = kwargs.get("lr_subdir", "BIx4")
        self.num_input_frames = kwargs.get("nframes", 0)
        assert self.num_input_frames <= 7
        self.usage = usage
        self.return_paths = return_paths
        self.transforms = transforms
        self.random_reverse = random_reverse
        if self.random_reverse is None:
            self.random_reverse = usage == "train"
        self.sep_file = train_annotation if usage == "train" else test_annotation
        assert self.sep_file is not None
        self._data = self.get_data(dataset_dir)

    def get_data(self, dataset_dir):
        data = []
        with open(os.path.join(dataset_dir, self.sep_file), "r") as f:
            folder_list = f.read().strip().split("\n")
        lr_pth = os.path.join(dataset_dir, self.lr_subdir)
        for folder in folder_list:
            folder_path = os.path.join(lr_pth, folder)
            if not os.path.exists(folder_path):
                warnings.warn(f"not folder_path {folder_path}")
                continue

            img_name_list = [
                f"im{i}.png" for i in range(4 - self.num_input_frames // 2, 4 + self.num_input_frames // 2 + 1)
            ]
            full_img_paths = []
            loss_img_paths = []
            for img_name in img_name_list:
                loss_img_path = os.path.join(folder_path, img_name)
                full_img_path = os.path.join(dataset_dir, self.gt_subdir, folder, img_name)
                if not all([os.path.exists(loss_img_path), os.path.exists(full_img_path)]):
                    warnings.warn(f"not path {full_img_path} or {loss_img_path}")
                    break
                full_img_paths.append(full_img_path)
                loss_img_paths.append(loss_img_path)
            img_data = {
                "full_img": full_img_paths,
                "loss_img": loss_img_paths,
                "filename": img_name_list,
            }
            data.append(img_data)
        assert len(data) > 0
        return data

    def __getitem__(self, index):
        if self.num_input_frames:
            start = index * self.num_input_frames
            end = (index + 1) * self.num_input_frames
            data_list = self._data[start:end]
            if self.random_reverse and random.random() < 0.5:
                data_list.reverse()
            indexes = [index]
            HRs = list(image_path for image_path in self._data[index]["full_img"])
            LRs = list(image_path for image_path in self._data[index]["loss_img"])
            filenames = self._data[index]["filename"]
            if not self.return_paths:
                HRs = np.array(list(self._load_image(image_path) for image_path in HRs))
                LRs = np.array(list(self._load_image(image_path) for image_path in LRs))
            if self.transforms is not None:
                return self.transforms(HRs, LRs, indexes, filenames)
            return HRs, LRs, indexes, filenames
        else:
            HR = [self._load_image(image) for image in self._data[index]["full_img"]]
            LR = [self._load_image(image) for image in self._data[index]["loss_img"]]
            filename = self._data[index]["filename"]
            return HR, LR, index, filename

    def __len__(self):
        return len(self._data)


class VimeoTriplet(DatasetBase):
    """
    Video Frame Interpolation (VFI) dataset
    """

    flow_types = ["flowformer", "liteflow", None]

    def __init__(
        self,
        dataset_dir,
        usage=None,
        train_annotation="tri_trainlist.txt",
        test_annotation="tri_testlist.txt",
        flow_type="flowformer",
        crop_border=None,
        **kwargs,
    ):
        super().__init__()

        self.crop_border = crop_border
        if self.crop_border is not None:
            if isinstance(self.crop_border, int):
                self.crop_border = [self.crop_border] * 4
            assert len(crop_border) == 4

        self._annot_files = dict(
            train=os.path.join(dataset_dir, train_annotation), val=os.path.join(dataset_dir, test_annotation)
        )
        self._images_folder = os.path.join(dataset_dir, "sequences")

        assert flow_type in self.flow_types
        self._enable_flow = flow_type is not None
        if self._enable_flow:
            self._flow_folder = os.path.join(dataset_dir, f"flow_{flow_type}")

        self._data = self.get_data(usage)

    @property
    def column_names(self):
        if self._enable_flow:
            return ["img0", "img1", "gt", "flow0", "flow1", "idx", "filename"]
        else:
            return ["img0", "img1", "gt", "idx", "filename"]

    def get_data(self, usage):
        assert usage in self._annot_files.keys()
        ann_file = self._annot_files[usage]
        with open(ann_file, "r") as f:
            keys = f.read().split("\n")
            keys = [k.strip() for k in keys if k.strip() is not None and k != ""]

        data_infos = []
        for key in keys:
            key = key.replace("/", os.sep)

            key_folder_img = os.path.join(self._images_folder, key)
            inputs_path = [os.path.join(key_folder_img, "im1.png"), os.path.join(key_folder_img, "im3.png")]
            target_path = os.path.join(key_folder_img, "im2.png")

            data_info = dict(inputs_path=inputs_path, target_path=target_path, key=key)

            flows_path = None
            if self._enable_flow:
                key_folder_flow = os.path.join(self._flow_folder, key)
                flows_path = [
                    os.path.join(key_folder_flow, "flow_t0.flo"),
                    os.path.join(key_folder_flow, "flow_t1.flo"),
                ]
            data_info["flow_path"] = flows_path

            data_infos.append(data_info)

        assert len(data_infos) > 0

        return data_infos

    def _load_flow(self, flow_path):
        with open(flow_path, "rb") as f:
            flow_bytes = f.read()

        header = flow_bytes[:4]
        if header.decode("utf-8") != "PIEH":
            raise Exception("Flow file header does not contain PIEH")

        width = np.frombuffer(flow_bytes, np.int32, 1, 4)[0]
        height = np.frombuffer(flow_bytes, np.int32, 1, 8)[0]

        flow = np.frombuffer(flow_bytes, np.float32, width * height * 2, 12)
        flow = flow.reshape((height, width, 2))
        flow = flow.transpose((2, 0, 1))

        return flow

    @staticmethod
    def _crop(image, crop_border):
        w_left, w_right, h_top, h_bottom = crop_border

        h, w = image.shape[-2:]
        out = image[..., h_top : (h - h_bottom), w_left : (w - w_right)]

        return out

    def __getitem__(self, index):
        data_info = self._data[index]
        img_0 = self._load_image(data_info["inputs_path"][0])
        img_1 = self._load_image(data_info["inputs_path"][1])
        img_target = self._load_image(data_info["target_path"])
        filename = self._data[index]["key"]

        if self.crop_border is not None:
            img_0 = self._crop(img_0, self.crop_border)
            img_1 = self._crop(img_1, self.crop_border)
            img_target = self._crop(img_target, self.crop_border)

        if self._enable_flow:
            flow_0 = self._load_flow(data_info["flow_path"][0])
            flow_1 = self._load_flow(data_info["flow_path"][1])

            if self.crop_border is not None:
                flow_0 = self._crop(flow_0, self.crop_border)
                flow_1 = self._crop(flow_1, self.crop_border)

            return img_0, img_1, img_target, flow_0, flow_1, index, filename

        return img_0, img_1, img_target, index, filename


def create_vimeo_denoising_dataset(
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
    source = VimeoDenoising(dataset_dir, usage=usage, **kwargs)
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


def create_vimeo_deblocking_dataset(
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
    source = VimeoDeblocking(dataset_dir, usage=usage, **kwargs)
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


def create_vimeo_blur_dataset(
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
    source = VimeoBlur(dataset_dir, usage=usage, **kwargs)
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


def create_vimeo_super_resolution_test_dataset(
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
    every_nth=0,
    **kwargs,
):
    source = VimeoSuperResolutionTest(dataset_dir, usage=usage, **kwargs)
    if every_nth > 1:
        source._data = source._data[::every_nth]
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


def create_vimeo_super_resolution_dataset(
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
    every_nth=0,
    **kwargs,
):
    source = VimeoSuperResolution(dataset_dir, usage=usage, **kwargs)
    if every_nth > 1:
        source._data = source._data[::every_nth]
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


def create_vimeo_triplet(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    every_nth=0,
    **kwargs,
):
    source = VimeoTriplet(dataset_dir, usage=usage, **kwargs)

    if every_nth > 1:
        source._data = source._data[::every_nth]

    dataset = ds.GeneratorDataset(
        source=source,
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        column_names=source.column_names,
        shuffle=shuffle,
    )

    return dataset
