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

"""imagent"""
import io
import os
import random

import imageio
import mindspore.dataset as ds
import numpy as np
from munch import DefaultMunch
from PIL import Image


def search(root, target="JPEG"):
    item_list = []
    items = os.listdir(root)
    for item in items:
        item_path = os.path.join(root, item)
        if os.path.isdir(item_path):
            item_list.extend(search(item_path, target))
        elif item_path.split(".")[-1] == target:
            item_list.append(item_path)
        elif item_path.split("/")[-1].startswith(target):
            item_list.append(item_path)
    return item_list


def get_patch_img(img, patch_size=96, scale=2):
    tp = scale * patch_size
    ih, iw = img.shape[:2]
    if (iw - tp) > -1 and (ih - tp) > 1:
        ix = random.randrange(0, iw - tp + 1)
        iy = random.randrange(0, ih - tp + 1)
        hr_img = img[iy : iy + tp, ix : ix + tp, :3]
    elif (iw - tp) > -1 >= (ih - tp):
        ix = random.randrange(0, iw - tp + 1)
        hr_img = img[:, ix : ix + tp, :3]
        pil_img = Image.fromarray(hr_img).resize((tp, tp), Image.BILINEAR)
        hr_img = np.array(pil_img)
    elif (iw - tp) <= -1 < (ih - tp):
        iy = random.randrange(0, ih - tp + 1)
        hr_img = img[iy : iy + tp, :, :3]
        pil_img = Image.fromarray(hr_img).resize((tp, tp), Image.BILINEAR)
        hr_img = np.array(pil_img)
    else:
        pil_img = Image.fromarray(img).resize((tp, tp), Image.BILINEAR)
        hr_img = np.array(pil_img)
    return hr_img


class ImagenetIpt:
    """imagent"""

    def __init__(self, dataset_dir, usage=None, **kwargs):
        self.scale = kwargs["scale"]
        self.idx_scale = 0
        self.dataroot = dataset_dir
        self.img_list = search(os.path.join(self.dataroot, "train"), "JPEG")
        self.img_list.extend(search(os.path.join(self.dataroot, "val"), "JPEG"))
        self.img_list = sorted(self.img_list)
        self.train = True if usage == "train" else False
        self.args = DefaultMunch.fromDict(kwargs)
        self.len = len(self.img_list)
        print("Data length:", len(self.img_list))
        if self.args.derain:
            self.derain_dataroot = os.path.join(self.dataroot, "RainTrainL")
            self.derain_img_list = search(self.derain_dataroot, "rainstreak")

    def _get_index(self, idx):
        return idx % len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def _np2Tensor(img, rgb_range):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = np_transpose.astype(np.float32)
        tensor = tensor * (rgb_range / 255)
        return tensor

    def _load_file(self, index):
        index = self._get_index(index)
        f_lr = self.img_list[index]
        lr = imageio.imread(f_lr)
        if len(lr.shape) == 2:
            lr = np.dstack([lr, lr, lr])
        return lr, f_lr

    def __getitem__(self, index):
        if self.args.model == "vtip" and self.train and self.args.alltask:
            lr_image, filename = self._load_file(index % self.len)
            rain = self._load_rain()
            rain = np.expand_dims(rain, axis=2)
            rain = self.get_patch(rain, 1)
            rain = self._np2Tensor(rain, rgb_range=self.args.rgb_range)
            pair_list = []
            for idx_scale in range(4):
                self.idx_scale = idx_scale
                pair = self.get_patch(lr_image)
                pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
                pair_list.append(pair_t)
            return pair_list[3], rain, pair_list[0], pair_list[1], pair_list[2], [self.scale], [filename]
        if self.args.model == "vtip" and self.train and len(self.scale) > 1:
            lr_image, filename = self._load_file(index % self.len)
            pair_list = []
            for idx_scale in range(3):
                self.idx_scale = idx_scale
                pair = self.get_patch(lr_image)
                pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
                pair_list.append(pair_t)
            return pair_list[0], pair_list[1], pair_list[2], filename
        if self.args.model == "vtip" and self.args.derain and self.scale[self.idx_scale] == 1:
            lr_image, filename = self._load_file(index % self.len)
            rain = self._load_rain()
            rain = np.expand_dims(rain, axis=2)
            rain = self.get_patch(rain, 1)
            rain = self._np2Tensor(rain, rgb_range=self.args.rgb_range)
            pair = self.get_patch(lr_image)
            pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
            return pair_t, rain, filename
        if self.args.jpeg:
            hr_image, filename = self._load_file(index % self.len)
            buffer = io.BytesIO()
            width, height = hr_image.size
            patch_size = self.scale[self.idx_scale] * self.args.patch_size
            if width < patch_size:
                hr_image = hr_image.resize((patch_size, height), Image.ANTIALIAS)
                width, height = hr_image.size
            if height < patch_size:
                hr_image = hr_image.resize((width, patch_size), Image.ANTIALIAS)
            hr_image.save(buffer, format="jpeg", quality=25)
            lr_image = Image.open(buffer)
            lr_image = np.array(lr_image).astype(np.float32)
            hr_image = np.array(hr_image).astype(np.float32)
            lr_image = self.get_patch(lr_image)
            hr_image = self.get_patch(hr_image)
            lr_image = self._np2Tensor(lr_image, rgb_range=self.args.rgb_range)
            hr_image = self._np2Tensor(hr_image, rgb_range=self.args.rgb_range)
            return lr_image, hr_image, filename
        lr, filename = self._load_file(index % self.len)
        pair = self.get_patch(lr)
        pair_t = self._np2Tensor(pair, rgb_range=self.args.rgb_range)
        return pair_t, filename

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

    def get_patch(self, lr, scale=0):
        if scale == 0:
            scale = self.scale[self.idx_scale]
        lr = get_patch_img(lr, patch_size=self.args.patch_size, scale=scale)
        return lr

    def _load_rain(self):
        idx = random.randint(0, len(self.derain_img_list) - 1)
        f_lr = self.derain_img_list[idx]
        lr = imageio.imread(f_lr, as_gray=True)
        return lr


def create_imagenet_dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    **kwargs
):
    source = ImagenetIpt(dataset_dir, usage=usage, **kwargs)
    dataset = ds.GeneratorDataset(
        source=source,
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        column_names=["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"],
        shuffle=shuffle,
    )
    return dataset
