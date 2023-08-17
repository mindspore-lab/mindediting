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

"""Dataset for CTSDG"""
import os
import random
from pathlib import Path
from typing import List

import mindspore.dataset as ds
import numpy as np
from mindspore.dataset.vision import Decode, Grayscale
from tqdm import tqdm


class Mask:
    """
    Base dataset class with images and masks for CTSDG
    Image Repair (Complete) Dataset
    Irregular Mask
    CelebA
    Paris Street-View
    Places2
    """

    def __init__(self, dataset_dir, usage=None, **kwargs):
        config = kwargs
        self.load_size = config["image_load_size"]
        self.is_training = True if usage == "train" else False
        self.masked_area = config.get("masked_area", None)

        if self.is_training:
            image_part_index = config["anno_train_index"]
            masks_root = config["train_masks_root"]
        else:
            image_part_index = config["anno_eval_index"]
            masks_root = config["eval_masks_root"]

        if "data_root" in config:
            dataset_path = config["data_root"]
        else:
            dataset_path = config["dataset_path"]
        self.image_dataset = get_images_dataset(dataset_path, config["anno_path"], image_part_index)
        self.masks_dataset = get_images_dataset(masks_root)

        area_ranges = {"s": (0, 0.2), "m": (0.2, 0.4), "l": (0.4, 0.6)}

        if self.masked_area is not None:
            if self.masked_area not in area_ranges:
                raise ValueError(
                    f"Invalid masked_area. Available options {tuple(area_ranges.keys())}, but provided value is {self.masked_area}"
                )

            area_range = area_ranges[self.masked_area]

            masked_fraction = []

            for mask_path in tqdm(self.masks_dataset, desc="loading masks"):
                with open(mask_path, "rb") as f:
                    mask = self.get_mask(f)
                masked_fraction.append(1 - (mask.sum() / mask.size))

            number_image = len(self.image_dataset)
            number_mask = len(self.masks_dataset)

            repeats = (number_image + number_mask - 1) // number_mask
            masked_fraction = (masked_fraction * repeats)[:number_image]
            self.masks_dataset = (self.masks_dataset * repeats)[:number_image]

            self.image_dataset = list(
                image_path
                for image_path, area in zip(self.image_dataset, masked_fraction)
                if self.is_in_range(area, area_range)
            )
            self.masks_dataset = list(
                image_path
                for image_path, area in zip(self.masks_dataset, masked_fraction)
                if self.is_in_range(area, area_range)
            )

        self.number_image = len(self.image_dataset)
        self.number_mask = len(self.masks_dataset)

    @staticmethod
    def get_mask(f):
        decode = Decode(to_pil=True)
        grayscale = Grayscale()
        x = f.read()
        x = decode(x)
        x = grayscale(x)
        x = np.array(x) / 255.0
        threshold = 0.5
        x = x < threshold
        return x

    @staticmethod
    def is_in_range(x, r):
        return r[0] < x <= r[1]

    def __getitem__(self, index: int):
        with open(self.image_dataset[index % self.number_image], "rb") as f:
            image = f.read()

        if self.is_training:
            mask_index = random.randint(0, self.number_mask - 1)
        else:
            mask_index = index % self.number_mask

        with open(self.masks_dataset[mask_index], "rb") as f:
            mask = f.read()
        return image, mask

    def __len__(self):
        return self.number_image


def get_images_dataset(data_root: str, anno_path=None, index=0) -> List[str]:
    """
    Find all images in specified dir and return all paths in list

    Args:
        data_root: Images root dir
        anno_path: Path to the annotation file with filenames and test/eval partitions
        index: Index to select

    Returns:
        List with image paths in root dir
    """
    root_dir = Path(data_root)
    images = []
    img_ext = [".png", ".jpg", ".jpeg"]
    if anno_path:
        if not Path(anno_path).exists():
            raise FileExistsError(f"Error - anno_path does not exist:\n{anno_path}!")

        with open(anno_path, "r") as f:
            anno_lines = f.readlines()

        for line in anno_lines:
            attr = line.split(" ")
            if int(attr[-1]) == index:
                img_path = root_dir / attr[0]
                if img_path.is_file() and img_path.suffix in img_ext:
                    images.append(img_path.as_posix())
    else:
        for image_name in os.listdir(root_dir):
            image_path = os.path.join(root_dir, image_name)
            if os.path.isfile(image_path) and "." + image_name.split(".")[-1] in img_ext:
                images.append(image_path)

    return sorted(images)


def create_mask_dataset(
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
    source = Mask(dataset_dir, usage=usage, **kwargs)
    dataset = ds.GeneratorDataset(
        source=source,
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        column_names=["image", "mask"],
        shuffle=shuffle,
    )
    return dataset
