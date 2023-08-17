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

import argparse
import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from mindediting.dataset.src.vimeo_preprocess import imresize


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess SISR datasets")
    parser.add_argument("--root", required=True, help="Path to the root directory of the dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["div2k", "flickr2k", "set5", "set14", "urban100", "bsd100"],
        help="Name of the processed dataset",
    )
    parser.add_argument("--scale", type=int, default=4, help="Scale of law resolution samples")
    parser.add_argument("--crop-size", type=int, default=320, help="Size of samples crops")
    parser.add_argument("--output", required=True, help="Path to a result pickle file or directory")
    dataset_args = parser.parse_args()
    return dataset_args


def read_image(path, to_rgb=True, to_chw=False, to_float=False):
    img = cv2.imread(path)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if to_chw:
        img = img.transpose((2, 0, 1))
    if to_float:
        img = img.astype(np.float32)
    return img


def collect_all_images(path):
    return sorted([os.path.join(path, img) for img in os.listdir(path) if img.endswith(".png")])


def collect_bsd100_images(path):
    return [
        sorted([os.path.join(path, img) for img in os.listdir(path) if img.endswith("HR.png")]),
        sorted([os.path.join(path, img) for img in os.listdir(path) if img.endswith("LR.png")]),
    ]


def get_crops(img, crop_size=320):
    h, w = img.shape[:2]
    crops = []
    for y in range(0, h, crop_size):
        for x in range(0, w, crop_size):
            left = x if x + crop_size < w else w - crop_size
            up = y if y + crop_size < h else h - crop_size
            crop = img[up : up + crop_size, left : left + crop_size, :]
            hc, wc = crop.shape[:2]
            assert hc == wc == crop_size
            crops.append(crop)
    return crops


def get_result(path, scale=4, crop_size=320, crop=True, to_rgb=True, to_chw=True, bsd100=False):
    hr_lr_pairs = []
    if bsd100:
        images = collect_bsd100_images(path)
        images = [[hr, lr] for hr, lr in zip(images[0], images[1])]
    else:
        images = collect_all_images(path)
    for image in tqdm(images, f"Processing {path}"):
        if isinstance(image, list) and len(image) == 2:
            img, img_lr = read_image(image[0], to_rgb=to_rgb), read_image(image[1], to_rgb=to_rgb)
        else:
            img, img_lr = read_image(image, to_rgb=to_rgb), None
        if crop:
            crops = get_crops(img, crop_size)
            for crop_hr in crops:
                crop_lr = imresize(crop_hr, 1 / scale)
                if to_chw:
                    crop_hr = crop_hr.transpose((2, 0, 1))
                    crop_lr = crop_lr.transpose((2, 0, 1))
                hr_lr_pairs.append((crop_hr, crop_lr))
        else:
            if img_lr is None:
                img_lr = imresize(img, 1 / scale)
            if to_chw:
                img = img.transpose((2, 0, 1))
                img_lr = img_lr.transpose((2, 0, 1))
            hr_lr_pairs.append((img, img_lr))
    return hr_lr_pairs


if __name__ == "__main__":
    args = parse_args()

    save_as_pickle = True if args.output.endswith("pickle") else False
    to_chw = True if save_as_pickle else False
    to_rgb = True if save_as_pickle else False

    if args.dataset == "div2k":
        train_dir = os.path.join(args.root, "DIV2K_train_HR")
        val_dir = os.path.join(args.root, "DIV2K_valid_HR")
    elif args.dataset == "flickr2k":
        train_dir = os.path.join(args.root, "Flickr2K_HR")
        val_dir = None
    elif args.dataset in ["set5", "set14"]:
        train_dir = None
        val_dir = os.path.join(args.root, "original")
    elif args.dataset == "urban100":
        train_dir = None
        val_dir = os.path.join(args.root, "Urban100_HR")
    else:
        train_dir = None
        val_dir = os.path.join(args.root, "image_SRF_4")

    result = dict()
    result["train"] = (
        get_result(train_dir, args.scale, args.crop_size, crop=True, to_rgb=to_rgb, to_chw=to_chw)
        if train_dir is not None
        else []
    )
    result["val"] = (
        get_result(val_dir, scale=args.scale, crop=False, to_rgb=to_rgb, to_chw=to_chw, bsd100=args.dataset == "bsd100")
        if val_dir is not None
        else []
    )

    if save_as_pickle:
        print(f"Saving to {args.output}...")
        with open(args.output, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        for subset in ["train", "val"]:
            out_dir = os.path.join(args.output, subset)
            print(f"Saving to {out_dir}...")
            hr_dir = os.path.join(out_dir, "X1")
            lr_dir = os.path.join(out_dir, f"X{args.scale}")
            os.makedirs(hr_dir, exist_ok=True)
            os.makedirs(lr_dir, exist_ok=True)
            for i, (hr, lr) in enumerate(result[subset]):
                cv2.imwrite(os.path.join(hr_dir, f"image_{i:04}.png"), hr)
                cv2.imwrite(os.path.join(lr_dir, f"image_{i:04}.png"), lr)
    print(f"train: {len(result['train'])}\nval: {len(result['val'])}")
