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

import random

import cv2
import numpy as np


def paired_random_crop(img_gts, img_lqs, gt_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_size [int]: GT patch size.
        scale [int]: Scale factor.
        gt_path [str]: Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    assert isinstance(img_gts, np.ndarray)
    n, c, h_lq, w_lq = img_lqs.shape
    n, c, h_gt, w_gt = img_gts.shape
    lq_size = gt_size // scale

    if h_lq < lq_size or w_lq < lq_size:
        raise ValueError(
            f"LQ ({h_lq}, {w_lq}) is smaller than patch size " f"({lq_size}, {lq_size}). " f"Please remove {gt_path}."
        )

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ", f"multiplication of LQ ({h_lq}, {w_lq})."
        )

    top = random.randint(0, h_lq - lq_size)
    left = random.randint(0, w_lq - lq_size)

    img_lqs = img_lqs[:, :, top : top + lq_size, left : left + lq_size]

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = img_gts[:, :, top_gt : top_gt + gt_size, left_gt : left_gt + gt_size]

    if img_gts.shape[0] == 1:
        img_gts = img_gts[0]
    if img_lqs.shape[0] == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def _augment(img, hflip, vflip, rot90):
    if hflip:  # horizontal
        cv2.flip(img, 1)
    if vflip:  # vertical
        cv2.flip(img, 0)
    if rot90:
        img = img.transpose(0, 2, 1)
    return img


def _augment_flow(flow, hflip, vflip, rot90):
    if hflip:  # horizontal
        cv2.flip(flow, 1, flow)
        flow[:, :, 0] *= -1
    if vflip:  # vertical
        cv2.flip(flow, 0, flow)
        flow[:, :, 1] *= -1
    if rot90:
        flow = flow.transpose(0, 2, 1)
        flow = flow[:, :, [1, 0]]
    return flow


def augment(images, hflip=True, rotation=True, flows=None, return_status=False):
    rot90 = rotation and random.random() < 0.5
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5

    if not isinstance(images, list):
        images = [images]
    images = [_augment(img, hflip, vflip, rot90) for img in images]
    if len(images) == 1:
        images = images[0]

    if flows is None:
        if return_status:
            return images, (hflip, vflip, rot90)
        else:
            return images
    else:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow, hflip, vflip, rot90) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return images, flows


def img2np_tensor(imgs, bgr2rgb=False, float32=True):
    """Numpy array to tensor.

    Args:
        imgs [list[ndarray] | ndarray]: Input images.
        bgr2rgb [bool]: Whether to change bgr to rgb.
        float32 [bool]: Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor."""

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if float32:
            img = img.astype(np.float32)
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def transform_vrt_train(**kwargs):
    scale = kwargs.get("scale", 4)
    gt_size = kwargs.get("gt_size", 256)
    use_hflip = kwargs.get("use_hflip", True)
    use_rot = kwargs.get("use_rot", True)

    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["LR", "HR"]

    def operation(HR, LR, idx, filename, batchInfo=1):
        new_image, new_label = [], []
        for i in range(len(HR)):
            label, image = HR[i], LR[i]
            # randomly crop
            img_gts, img_lqs = paired_random_crop(label, image, gt_size, scale, filename[i])
            # augmentation - flip, rotate
            imgs = [i for i in img_gts] + [j for j in img_lqs]
            img_results = augment(imgs, use_hflip, use_rot)
            img_results = img2np_tensor(img_results)
            img_gts = img_results[: len(img_results) // 2]
            img_gts = np.stack(img_gts, axis=0)
            img_lqs = img_results[len(img_results) // 2 :]
            img_lqs = np.stack(img_lqs, axis=0)

            image = img_lqs.astype(np.float32) / 255
            label = img_gts.astype(np.float32) / 255
            new_image.append(image)
            new_label.append(label)
        return new_image, new_label

    return operation, input_columns, output_columns


def transform_vrt_val(**kwargs):
    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["LR", "HR"]

    def operation(HR, LR, idx, filename, batchInfo=1):
        new_image, new_label = [], []
        for i in range(len(HR)):
            label, image = HR[i], LR[i]
            image = img2np_tensor(image, bgr2rgb=False, float32=True)
            image = np.stack(image, axis=0)

            label = img2np_tensor(label, bgr2rgb=False, float32=True)
            label = np.stack(label, axis=0)

            image = image.astype(np.float32) / 255
            label = label.astype(np.float32) / 255
            new_image.append(image)
            new_label.append(label)
        return new_image, new_label

    return operation, input_columns, output_columns
