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

import cv2
import numpy as np


def _convert_input_type_range(image):
    image = image.astype(np.float32)
    img_type = image.dtype
    if img_type == np.float32:
        return image
    elif img_type == np.uint8:
        image /= 255.0
    else:
        raise TypeError("The img type should be np.float32 or np.uint8, " f"but got {img_type}")
    return image


def _convert_output_type_range(image, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError("The dst_type should be np.float32 or np.uint8, " f"but got {dst_type}")
    if dst_type == np.uint8:
        image = image.round()
    else:
        image /= 255.0
    return image.astype(dst_type)


def rgb2ycbcr(image, y_only=False):
    img_type = image.dtype
    image = _convert_input_type_range(image)
    if y_only:
        out_img = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        out_img = np.matmul(
            image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def bgr2ycbcr(image, y_only=False):
    image = _convert_input_type_range(image)
    img_type = image.dtype
    if y_only:
        out_img = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def imflip_(img, direction="horizontal"):
    assert direction in ["horizontal", "vertical", "diagonal"]
    if direction == "horizontal":
        flip_code = 1
    elif direction == "vertical":
        flip_code = 0
    else:
        flip_code = -1
    return cv2.flip(img, flip_code, img)


def cal_mean(x, coefficient):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean * coefficient
    x[:, 1, :, :] += green_channel_mean * coefficient
    x[:, 2, :, :] += blue_channel_mean * coefficient
    return x


def sub_mean(x):
    return cal_mean(x, -1)


def add_mean(x):
    return cal_mean(x, 1)


def quantize(img, rgb_range):
    """quantize image range to 0-255"""
    pixel_range = 255 / rgb_range
    img = np.multiply(img, pixel_range)
    img = np.clip(img, 0, 255)
    img = np.round(img) / pixel_range
    return img


def write_image(fname, data, root="output"):
    if data.ndim == 4:
        assert data.shape[0] == 1
        data = data[0, ...]
    data = np.transpose(data, (1, 2, 0))
    data = data.clip(0.0, 1.0) * 255.0
    data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not os.path.exists(root):
        os.makedirs(root)
    cv2.imwrite(os.path.join(root, fname), data)


def upsample_image(image, dsize):
    assert image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 3
    image = np.transpose(image[0, ...], (1, 2, 0))
    image = cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC)
    image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
    return image
