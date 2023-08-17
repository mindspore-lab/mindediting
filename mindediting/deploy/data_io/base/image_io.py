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

import glob
import os

import cv2
import numpy as np

from mindediting.deploy.utils import constant as const


# image io
class ImageWriter:
    def __init__(self, output_path, channel_order="rgb"):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        if channel_order not in {"rgb", "bgr"}:
            raise ValueError(f"Invalid channel_order: {channel_order}")
        self.channel_order = channel_order
        self.count = 0

    def write(self, input_data):
        if not isinstance(input_data, list):
            input_data = [input_data]
        for data_item in input_data:
            if self.channel_order == "rgb":
                data_item = cv2.cvtColor(data_item.astype(np.uint8), cv2.COLOR_RGB2BGR)
            path = os.path.join(self.output_path, f"{self.count+1:06}.png")
            cv2.imwrite(path, data_item)
            self.count += 1


class ImageReader:
    def __init__(self, img_dir, windows_size=25, fixed_width=0, fixed_height=0, channel_order="rgb"):
        self.images = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.window_size = windows_size
        if channel_order not in {"rgb", "bgr"}:
            raise ValueError(f"Invalid channel_order: {channel_order}")
        self.channel_order = channel_order
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        frame_list = []
        for _ in range(self.window_size):
            frame = cv2.imread(self.images[item])
            if self.channel_order == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.fixed_width > 0 and self.fixed_height > 0:
                frame = cv2.resize(frame, (self.fixed_width, self.fixed_height), interpolation=cv2.INTER_AREA)
            frame_list.append(frame)
        return frame_list


def imread(input_file, target_color_space="rgb"):
    """
    read image using by cv2
    """

    target_color_space = target_color_space.lower()
    assert target_color_space in const.VALID_COLORSPACE

    if input_file.endswith(".exr"):
        # read hdr
        im = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
    else:
        im = cv2.imread(input_file)

    if target_color_space == "gray3d":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif target_color_space == "gray":
        im = im[:, :, 0:1]

    # data_format convert
    if target_color_space in ["bgr", "gray", "gray3d"]:
        out = im
    elif target_color_space == "rgb":
        out = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif target_color_space == "lab":
        out = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    elif target_color_space == "ycrcb":
        out = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    elif target_color_space == "yuv":
        out = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    elif target_color_space == "y":
        out = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
        out = out[:, :, 0:1]
    else:
        raise ValueError("Unknown data_format as {}, or maybe just mismatched!".format(target_color_space))

    return out


def imwrite(input_name, input_data, source_color_space="rgb", benormalized=True):
    source_color_space = source_color_space.lower()
    assert source_color_space in const.VALID_COLORSPACE

    hdr = input_name.endswith(".exr")
    out = image_deprocess(input_data, source_color_space, benormalized, hdr)
    if hdr:
        hdr_image_write(input_name, out)
    else:
        sdr_image_write(input_name, out)


def image_deprocess(x, source_color_space="rgb", benormalized=True, hdr=False):
    if hdr:
        return hdr_image_deprocess(x, source_color_space, benormalized)
    else:
        return sdr_image_deprocess(x, source_color_space, benormalized)


def hdr_image_deprocess(x, source_color_space="rgb", benormalized=True):
    if source_color_space == "rgb":
        x = x[..., ::-1]
    elif source_color_space == "bgr":
        pass
    else:
        raise NotImplementedError(f"HDR output does not support color-spaces other than RGB and BGR.")
    return x


def sdr_image_deprocess(x, source_color_space="rgb", benormalized=True):
    source_color_space = source_color_space.lower()
    assert source_color_space in const.VALID_COLORSPACE

    if benormalized and source_color_space not in ["ycrcb", "yuv", "y"]:
        x[...] = x[...] * 255
    x = np.clip(x, 0.0, 255.0)

    if source_color_space in ["bgr", "gray"]:
        out = x
    elif source_color_space == "rgb":
        out = cv2.cvtColor(x, cv2.COLOR_RGB2BGR, cv2.CV_32F)
    elif source_color_space in ["lab", "gray3d"]:
        x[:, :, 0:1] = x[:, :, 0:1] / 2.55
        x[:, :, 1:3] = x[:, :, 1:3] - 128.0
        out = cv2.cvtColor(x, cv2.COLOR_LAB2BGR, cv2.CV_32F)
        out[...] = out[...] * 255.0
    elif source_color_space == "ycrcb":
        out = cv2.cvtColor(x, cv2.COLOR_YCrCb2BGR, cv2.CV_32F)
        if benormalized:
            out = np.clip(out * 255.0, 0.0, 255.0)
    elif source_color_space == "yuv":
        out = cv2.cvtColor(x, cv2.COLOR_YUV2BGR, cv2.CV_32F)
        out = np.clip(out * 255, 0.0, 255.0)
    elif source_color_space == "y":
        out = np.clip(x * 255, 0.0, 255.0)
    else:
        raise ValueError

    out = out.astype(np.uint8)
    return out


def sdr_image_write(name, out):
    cv2.imwrite(name, out)


def hdr_image_write(name, out):
    out = np.maximum(out, 0.0)
    cv2.imwrite(name, out, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    # cv2.imwrite(name, out, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])


def sub_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, :, :, 0] -= red_channel_mean
    x[:, :, :, 1] -= green_channel_mean
    x[:, :, :, 2] -= blue_channel_mean
    return x


def add_mean(x):
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, :, :, 0] += red_channel_mean
    x[:, :, :, 1] += green_channel_mean
    x[:, :, :, 2] += blue_channel_mean
    return x


def quantize(img, rgb_range):
    """quantize image range to 0-255"""
    pixel_range = 255 / rgb_range
    img = np.multiply(img, pixel_range)
    img = np.clip(img, 0, 255)
    img = np.round(img) / pixel_range
    return img


def image_pre_transform(img):
    img = np.expand_dims(img, axis=0)
    img = img.astype("float32")
    img = sub_mean(img)
    return img


def image_post_transform(img):
    img = quantize(add_mean(img), 255)
    img = np.squeeze(img, axis=0)
    return img


def draw(img, save_path):
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR, cv2.CV_32F)
    out = out.astype(np.uint8)
    cv2.imwrite(save_path, out)
