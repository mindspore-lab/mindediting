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

import sys

import cv2
import numpy as np

from mindediting.dataset.transforms.common import Collect, Compose, RescaleToZeroOne

IMG_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP")


class GenerateSegmentIndices:
    """Generate frame indices for a segment. It also performs temporal
    augmentation with random interval.

    Required keys: lq_path, gt_path, key, num_input_frames, sequence_length
    Added or modified keys:  lq_path, gt_path, interval, reverse

    Args:
        interval_list [list[int]]: Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        start_idx [int]: The index corresponds to the first frame in the
            sequence. Default: 0.
        filename_tmpl [str]: Template for file name. Default: '{:08d}.png'.
    """

    def __init__(self, interval_list, start_idx=0, filename_tmpl="{:08d}.png"):
        self.interval_list = interval_list
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx

    @staticmethod
    def __call__(results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information.
        """
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(interval_list={self.interval_list})"
        return repr_str


class LoadImageFromFile:
    """Load image from file.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        backend (str): The image loading backend type. Options are `cv2`,
            `pillow`, and 'turbojpeg'. Default: None.
        kwargs (dict): Args for file client.
    """

    def __init__(
        self,
        io_backend="disk",
        key="gt",
        flag="color",
        channel_order="bgr",
        save_original_img=False,
        use_cache=False,
        backend=None,
        **kwargs,
    ):
        self.io_backend = io_backend
        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.kwargs = kwargs
        self.file_client = None
        self.use_cache = use_cache
        self.cache = None
        self.backend = backend

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(io_backend={self.io_backend}, key={self.key}, "
            f"flag={self.flag}, save_original_img={self.save_original_img}, "
            f"channel_order={self.channel_order}, use_cache={self.use_cache})"
        )
        return repr_str

    @staticmethod
    def __call__(results):
        """Call function.

        Args:
            results [(]dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information."""
        return results


class LoadImageFromFileList(LoadImageFromFile):
    """Load image from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        return results


class FramesToTensor:
    def __init__(self, keys, to_float32=True):
        self.keys = keys
        self.to_float32 = to_float32

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information."""

        for key in self.keys:
            for idx, v in enumerate(results[key]):
                # deal with gray scale img: expand a color channel
                if len(v.shape) == 2:
                    v = v[None, ...]

                if self.to_float32 and not isinstance(v, np.float32):
                    v = v.astype(np.float32)

                results[key][idx] = v  # .transpose(2, 0, 1)

        return results

    def __repr__(self):
        return self.__class__.__name__ + (f"(keys={self.keys}, to_float32={self.to_float32})")


class TemporalReverse:
    """Reverse frame lists for temporal augmentation.

    Required keys are the keys in attributes "lq" and "gt",
    added or modified keys are "lq", "gt" and "reverse".

    Args:
        keys [list[str]]: The frame lists to be reversed.
        reverse_ratio [float]: The probability to reverse the frame lists.
            Default: 0.5.
    """

    def __init__(self, keys, reverse_ratio=0.5):
        self.keys = keys
        self.reverse_ratio = reverse_ratio

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys}, reverse_ratio={self.reverse_ratio})"
        return repr_str

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        reverse = np.random.random() < self.reverse_ratio

        if reverse:
            for key in self.keys:
                results[key] = np.flip(results[key], axis=0)

        results["reverse"] = reverse

        return results


class PairedRandomCrop:
    """Paried random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Args:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information.
        """
        scale = results["scale"]
        lq_patch_size = self.gt_patch_size // scale

        b, _, h_lq, w_lq = results["lq"].shape
        b, _, h_gt, w_gt = results["gt"].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f"Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x " f"multiplication of LQ ({h_lq}, {w_lq})."
            )
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
                f"({lq_patch_size}, {lq_patch_size}). Please check "
                f'{results["lq_path"]} and {results["gt_path"]}.'
            )

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_lq - lq_patch_size + 1)
        left = np.random.randint(w_lq - lq_patch_size + 1)
        # crop lq patch
        results["lq"] = results["lq"][:, :, top : top + lq_patch_size, left : left + lq_patch_size]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results["gt"] = results["gt"][
            :, :, top_gt : top_gt + self.gt_patch_size, left_gt : left_gt + self.gt_patch_size
        ]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(gt_patch_size={self.gt_patch_size})"
        return repr_str


def imflip_(img, direction="horizontal"):
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image (inplace).
    """
    assert direction in ["horizontal", "vertical", "diagonal"]
    if direction == "horizontal":
        return cv2.flip(img, 1)
    elif direction == "vertical":
        return cv2.flip(img, 0)
    else:
        return cv2.flip(img, -1)


class Flip:
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are "flip",
    "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Args:
        keys [list[str]]: The images to be flipped.
        flip_ratio [float]: The propability to flip the images.
        direction [str]: Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal"."""

    _directions = ["horizontal", "vertical"]

    def __init__(self, keys, flip_ratio=0.5, direction="horizontal"):
        self.keys = keys
        self.flip_ratio = flip_ratio
        if direction not in self._directions:
            raise ValueError(
                f"Direction {direction} is not supported." f"Currently support ones are {self._directions}"
            )
        self.direction = direction

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys}, flip_ratio={self.flip_ratio}, " f"direction={self.direction})"
        return repr_str

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        flip = np.random.random() < self.flip_ratio
        if flip:
            for key in self.keys:
                for v in results[key]:
                    imflip_(v, self.direction)

        results["flip"] = flip
        results["flip_direction"] = self.direction

        return results


class RandomTransposeHW:
    """Randomly transpose images in H and W dimensions with a probability.

    [TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees]
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".
    It also supports randomly transposing a list of images.

    Args:
        keys [list[str]]: The images to be transposed.
        transpose_ratio [float]: The propability to transpose the images.
    """

    def __init__(self, keys, transpose_ratio=0.5):
        self.keys = keys
        self.transpose_ratio = transpose_ratio

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys}, transpose_ratio={self.transpose_ratio})"
        return repr_str

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information."""
        transpose = np.random.random() < self.transpose_ratio

        if transpose:
            for key in self.keys:
                if isinstance(results[key], list):
                    results[key] = [v.transpose(0, 2, 1) for v in results[key]]
                else:
                    results[key] = results[key].transpose(0, 1, 3, 2)
        results["transpose"] = transpose
        return results


def create_pipeline(pipeline):
    transforms = []
    for transform in pipeline:
        for transform_name, parameters in transform.cfg_dict.items():
            transforms.append(getattr(sys.modules[__name__], transform_name)(**parameters))
    return Compose(transforms)


def transform_ttvsr_train(**kwargs):
    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["image", "label"]
    pipeline = create_pipeline(kwargs.get("pipeline", []))
    scale = kwargs.get("scale", 4)

    def operation(HR, LR, idx, filename, batchInfo=1):
        new_image, new_label = [], []
        for i in range(len(HR)):
            data = {"lq": LR[i], "gt": HR[i], "scale": scale}
            image, label = pipeline(data)
            new_image.append(image)
            new_label.append(label)
        return new_image, new_label

    return operation, input_columns, output_columns


def transform_ttvsr_val(**kwargs):
    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["image", "label"]
    pipeline = create_pipeline(kwargs.get("pipeline", []))

    def operation(HR, LR, idx, filename, batchInfo=1):
        new_image, new_label = [], []
        for i in range(len(HR)):
            data = {
                "lq": LR[i],
                "gt": HR[i],
                "lq_path": filename[i],
                # "scale": scale
            }
            image, label = pipeline(data)
            new_image.append(image)
            new_label.append(label)
        return new_image, new_label

    return operation, input_columns, output_columns
