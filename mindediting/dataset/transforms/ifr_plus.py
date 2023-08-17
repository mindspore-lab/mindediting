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


class RandomResize:
    """Resize data to a specific size for training or resize the images to fit
    the network input regulation for testing.

    When used for resizing images to fit network input regulation, the case is
    that a network may have several downsample and then upsample operation,
    then the input height and width should be divisible by the downsample
    factor of the network.
    For example, the network would downsample the input for 5 times with
    stride 2, then the downsample factor is 2^5 = 32 and the height
    and width should be divisible by 32.

    Required keys are the keys in attribute "keys", added or modified keys are
    "keep_ratio", "scale_factor", "interpolation" and the
    keys in attribute "keys".

    All keys in "keys" should have the same shape. "test_trans" is used to
    record the test transformation to align the input's shape.

    Args:
        keys (list[str]): The images to be resized.
        scale (float): Target spatial size is scaled by input size.
    """

    _valid_modes = ["img", "flow"]

    def __init__(self, keys, resize_ratio=0.5, modes="img", scale=1.0):
        assert len(keys) > 0
        assert 0.0 <= resize_ratio <= 1.0

        if isinstance(modes, str):
            assert modes in self._valid_modes
            modes = [modes] * len(keys)
        elif isinstance(modes, (tuple, list)):
            assert len(modes) == len(keys)
            for mode in modes:
                assert mode in self._valid_modes
        else:
            raise TypeError(f"Modes must be str or tuple/list of str, but got {type(modes)}.")

        self.keys = keys
        self.modes = modes
        self.resize_ratio = resize_ratio
        self.scale = scale

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys}, modes={self.modes}, scale={self.scale}, " f"keep_ratio={self.keep_ratio})"
        return repr_str

    def _resize(self, data, size, mode):
        data = data.transpose(1, 2, 0)
        data = cv2.resize(data, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        data = data.transpose(2, 0, 1)

        if mode == "flow":
            data = self.scale * data

        return data

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        resize = np.random.random() < self.resize_ratio
        results["resize"] = resize

        if resize:
            for key, mode in zip(self.keys, self.modes):
                if isinstance(results[key], (tuple, list)):
                    output_data = []
                    for input_img in results[key]:
                        output_img = self._resize(input_img, self.scale, mode)
                        if len(output_img.shape) == 2:
                            output_img = np.expand_dims(output_img, axis=0)
                        output_data.append(output_img)
                    results[key] = output_data
                else:
                    results[key] = self._resize(results[key], self.scale, mode)
                    if len(results[key].shape) == 2:
                        results[key] = np.expand_dims(results[key], axis=0)

            results["scale"] = self.scale

        return results


class RandomCrop:
    """Crop paired data (at a specific position) to specific size for training.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
    """

    def __init__(self, keys, crop_size):
        assert len(keys) > 0
        assert len(crop_size) == 2

        self.keys = keys
        self.crop_size = crop_size

    def _crop(self, data, x_offset, y_offset, crop_w, crop_h):
        return data[:, y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"keys={self.keys}, crop_size={self.crop_size}"
        return repr_str

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if isinstance(results[self.keys[0]], list):
            data_h, data_w = results[self.keys[0]][0].shape[1:]
        else:
            data_h, data_w = results[self.keys[0]].shape[1:]
        crop_h, crop_w = self.crop_size
        crop_h = min(data_h, crop_h)
        crop_w = min(data_w, crop_w)

        x_offset = np.random.randint(0, data_w - crop_w + 1)
        y_offset = np.random.randint(0, data_h - crop_h + 1)

        for k in self.keys:
            images = results[k]
            is_list = isinstance(images, (tuple, list))
            if not is_list:
                images = [images]

            cropped_images = []
            for image in images:
                if image.shape[1] != data_h or image.shape[2] != data_w:
                    raise ValueError(
                        "The sizes of paired images should be the same. "
                        f"Expected ({data_h}, {data_w}), "
                        f"but got ({image.shape[1]}, {image.shape[2]})."
                    )

                data_ = self._crop(image, x_offset, y_offset, crop_w, crop_h)
                cropped_images.append(data_)

            if not is_list:
                cropped_images = cropped_images[0]

            results[k] = cropped_images

        results["crop_size"] = self.crop_size

        return results


class RandomFlip:
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are "flip",
    "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Args:
        keys [list[str]]: The images to be flipped.
        modes [list[str]]: The operation type (image or flow) to be applied.
        flip_ratio [float]: The probability to flip the images.
        direction [str]: Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal"."""

    _valid_directions = ["horizontal", "vertical"]
    _valid_modes = ["img", "flow"]

    def __init__(self, keys, modes, flip_ratio=0.5, direction="horizontal"):
        assert len(keys) > 0
        assert 0 <= flip_ratio <= 1

        if isinstance(modes, str):
            assert modes in self._valid_modes
            modes = [modes] * len(keys)
        elif isinstance(modes, (tuple, list)):
            assert len(modes) == len(keys)
            for mode in modes:
                assert mode in self._valid_modes
        else:
            raise TypeError(f"modes must be str or tuple/list of str, but got {type(modes)}.")

        if direction not in self._valid_directions:
            raise ValueError(
                f"Direction {direction} is not supported." f"Currently support ones are {self._valid_directions}"
            )

        self.keys = keys
        self.modes = modes
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys}, flip_ratio={self.flip_ratio}, " f"direction={self.direction})"
        return repr_str

    @staticmethod
    def _flip(data, direction, mode):
        """Inplace flip an image horizontally or vertically.

        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or
                "vertical" or "diagonal".
            mode (str): Image of Flow method

        Returns:
            ndarray: The flipped image (inplace).
        """

        if direction == "horizontal":
            data = cv2.flip(data, 2)
        elif direction == "vertical":
            data = cv2.flip(data, 1)

        if mode == "flow":
            if direction == "horizontal":
                data = np.concatenate((-data[0:1], data[1:2]), 0)
            else:
                data = np.concatenate((data[0:1], -data[1:2]), 0)

        return data

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        flip = np.random.random() < self.flip_ratio
        results["flip"] = flip
        results["flip_direction"] = self.direction

        if flip:
            for key, mode in zip(self.keys, self.modes):
                if isinstance(results[key], list):
                    processed_data = []
                    for data in results[key]:
                        processed_data.append(self._flip(data, self.direction, mode))
                    results[key] = processed_data
                else:
                    results[key] = self._flip(results[key], self.direction, mode)

        return results


class RandomTransposeHW:
    """Randomly transpose images in H and W dimensions with a probability.

    [TransposeHW = horizontal flip + anti-clockwise rotation by 90 degrees]
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".
    It also supports randomly transposing a list of images.

    Args:
        keys [list[str]]: The images / flows to be transposed.
        modes [list[str]]: The operation type (image or flow) to be applied.
        transpose_ratio [float]: The probability to transpose the images.
    """

    _valid_modes = ["img", "flow"]

    def __init__(self, keys, modes, transpose_ratio=0.5):
        assert len(keys) > 0
        assert 0 <= transpose_ratio <= 1

        if isinstance(modes, str):
            assert modes in self._valid_modes
            modes = [modes] * len(keys)
        elif isinstance(modes, (tuple, list)):
            assert len(modes) == len(keys)
            for mode in modes:
                assert mode in self._valid_modes
        else:
            raise TypeError(f"modes must be str or tuple/list of str, but got {type(modes)}.")

        self.keys = keys
        self.modes = modes
        self.transpose_ratio = transpose_ratio

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys}, transpose_ratio={self.transpose_ratio})"
        return repr_str

    @staticmethod
    def _transpose(data, mode):
        data = data.transpose(0, 2, 1)

        if mode == "flow":
            data = np.concatenate((data[1:2], data[0:1]), 0)

        return data

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information."""

        transpose = np.random.random() < self.transpose_ratio
        results["transpose"] = transpose

        if transpose:
            for key, mode in zip(self.keys, self.modes):
                if isinstance(results[key], list):
                    results[key] = [self._transpose(v, mode) for v in results[key]]
                else:
                    results[key] = self._transpose(results[key], mode)

        return results


class RandomChannelReverse:
    """Reverse channels.

    Args:
        keys (list[str]): The frame lists to be reversed.
        reverse_ratio (float): The probability to reverse the frame lists.
            Default: 0.5.
    """

    def __init__(self, keys, reverse_ratio=0.5):
        assert len(keys) > 0
        assert 0 <= reverse_ratio <= 1

        self.keys = keys
        self.reverse_ratio = reverse_ratio

    @staticmethod
    def _reverse(img):
        return np.flip(img, axis=0)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys}, reverse_ratio={self.reverse_ratio})"
        return repr_str

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        reverse = np.random.random() < self.reverse_ratio
        results["channel_reverse"] = reverse

        if reverse:
            for key in self.keys:
                if isinstance(results[key], list):
                    results[key] = [self._reverse(data) for data in results[key]]
                else:
                    results[key] = self._reverse(results[key])

                old_order = results[f"{key}_channel_order"]
                results[f"{key}_channel_order"] = "rgb" if old_order == "bgr" else "bgr"

        return results


class RandomTemporalReverse:
    """Reverse frame lists for temporal augmentation.

    Required keys are the keys in attributes "lq" and "gt",
    added or modified keys are "lq", "gt" and "reverse".

    Args:
        keys [list[str]]: The frame lists to be reversed.
        reverse_ratio [float]: The probability to reverse the frame lists.
            Default: 0.5.
    """

    def __init__(self, keys, reverse_ratio=0.5):
        assert len(keys) > 0
        assert 0 <= reverse_ratio <= 1

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
        results["temporal_reverse"] = reverse

        if reverse:
            for key in self.keys:
                assert isinstance(results[key], (tuple, list))
                results[key].reverse()

        return results


def create_pipeline(pipeline):
    transforms = []
    for transform in pipeline:
        for transform_name, parameters in transform.cfg_dict.items():
            transforms.append(getattr(sys.modules[__name__], transform_name)(**parameters))

    return Compose(transforms)


def transform_ifr_plus_train(**kwargs):
    input_columns = ["img0", "img1", "gt", "flow0", "flow1", "idx", "filename"]
    output_columns = ["inputs", "target"]
    pipeline = create_pipeline(kwargs.get("pipeline", []))

    def operation(img0, img1, gt, flow0, flow1, idx, filename, batchInfo=1):
        out_inputs, out_gt = [], []
        for i in range(len(img0)):
            data = {
                "inputs": [img0[i], img1[i]],
                "inputs_channel_order": "rgb",
                "target": gt[i],
                "target_channel_order": "rgb",
                "flow": [flow0[i], flow1[i]],
            }

            inputs, target, flow = pipeline(data)

            out_inputs.append(inputs)
            out_gt.append(np.concatenate(flow + [target], axis=0))

        return out_inputs, out_gt

    return operation, input_columns, output_columns


def transform_ifr_plus_val(**kwargs):
    input_columns = ["img0", "img1", "gt", "idx", "filename"]
    output_columns = ["inputs", "target"]
    pipeline = create_pipeline(kwargs.get("pipeline", []))

    def operation(img0, img1, gt, idx, filename, batchInfo=1):
        out_inputs, out_target = [], []
        for i in range(len(img0)):
            data = {
                "inputs": [img0[i], img1[i]],
                "target": gt[i],
            }

            inputs, target = pipeline(data)

            out_inputs.append(inputs)
            out_target.append(target)

        return out_inputs, out_target

    return operation, input_columns, output_columns
