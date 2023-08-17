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

import copy
import sys
from time import time

import cv2
import numpy as np
from mindspore.dataset.transforms import Compose
from PIL import Image

from mindediting.dataset.src.vimeo_preprocess import imresize
from mindediting.dataset.transforms.common import Compose, RescaleToZeroOne


def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.0
    else:
        raise TypeError("The img type should be np.float32 or np.uint8, " f"but got {img_type}")
    return img


def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError("The dst_type should be np.float32 or np.uint8, " f"but got {dst_type}")
    img = img.round() if dst_type == np.uint8 else img / 255.0
    return img.astype(dst_type)


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
    img_type = image.dtype
    image = _convert_input_type_range(image)
    if y_only:
        out_img = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]
        ) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def imflip_(img, direction="horizontal"):
    """Inplace flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or "vertical" or "diagonal".

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


class LoadImageFromFile:
    def __init__(self, key="gt", channel_order="bgr", convert_to=None, use_cache=False, **kwargs):
        self.key = key
        self.channel_order = channel_order
        self.convert_to = convert_to
        self.kwargs = kwargs
        self.use_cache = use_cache
        self.cache = dict() if use_cache else None
        self.archive = None

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"key={self.key}, channel_order={self.channel_order}, use_cache={self.use_cache})"
        return repr_str

    def __call__(self, results):
        filepath = str(results[f"{self.key}_path"])
        if self.use_cache:
            if filepath in self.cache:
                img = self.cache[filepath]
            else:
                img = Image.open(filepath).convert(self.channel_order.upper())
                self.cache[filepath] = img
        else:
            img = Image.open(filepath).convert(self.channel_order.upper())
        img = np.array(img)

        if self.convert_to is not None:
            if self.channel_order == "bgr" and self.convert_to.lower() == "y":
                img = bgr2ycbcr(img, y_only=True)
            elif self.channel_order == "rgb":
                img = rgb2ycbcr(img, y_only=True)
            else:
                raise ValueError('Currently support only "bgr2ycbcr" or ' '"bgr2ycbcr".')
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

        results[self.key] = img
        results[f"{self.key}_path"] = filepath
        results[f"{self.key}_ori_shape"] = img.shape
        return results


class LoadImageFromFileList(LoadImageFromFile):
    """Load image from file list.

    It accepts a list of path and read each frame from each path.
    A numpy.ndarray with a NCHW layout is returned.

    Args:
        key (str): Key in results to find corresponding path. Default: 'gt'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
    """

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information.
        """
        img_paths = results[self.key]
        imgs = []

        for img_path in img_paths:
            img = cv2.imread(img_path)
            imgs.append(img)
        imgs = np.stack(imgs)
        # To NCHW
        imgs = imgs.transpose((0, 3, 1, 2))
        if self.channel_order == "rgb":
            imgs = np.flip(imgs, axis=1)

        results[self.key] = imgs
        results[f"{self.key}_path"] = np.asarray(img_paths)

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
        flip_ratio [float]: The propability to flip the images.
        direction [str]: Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """

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
            results [dict]: A dict containing the necessary information and data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information.
        """
        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                for v in results[key]:
                    imflip_(v, self.direction)
        results["flip"] = flip
        results["flip_direction"] = self.direction

        return results


class RandomMultidirectionalFlip:
    def __init__(self, keys, horizontal_flip_ratio=0.5, vertical_flip_ratio=0.5, diagonal_flip_ratio=0.5):
        self.keys = keys
        self.horizontal_flip_ratio = horizontal_flip_ratio
        self.vertical_flip_ratio = vertical_flip_ratio
        self.diagonal_flip_ratio = diagonal_flip_ratio

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            "("
            + ", ".join(
                f"keys={self.keys}",
                f"horizontal_flip_ratio={self.horizontal_flip_ratio}",
                f"vertical_flip_ratio={self.vertical_flip_ratio}",
                f"diagonal_flip_ratio={self.diagonal_flip_ratio}",
            )
            + ")"
        )
        return repr_str

    def __call__(self, results):
        flip_sample = np.random.random(3)
        for key in self.keys:
            x = results[key]
            if flip_sample[0] < self.horizontal_flip_ratio:
                x = np.flip(x, axis=3)
            if flip_sample[1] < self.vertical_flip_ratio:
                x = np.flip(x, axis=2)
            if flip_sample[2] < self.diagonal_flip_ratio:
                x = np.swapaxes(x, 2, 3)
            results[key] = x
        return results


class RandomCrop:
    """Random crop.

    Args:
        key [str]: results dict key to process.
        patch_size [int]: cropped patch size.
    """

    def __init__(self, key, patch_size):
        self.key = key
        self.patch_size = patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information.
        """
        h, w = results[self.key].shape[-2:]

        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Data size ({h}x{w}) is smaller than patch size " f"({self.patch_size}x{self.patch_size})."
            )

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h - self.patch_size + 1)
        left = np.random.randint(w - self.patch_size + 1)
        # crop patch
        results[self.key] = results[self.key][:, :, top : top + self.patch_size, left : left + self.patch_size]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(key={self.key}, patch_size={self.patch_size})"
        return repr_str


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

        if isinstance(results["lq"], list):
            results["lq"] = np.concatenate(np.expand_dims(results["lq"], axis=0))
        if isinstance(results["gt"], list):
            results["gt"] = np.concatenate(np.expand_dims(results["gt"], axis=0))

        h_lq, w_lq = results["lq"].shape[-2:]
        h_gt, w_gt = results["gt"].shape[-2:]

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

        if len(results["lq"].shape) == 4:
            results["lq"] = results["lq"][:, :, top : top + lq_patch_size, left : left + lq_patch_size]
        else:
            results["lq"] = results["lq"][:, top : top + lq_patch_size, left : left + lq_patch_size]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        if len(results["lq"].shape) == 4:
            results["gt"] = results["gt"][
                :, :, top_gt : top_gt + self.gt_patch_size, left_gt : left_gt + self.gt_patch_size
            ]
        else:
            results["gt"] = results["gt"][
                :, top_gt : top_gt + self.gt_patch_size, left_gt : left_gt + self.gt_patch_size
            ]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(gt_patch_size={self.gt_patch_size})"
        return repr_str


class RandomTransposeHW:
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees)
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
            [dict]: A dict containing the processed data and information.
        """
        transpose = np.random.random() < self.transpose_ratio

        if transpose:
            for key in self.keys:
                if isinstance(results[key], list):
                    results[key] = [v.transpose(0, 2, 1) for v in results[key]]
                else:
                    if len(results[key].shape) == 4:
                        results[key] = results[key].transpose(0, 1, 3, 2)
                    elif len(results[key].shape) == 3:
                        results[key] = results[key].transpose(0, 2, 1)
                    else:
                        raise ValueError

        results["transpose"] = transpose
        return results


class MirrorSequence:
    """Extend short sequences (e.g. Vimeo-90K) by mirroring the sequences.

    Given a sequence with N frames (x1, ..., xN), extend the sequence to
    (x1, ..., xN, xN, ..., x1).

    Args:
        keys (list[str]): The frame lists to be extended.
    """

    def __init__(self, keys):
        self.keys = keys

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys})"
        return repr_str

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information.
        """
        for key in self.keys:
            results[key] = np.concatenate((results[key], results[key][::-1]))
        return results


class CopyValues:
    """Copy the value of a source key to a destination key.

    It does the following: results[dst_key] = results[src_key] for
    (src_key, dst_key) in zip(src_keys, dst_keys).

    Added keys are the keys in the attribute "dst_keys".

    Args:
        src_keys (list[str]): The source keys.
        dst_keys (list[str]): The destination keys.
    """

    def __init__(self, src_keys, dst_keys):

        if not isinstance(src_keys, list) or not isinstance(dst_keys, list):
            raise AssertionError('"src_keys" and "dst_keys" must be lists.')

        if len(src_keys) != len(dst_keys):
            raise ValueError('"src_keys" and "dst_keys" should have the same' "number of elements.")

        self.src_keys = src_keys
        self.dst_keys = dst_keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with a key added/modified.
        """
        for (src_key, dst_key) in zip(self.src_keys, self.dst_keys):
            results[dst_key] = copy.deepcopy(results[src_key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(src_keys={self.src_keys}, dst_keys={self.dst_keys})"
        return repr_str


class RandomNoise:
    """Apply random noise to the input.

    Currently support Gaussian noise and Poisson noise.

    Modified keys are the attributed specified in "keys".

    Args:
        params (dict): A dictionary specifying the degradation settings.
        keys (list[str]): A list specifying the keys whose values are
            modified.
    """

    def __init__(self, params, keys):
        self.keys = keys
        self.params = params
        self.prob = params.get("prob", 1)
        self.gaussian_gray_noise_prob = self.params.get("gaussian_gray_noise_prob", 0)
        self.rng = np.random.default_rng()

    def _is_gaussian_gray_noise(self):
        if self.gaussian_gray_noise_prob > 0:
            return np.random.uniform() < self.gaussian_gray_noise_prob
        else:
            return False

    def _apply_gaussian_noise(self, imgs):
        sigma_range = self.params["gaussian_sigma"]
        sigma = np.random.uniform(sigma_range[0], sigma_range[1]) / 255.0

        sigma_step = self.params.get("gaussian_sigma_step", 0)

        n, c, h, w = imgs.shape

        if sigma_step == 0:
            if self._is_gaussian_gray_noise():
                noise = self.rng.standard_normal(size=(n, 1, h, w), dtype=np.float32) * sigma
            else:
                noise = self.rng.standard_normal(size=(n, c, h, w), dtype=np.float32) * sigma
            imgs += noise
        else:
            for img in imgs:
                if self._is_gaussian_gray_noise():
                    noise = self.rng.standard_normal(size=(1, h, w), dtype=np.float32) * sigma
                else:
                    noise = self.rng.standard_normal(size=(c, h, w), dtype=np.float32) * sigma
                img += noise

                # update noise level
                sigma += np.random.uniform(-sigma_step, sigma_step) / 255.0
                sigma = np.clip(sigma, sigma_range[0] / 255.0, sigma_range[1] / 255.0)

        return imgs

    def _apply_poisson_noise(self, imgs):
        scale_range = self.params["poisson_scale"]
        scale = np.random.uniform(scale_range[0], scale_range[1])

        scale_step = self.params.get("poisson_scale_step", 0)

        gray_noise_prob = self.params["poisson_gray_noise_prob"]
        is_gray_noise = np.random.uniform() < gray_noise_prob

        for img in imgs:
            noise = img.copy()
            if is_gray_noise:
                noise = cv2.cvtColor(np.transpose(noise, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
                noise = noise[np.newaxis, ...]
            noise = np.clip((noise * 255.0).round(), 0, 255) / 255.0
            unique_val = 2 ** np.ceil(np.log2(len(np.unique(noise))))
            noise = np.random.poisson(noise * unique_val) / unique_val - noise

            img += noise * scale

            # update noise level
            scale += np.random.uniform(-scale_step, scale_step)
            scale = np.clip(scale, scale_range[0], scale_range[1])

        return imgs

    def _apply_random_noise(self, imgs):

        noise_types = self.params["noise_type"]
        if len(noise_types) > 1:
            noise_type = np.random.choice(noise_types, p=self.params["noise_prob"])
        else:
            noise_type = noise_types[0]

        if noise_type.lower() == "gaussian":
            imgs = self._apply_gaussian_noise(imgs)
        elif noise_type.lower() == "poisson":
            imgs = self._apply_poisson_noise(imgs)
        else:
            raise NotImplementedError(f'"noise_type" [{noise_type}] is ' "not implemented.")
        return imgs

    def __call__(self, results):
        if self.prob < 1 and np.random.uniform() > self.prob:
            return results

        for key in self.keys:
            results[key] = self._apply_random_noise(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(params={self.params}, keys={self.keys})"
        return repr_str


class GenerateSegmentIndices:
    """Generate frame indices for a segment. It also performs temporal
    augmention with random interval.

    Required keys: lq_path, gt_path, key, num_input_frames, sequence_length
    Added or modified keys:  lq_path, gt_path, interval, reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        start_idx (int): The index corresponds to the first frame in the
            sequence. Default: 0.
        filename_tmpl (str): Template for file name. Default: '{:08d}.png'.
    """

    def __init__(self, keys, num_frames, interval_list, random=True):
        self.keys = keys
        self.interval_list = interval_list
        self.num_frames = num_frames
        self.random = random

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        interval = np.random.choice(self.interval_list)

        original_sequence_length = len(results[self.keys[0]])
        for key in self.keys:
            assert len(results[key]) == original_sequence_length

        # randomly select a frame as start
        if original_sequence_length - self.num_frames * interval < 0:
            raise ValueError(
                f"The input sequence is not long enough [{original_sequence_length}] to "
                f"support the current choice of interval [{interval}] or "
                f"num_input_frames [{self.num_frames}]."
            )
        if self.random:
            start_frame_idx = np.random.randint(0, original_sequence_length - self.num_frames * interval + 1)
        else:
            start_frame_idx = 0
        end_frame_idx = start_frame_idx + self.num_frames * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        for key in self.keys:
            x = results[key]
            x = list(x[i] for i in neighbor_list)
            results[key] = np.asarray(x)
        results["interval"] = interval

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(interval_list={self.interval_list})"
        return repr_str


class Pad:
    def __init__(self, keys, divisor=1):
        self.keys = keys
        self.divisor = divisor

    @staticmethod
    def pad(img, divisor):
        n, c, h, w = img.shape
        pad_h = (h + divisor - 1) // divisor * divisor - h
        pad_w = (w + divisor - 1) // divisor * divisor - w
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
        return img

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.pad(results[key], self.divisor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys}, divisor={self.divisor})"
        return repr_str


class Collect:
    """Stacks the result data with selected keys into the tuple.

    Args:
        keys (list[str]): The frame lists to be stacked.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A tuple containing the processed data.
        """
        output = []
        for key in self.keys:
            if isinstance(results[key], list):
                output.append(np.transpose(np.stack(results[key], axis=-1), (3, 0, 1, 2)).astype(np.float32))
            else:
                output.append(results[key].astype(np.float32))
        return tuple(output)


class Normalize:
    def __init__(self, keys, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.keys = keys
        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std = np.array(std, dtype=np.float32)[:, None, None]

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [(x - self.mean) / self.std for x in results[key]]
            else:
                results[key] = (results[key] - self.mean) / self.std
        return results


class Resize:
    def __init__(self, in_key, out_key=None, scale=4):
        self.in_key = in_key
        self.out_key = out_key
        self.scale = scale

    def __call__(self, results):
        if isinstance(results[self.in_key], list):
            resized = [
                imresize(x.transpose(1, 2, 0), scalar_scale=self.scale).transpose(2, 0, 1) for x in results[self.in_key]
            ]
        else:
            resized = imresize(results[self.in_key].transpose(1, 2, 0), scalar_scale=self.scale).transpose(2, 0, 1)
        if self.out_key is not None:
            results[self.out_key] = resized
        else:
            results[self.in_key] = resized
        return results


def create_pipeline(pipeline, profile=False):
    transforms = []
    for transform in pipeline:
        for transform_name, parameters in transform.cfg_dict.items():
            transforms.append(getattr(sys.modules[__name__], transform_name)(**parameters))
    return Compose(transforms, profile)


def transform_basicvsr(pipeline=(), scale=4, profile_preproc_pipeline=False, **kwargs):
    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["image", "label"]
    pipeline = create_pipeline(pipeline, profile_preproc_pipeline)

    def operation(HR, LR, idx, filename, batchInfo=None):
        t = time()
        if batchInfo is None:
            HR = [HR]
            LR = [LR]
        new_image, new_label = [], []
        for i in range(len(HR)):
            data = {"lq": LR[i], "gt": HR[i], "scale": scale}
            image, label = pipeline(data)
            new_image.append(image)
            new_label.append(label)
        if batchInfo is None:
            new_image = new_image[0]
            new_label = new_label[0]
        if profile_preproc_pipeline:
            print(f"data preproc time {(time() - t) * 1000:4.2f} ms")
        return new_image, new_label

    return operation, input_columns, output_columns


transform_basicvsr_train = transform_basicvsr
transform_basicvsr_val = transform_basicvsr
