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

import numpy as np

from mindediting.dataset.transforms.basicvsr import Collect, RandomFlip
from mindediting.dataset.transforms.common import Compose, RescaleToZeroOne


def create_pipeline(pipeline):
    transforms = []
    for transform in pipeline:
        for transform_name, parameters in transform.cfg_dict.items():
            transforms.append(getattr(sys.modules[__name__], transform_name)(**parameters))
    return Compose(transforms)


class PairedRandomCrop(object):
    """Paried random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Args:
        patch_size (int): cropped patch size.
    """

    def __init__(self, gt_patch_size):
        self.patch_size = gt_patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            [dict]: A dict containing the processed data and information.
        """

        h_lq, w_lq = results["lq"].shape[-2:]
        h_gt, w_gt = results["gt"].shape[-2:]

        if h_gt != h_lq or w_gt != w_lq:
            raise ValueError(f"Shape mismatches. GT ({h_gt}, {w_gt}) and LQ ({h_lq}, {w_lq}).")
        if h_lq < self.patch_size or w_lq < self.patch_size:
            raise ValueError(
                f"LQ ({h_lq}, {w_lq}) is smaller than patch size "
                f"({self.patch_size}, {self.patch_size}). Please check "
                f'{results["lq_path"]} and {results["gt_path"]}.'
            )

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_lq - self.patch_size + 1)
        left = np.random.randint(w_lq - self.patch_size + 1)
        # crop lq patch
        results["lq"] = results["lq"][:, top : top + self.patch_size, left : left + self.patch_size]
        # crop corresponding gt patch
        results["gt"] = results["gt"][:, top : top + self.patch_size, left : left + self.patch_size]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(patch_size={self.patch_size})"
        return repr_str


class RandomTransposeHW(object):
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
                    results[key] = results[key].transpose(0, 2, 1)

        results["transpose"] = transpose
        return results


def transform_nafnet(**kwargs):
    input_columns = ["HR", "LR", "idx", "filename"]
    output_columns = ["image", "label"]
    pipeline = create_pipeline(kwargs.get("pipeline", []))

    def operation(HR, LR, idx, filename, batchInfo=None):
        if batchInfo is None:
            HR = [HR]
            LR = [LR]
        new_image, new_label = [], []
        for i in range(len(HR)):
            data = {"lq": LR[i], "gt": HR[i]}
            image, label = pipeline(data)
            new_image.append(image)
            new_label.append(label)
        if batchInfo is None:
            new_image = new_image[0]
            new_label = new_label[0]
        return new_image, new_label

    return operation, input_columns, output_columns
