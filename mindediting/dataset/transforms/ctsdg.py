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

from typing import Tuple

import numpy as np
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.vision.py_transforms import CenterCrop, Decode, Grayscale, Normalize, Resize, ToTensor
from mindspore.dataset.vision.utils import Inter
from skimage.color import rgb2gray
from skimage.feature import canny

input_columns = ["image", "mask"]
output_columns = ["image", "mask", "edge", "gray_image"]


def transforms_ctsdg(**kwargs):
    image_load_size = kwargs.get("image_load_size", (256, 256))
    sigma = kwargs.get("sigma", 2.0)
    image_transforms = Compose(
        [
            Decode(),
            CenterCrop((178, 178)),
            Resize(image_load_size, Inter.BILINEAR),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            np.asarray,
        ]
    )

    masks_transforms = Compose(
        [
            Decode(),
            Grayscale(),
            Resize(image_load_size, Inter.NEAREST),
            np.asarray,
        ]
    )

    def image_to_edge(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get edge and gray image of specified image

        Args:
            image: numpy image with values in [-1, 1] range

        Returns:
            Image edges and image in gray format
        """
        gray_image = rgb2gray(np.asarray(image.transpose((1, 2, 0)) * 255.0, np.uint8)).astype(np.float32)
        edge = canny(gray_image, sigma=sigma).astype(np.float32)
        return edge, gray_image

    def operation(images, masks, batchInfo=1):
        new_image, new_mask, new_edge, new_gray_image = [], [], [], []
        for i in range(len(images)):
            image = images[i]
            mask = masks[i]

            image = image_transforms(image)[0].astype(np.float32)
            edge, gray_image = image_to_edge(image)
            mask = masks_transforms(mask)[0] / 255.0

            threshold = 0.5
            ones = mask >= threshold
            mask = (1 - ones).astype(np.float32)

            mask = np.expand_dims(mask, axis=0)
            gray_image = np.expand_dims(gray_image, axis=0)
            edge = np.expand_dims(edge, axis=0)
            new_image.append(image)
            new_mask.append(mask)
            new_edge.append(edge)
            new_gray_image.append(gray_image)
        return new_image, new_mask, new_edge, new_gray_image

    return operation, input_columns, output_columns
