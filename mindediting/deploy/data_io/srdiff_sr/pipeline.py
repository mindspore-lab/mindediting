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
from time import time

import numpy as np
from deploy.data_io.srdiff_sr.utils_preprocess import imresize


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


class RescaleToZeroOne:
    """Transform the images into a range between 0 and 1.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys".
    It also supports rescaling a list of images.

    Args:
        keys (Sequence[str]): The images to be transformed.
    """

    def __init__(self, keys):
        assert len(keys) > 0
        self.keys = keys

    def __repr__(self):
        return self.__class__.__name__ + f"(keys={self.keys})"

    def __call__(self, results):
        """Call function.

        Args:
            results [dict]: A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information."""

        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [v.astype(np.float32) / 255.0 for v in results[key]]
            else:
                results[key] = results[key].astype(np.float32) / 255.0
        return results


class Compose:
    def __init__(self, transforms, profile=False):
        self.transforms = transforms
        self.profile = profile

    def __call__(self, x):
        for t in self.transforms:
            ts = time()
            x = t(x)
            if self.profile:
                print(f"\t{t.__class__.__name__}: {(time() - ts) * 1000:4.2f} ms")
            if x is None:
                return None
        return x

    def __len__(self):
        return len(self.transforms)


class Collect:
    def __init__(self, keys):
        assert len(keys) > 0
        self.keys = keys

    def __call__(self, results):
        return (results[key] for key in self.keys)

    def __repr__(self):
        return self.__class__.__name__ + (f"(keys={self.keys})")


def create_pipeline(pipeline, profile=False):
    transforms = []
    for transform in pipeline:
        for transform_name, parameters in transform.cfg_dict.items():
            transforms.append(getattr(sys.modules[__name__], transform_name)(**parameters))
    return Compose(transforms, profile)
