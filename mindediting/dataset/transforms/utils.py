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

import numpy as np
import scipy
import scipy.signal


def add_gaussian_noise(image, stddev, seed=None):
    if seed is not None:
        np.random.seed(seed)
    assert stddev is not None
    noise = np.random.normal(0.0, stddev, image.shape).astype(image.dtype)
    lq_img = image + noise
    lq_img = np.clip(lq_img, 0, 1)
    return lq_img


def add_gaussian_blur(image, stddev, size=21):
    assert size is not None and stddev is not None
    assert image.ndim == 3
    pad = (size - 1) // 2
    blurred = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), "edge")
    kernel1d = scipy.signal.windows.gaussian(size, stddev)
    kernel2d = np.reshape(np.outer(kernel1d, kernel1d), (size, size, 1)).astype(image.dtype)
    kernel2d = kernel2d / kernel2d.sum()
    blurred = scipy.signal.fftconvolve(blurred, kernel2d, mode="valid")
    assert blurred.shape == image.shape
    return blurred
