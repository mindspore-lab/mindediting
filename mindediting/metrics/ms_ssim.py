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
from scipy import signal

from mindediting.metrics.base_metrics import BaseMetric


def _f_special_gauss(size, sigma):
    r"""Return a circular symmetric gaussian kernel.
    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa
    Args:
        size (int): Size of Gaussian kernel.
        sigma (float): Standard deviation for Gaussian blur kernel.
    Returns:
        ndarray: Gaussian kernel.
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start : stop, offset + start : stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def _hox_downsample(img):
    r"""Downsample images with factor equal to 0.5.
    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa
    Args:
        img (ndarray): Images with order "NHWC".
    Returns:
        ndarray: Downsampled images with order "NHWC".
    """
    return (img[:, 0::2, 0::2, :] + img[:, 1::2, 0::2, :] + img[:, 0::2, 1::2, :] + img[:, 1::2, 1::2, :]) * 0.25


def _ssim_for_multi_scale(pred, gt, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """Calculate SSIM (structural similarity) and contrast sensitivity.
    Ref:
    Image quality assessment: From error visibility to structural similarity.
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Args:
        pred (ndarray): Images with range [0, 255] and order "NHWC".
        gt (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.
    Returns:
        tuple: Pair containing the mean SSIM and contrast sensitivity between
        `pred` and `gt`.
    """
    if pred.shape != gt.shape:
        raise RuntimeError("Input images must have the same shape (%s vs. %s)." % (pred.shape, gt.shape))
    if pred.ndim != 4:
        raise RuntimeError("Input images must have four dimensions, not %d" % pred.ndim)

    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    _, height, width, _ = pred.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_f_special_gauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(pred, window, mode="valid")
        mu2 = signal.fftconvolve(gt, window, mode="valid")
        sigma11 = signal.fftconvolve(pred * pred, window, mode="valid")
        sigma22 = signal.fftconvolve(gt * gt, window, mode="valid")
        sigma12 = signal.fftconvolve(pred * gt, window, mode="valid")
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = pred, gt
        sigma11 = pred * pred
        sigma22 = gt * gt
        sigma12 = pred * gt

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean(
        (((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)), axis=(1, 2, 3)
    )  # Return for each image individually.
    cs = np.mean(v1 / v2, axis=(1, 2, 3))
    return ssim, cs


def ms_ssim(
    pred,
    gt,
    max_val=255,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    weights=None,
    reduction="avg",
) -> np.ndarray:
    """Calculate MS-SSIM (multi-scale structural similarity).
    Ref:
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    PGGAN's implementation:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py
    Args:
        pred (ndarray): Images with range [0, 255] and order "NHWC".
        gt (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.
        weights (list): List of weights for each level; if none, use five
            levels and the weights from the original paper. Default to None.
    Returns:
        np.ndarray: MS-SSIM score between `pred` and `gt`.
    """
    if pred.shape != gt.shape:
        raise RuntimeError("Input images must have the same shape (%s vs. %s)." % (pred.shape, gt.shape))
    if pred.ndim != 4:
        raise RuntimeError("Input images must have four dimensions, not %d" % pred.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab
    # code.
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    im1, im2 = [x.astype(np.float32) for x in [pred, gt]]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim_for_multi_scale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        )
        mssim.append(ssim)
        mcs.append(cs)
        n, h, w, c = im1.shape
        pad_width = ((0, 0), (0, h % 2), (0, w % 2), (0, 0))
        im1 = np.pad(im1, pad_width, mode="constant")
        im2 = np.pad(im2, pad_width, mode="constant")
        im1, im2 = [_hox_downsample(x) for x in [im1, im2]]

    # Clip to zero. Otherwise we get NaNs.
    mssim = np.clip(np.asarray(mssim), 0.0, np.inf)
    mcs = np.clip(np.asarray(mcs), 0.0, np.inf)

    results = np.prod(mcs[:-1, :] ** weights[:-1, np.newaxis], axis=0) * (mssim[-1, :] ** weights[-1])
    if reduction == "avg":
        # Average over images only at the end.
        results = np.mean(results)
    return results


class MS_SSIM(BaseMetric):
    def calculate_metrics(self, pred, gt):
        pred, gt = self.preprocess(pred=pred, gt=gt)
        if isinstance(self.convert_to, str) and self.convert_to.lower() == "y":
            pred = np.expand_dims(pred, axis=2)
            gt = np.expand_dims(gt, axis=2)
        pred = np.expand_dims(pred, axis=0)
        gt = np.expand_dims(gt, axis=0)
        return ms_ssim(pred, gt, reduction=self.reduction)
