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

import math

import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from scipy.special import gamma
from skimage import color

from mindediting.metrics.base_metrics import BaseMetric


def _cubic(x):
    """Cubic function.

    Args:
        x (np.ndarray): The distance from the center position.

    Returns:
        np.ndarray: The weight corresponding to a particular distance.
    """

    x = np.array(x, dtype=np.float32)
    x_abs = np.abs(x)
    x_abs_sq = x_abs**2
    x_abs_cu = x_abs_sq * x_abs

    # if |x| <= 1: y = 1.5|x|^3 - 2.5|x|^2 + 1
    # if 1 < |x| <= 2: -0.5|x|^3 + 2.5|x|^2 - 4|x| + 2
    f = (1.5 * x_abs_cu - 2.5 * x_abs_sq + 1) * (x_abs <= 1) + (-0.5 * x_abs_cu + 2.5 * x_abs_sq - 4 * x_abs + 2) * (
        (1 < x_abs) & (x_abs <= 2)
    )

    return f


def get_size_from_scale(input_size, scale_factor):
    """Get the output size given input size and scale factor.

    Args:
        input_size (tuple): The size of the input image.
        scale_factor (float): The resize factor.

    Returns:
        output_shape (list[int]): The size of the output image.
    """

    output_shape = [int(np.ceil(scale * shape)) for (scale, shape) in zip(scale_factor, input_size)]

    return output_shape


def get_scale_from_size(input_size, output_size):
    """Get the scale factor given input size and output size.

    Args:
        input_size (tuple(int)): The size of the input image.
        output_size (tuple(int)): The size of the output image.

    Returns:
        scale (list[float]): The scale factor of each dimension.
    """

    scale = [1.0 * output_shape / input_shape for (input_shape, output_shape) in zip(input_size, output_size)]

    return scale


def get_weights_indices(input_length, output_length, scale, kernel, kernel_width):
    """Get weights and indices for interpolation.

    Args:
        input_length (int): Length of the input sequence.
        output_length (int): Length of the output sequence.
        scale (float): Scale factor.
        kernel (func): The kernel used for resizing.
        kernel_width (int): The width of the kernel.

    Returns:
        tuple(list[np.ndarray], list[np.ndarray]): The weights and the indices
            for interpolation.
    """
    if scale < 1:  # modified kernel for antialiasing

        def h(x):
            return scale * kernel(scale * x)

        kernel_width = 1.0 * kernel_width / scale
    else:
        h = kernel
        kernel_width = kernel_width

    # coordinates of output
    x = np.arange(1, output_length + 1).astype(np.float32)

    # coordinates of input
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)  # leftmost pixel
    p = int(np.ceil(kernel_width)) + 2  # maximum number of pixels

    # indices of input pixels
    ind = left[:, np.newaxis, ...] + np.arange(p)
    indices = ind.astype(np.int32)

    # weights of input pixels
    weights = h(u[:, np.newaxis, ...] - indices - 1)

    weights = weights / np.sum(weights, axis=1)[:, np.newaxis, ...]

    # remove all-zero columns
    aux = np.concatenate((np.arange(input_length), np.arange(input_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]

    return weights, indices


def resize_along_dim(img_in, weights, indices, dim):
    """Resize along a specific dimension.

    Args:
        img_in (np.ndarray): The input image.
        weights (ndarray): The weights used for interpolation, computed from
            [get_weights_indices].
        indices (ndarray): The indices used for interpolation, computed from
            [get_weights_indices].
        dim (int): Which dimension to undergo interpolation.

    Returns:
        np.ndarray: Interpolated (along one dimension) image.
    """

    img_in = img_in.astype(np.float32)
    w_shape = weights.shape
    output_shape = list(img_in.shape)
    output_shape[dim] = w_shape[0]
    img_out = np.zeros(output_shape)

    if dim == 0:
        for i in range(w_shape[0]):
            w = weights[i, :][np.newaxis, ...]
            ind = indices[i, :]
            img_slice = img_in[ind, :]
            img_out[i] = np.sum(np.squeeze(img_slice, axis=0) * w.T, axis=0)
    elif dim == 1:
        for i in range(w_shape[0]):
            w = weights[i, :][:, :, np.newaxis]
            ind = indices[i, :]
            img_slice = img_in[:, ind]
            img_out[:, i] = np.sum(np.squeeze(img_slice, axis=1) * w.T, axis=1)

    if img_in.dtype == np.uint8:
        img_out = np.clip(img_out, 0, 255)
        return np.around(img_out).astype(np.uint8)
    else:
        return img_out


class MATLABLikeResize:
    """Resize the input image using MATLAB-like downsampling.

    Currently support bicubic interpolation only. Note that the output of
    this function is slightly different from the official MATLAB function.

    Required keys are the keys in attribute "keys". Added or modified keys
    are "scale" and "output_shape", and the keys in attribute "keys".

    Args:
        keys (list[str]): A list of keys whose values are modified.
        scale (float | None, optional): The scale factor of the resize
            operation. If None, it will be determined by output_shape.
            Default: None.
        output_shape (tuple(int) | None, optional): The size of the output
            image. If None, it will be determined by scale. Note that if
            scale is provided, output_shape will not be used.
            Default: None.
        kernel (str, optional): The kernel for the resize operation.
            Currently support 'bicubic' only. Default: 'bicubic'.
        kernel_width (float): The kernel width. Currently support 4.0 only.
            Default: 4.0.
    """

    def __init__(self, keys, scale=None, output_shape=None, kernel="bicubic", kernel_width=4.0):

        if kernel.lower() != "bicubic":
            raise ValueError("Currently support bicubic kernel only.")

        if float(kernel_width) != 4.0:
            raise ValueError("Current support only width=4 only.")

        if scale is None and output_shape is None:
            raise ValueError('"scale" and "output_shape" cannot be both None')

        self.kernel_func = _cubic
        self.keys = keys
        self.scale = scale
        self.output_shape = output_shape
        self.kernel = kernel
        self.kernel_width = kernel_width

    def _resize(self, img):
        """resize an image to the require size.

        Args:
            img (np.ndarray): The original image.
        Returns:
            output (np.ndarray): The resized image.
        """
        weights = {}
        indices = {}

        # compute scale and output_size
        if self.scale is not None:
            scale = float(self.scale)
            scale = [scale, scale]
            output_size = get_size_from_scale(img.shape, scale)
        else:
            scale = get_scale_from_size(img.shape, self.output_shape)
            output_size = list(self.output_shape)

        # apply cubic interpolation along two dimensions
        order = np.argsort(np.array(scale))
        for k in range(2):
            key = (img.shape[k], output_size[k], scale[k], self.kernel_func, self.kernel_width)
            weight, index = get_weights_indices(
                img.shape[k], output_size[k], scale[k], self.kernel_func, self.kernel_width
            )
            weights[key] = weight
            indices[key] = index

        output = np.copy(img)
        if output.ndim == 2:  # grayscale image
            output = output[:, :, np.newaxis]

        for k in range(2):
            dim = order[k]
            key = (img.shape[dim], output_size[dim], scale[dim], self.kernel_func, self.kernel_width)
            output = resize_along_dim(output, weights[key], indices[key], dim)

        return output


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (np.ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1) ** 2)
    array_position = np.argmin((r_gam - rhatnorm) ** 2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute features.

    Args:
        block (np.ndarray): 2D Image block.

    Returns:
        feat (List): Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe_core(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=16, block_size_w=16):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implementation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (np.ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (np.ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (np.ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value). Default: 96.
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value). Default: 96.

    Returns:
        np.ndarray: NIQE quality.
    """
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0 : num_block_h * block_size_h, 0 : num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode="nearest")

        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode="nearest") - np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process each block
                block = img_nomalized[
                    idx_h * block_size_h // scale : (idx_h + 1) * block_size_h // scale,
                    idx_w * block_size_w // scale : (idx_w + 1) * block_size_w // scale,
                ]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        # matlab-like bicubic downsample with anti-aliasing
        if scale == 1:
            resize = MATLABLikeResize(keys=None, scale=0.5)
            img = resize._resize(img[:, :, np.newaxis] / 255.0)[:, :, 0] * 255.0

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)

    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param), np.transpose((mu_pris_param - mu_distparam))
    )

    return np.squeeze(np.sqrt(quality))


class NIQE(BaseMetric):
    def __init__(
        self,
        reduction="avg",
        crop_border=0,
        input_order="HWC",
        convert_to=None,
        process_middle_image=(False, False),
        model_path="",
    ):
        super().__init__(reduction, crop_border, input_order, convert_to, process_middle_image)
        self.model_path = model_path

    def calculate_metrics(self, pred, gt):
        pred, gt = self.preprocess(pred=pred, gt=gt)
        pred = pred.round()
        niqe_pris_params = np.load(self.model_path)
        mu_pris_param = niqe_pris_params["mu_pris_param"]
        cov_pris_param = niqe_pris_params["cov_pris_param"]
        gaussian_window = niqe_pris_params["gaussian_window"]
        if self.convert_to is None:
            pred = color.rgb2gray(pred)
        niqe_result = niqe_core(pred, mu_pris_param, cov_pris_param, gaussian_window)
        return niqe_result
