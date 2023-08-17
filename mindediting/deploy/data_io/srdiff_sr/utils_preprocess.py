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

from __future__ import print_function

from math import ceil

import numpy as np


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def cubic(_input):
    _input = np.array(_input).astype(np.float64)
    absx = np.absolute(_input)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    absx2sub = 2.5 * absx2
    y1 = np.multiply(1.5 * absx3 - absx2sub + 1, absx <= 1)
    y2 = np.multiply(-0.5 * absx3 + absx2sub - 4 * absx + 2, (1 < absx) & (absx <= 2))
    output = y1 + y2
    return output


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        index = 1
        slice_start, slice_end = indice, i_img
        outimg_start, outimg_end = i_w, i_img
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                indice = indices[i_w, :]
                im_slice = inimg[indice, i_img].astype(np.float64)
                mult = np.multiply(np.squeeze(im_slice, axis=0), w.T)
                outimg[i_w, i_img] = np.sum(mult, axis=0)
    elif dim == 1:
        index = 0
        slice_start, slice_end = i_img, indice
        outimg_start, outimg_end = i_img, i_w

    for i_img in range(in_shape[index]):
        for i_w in range(w_shape[0]):
            w = weights[i_w, :]
            indice = indices[i_w, :]
            im_slice = inimg[slice_start, slice_end].astype(np.float64)
            mult = np.multiply(np.squeeze(im_slice, axis=0), w.T)
            outimg[outimg_start, outimg_end] = np.sum(mult, axis=0)

    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        shape = (wshape[0], wshape[2], 1, 1)
        inimg_ind = inimg[indices]
        axis = 1
    elif dim == 1:
        shape = (1, wshape[0], wshape[2], 1)
        inimg_ind = inimg[:, indices]
        axis = 2
    weights = weights.reshape(shape)
    outimg = np.sum(weights * ((inimg_ind.squeeze(axis=axis)).astype(np.float64)), axis=axis)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    imfunc = imresizemex if mode == "org" else imresizevec
    out = imfunc(A, weights, indices, dim)
    return out


def contributions(in_length, out_length, scale, kernel, k_width):
    scale_reciprocal = 1 / scale
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width * scale_reciprocal
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x * scale_reciprocal + 0.5 * (1 - scale_reciprocal)
    left = np.floor(u - kernel_width * 0.5)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def triangle(_input):
    _input = np.array(_input).astype(np.float64)
    lessthanzero = np.logical_and((_input >= -1), _input < 0)
    greaterthanzero = np.logical_and((_input <= 1), _input >= 0)
    output = np.multiply((_input + 1), lessthanzero) + np.multiply((1 - _input), greaterthanzero)
    return output


def imresize(I, scalar_scale=None, method="bicubic", output_shape=None, mode="vec"):
    if method == "bicubic":
        kernel = cubic
    elif method == "bilinear":
        kernel = triangle
    else:
        raise ValueError("unidentified kernel method supplied")

    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError("either scalar_scale OR output_shape should be defined")
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError("either scalar_scale OR output_shape should be defined")
    weights, indices = [], []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255 * B
    return np.around(B).astype(np.uint8)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), swap_red_blue=True):
    """Convert Tensors into image numpy arrays.

    After clamping to (min, max), image values will be normalized to [0, 1].

    For different tensor shapes, this function will have different behaviors:

        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.

    Note that the image channel in input tensors should be RGB order. This
    function will convert it to cv2 convention, i.e., (H x W x C) with BGR order.

    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.

    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    if not isinstance(tensor, list):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.clip(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.ndim
        if n_dim == 4:
            img_np = np.transpose(_tensor, (0, 2, 3, 1))
            if swap_red_blue and img_np.shape[-1] == 3:
                img_np = np.flip(img_np, axis=-1)
        elif n_dim == 3:
            img_np = np.transpose(_tensor, (1, 2, 0))
            if swap_red_blue and img_np.shape[-1] == 3:
                img_np = np.flip(img_np, axis=-1)
        elif n_dim == 2:
            img_np = _tensor
        else:
            raise ValueError(f"Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}")
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result
