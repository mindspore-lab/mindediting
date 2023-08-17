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

import argparse
import os
import os.path as osp
from math import ceil

import cv2
import numpy as np
from tqdm import tqdm


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


def generate_anno_file(annotation, file_name="meta_info_Vimeo90K_{}_GT.txt", root=""):
    """Generate anno file for Vimeo90K datasets from the official train list.

    Args:
        train_list (str): Train list path for Vimeo90K datasets.
        file_name (str): Saved file name. Default: 'meta_info_Vimeo90K_GT.txt'.
    """
    anno_target = "train" if "train" in annotation else "test"
    file_name = file_name.format(anno_target)
    # read official train or test list
    with open(annotation) as f:
        lines = [line.rstrip() for line in f]
    txt_file = osp.join(osp.dirname(annotation), file_name)
    with open(txt_file, "w") as f:
        for line in tqdm(lines, f"Generate annotation files {file_name}..."):
            if osp.exists(osp.join(root, line)):
                f.write(f"{line} (256, 448, 3)\n")
            else:
                p = osp.join(root, line)
                print(f"{p} skipped")


def generate_lq_samples(annotation, images_root, output_dir, scale=4):
    h_dst, w_dst = 64, 112

    with open(annotation) as f:
        train_list = [line.strip() for line in f]

    keys = []
    all_img_list = []
    for line in train_list:
        folder, sub_folder = line.split("/")
        for j in range(1, 8):
            all_img_list.append(osp.join(images_root, folder, sub_folder, f"im{j}.png"))
            keys.append("{}_{}_{}".format(folder, sub_folder, j))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    for path, key in tqdm(zip(all_img_list, keys), f"Downsampling {annotation}..."):
        if not os.path.exists(path):
            print(f"{path} not exist. Skipped")
            continue
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            print(f"{path} not found. Skipped")
            continue
        try:
            img = imresize(img, scalar_scale=1 / scale)
        except Exception:
            print(f"{path} is error! Please check it.")
        subs = key.split("_")
        save_dir = osp.join(output_dir, subs[0], subs[1])
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_path = osp.join(save_dir, osp.basename(path))
        if osp.exists(save_path):
            print(f"Image {save_path} already exists. Skipped")
        else:
            cv2.imwrite(save_path, img)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Vimeo90K datasets",
        epilog="You can download the Vimeo90K dataset from url: toflow.csail.mit.edu",
    )
    parser.add_argument("--train-annotation", help="official training list path for Vimeo90K")
    parser.add_argument("--test-annotation", help="official test list path for Vimeo90K")
    parser.add_argument("--images-root", default=None, required=True, help="GT path for Vimeo90K")
    parser.add_argument("--output-dir", default=None, help="LQ path for Vimeo90K")
    parser.add_argument("--generate-lq", action="store_true", help="generate LQ images")
    parser.add_argument("--scale", type=int, default=4, help="scale for LQ images")
    dataset_args = parser.parse_args()
    return dataset_args


if __name__ == "__main__":
    args = parse_args()
    assert args.train_annotation or args.test_annotation

    if args.train_annotation:
        generate_anno_file(args.train_annotation, root=args.images_root)
    if args.test_annotation:
        generate_anno_file(args.test_annotation, root=args.images_root)

    if args.generate_lq:
        assert args.images_root is not None and args.output_dir is not None
        if args.train_annotation:
            generate_lq_samples(args.train_annotation, args.images_root, args.output_dir, args.scale)
        if args.test_annotation:
            generate_lq_samples(args.test_annotation, args.images_root, args.output_dir, args.scale)
