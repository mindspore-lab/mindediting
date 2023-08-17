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


def unfold(x, kernel_size):
    """ipt"""
    N, C, H, W = x.shape
    numH = H // kernel_size
    numW = W // kernel_size
    if numH * kernel_size != H or numW * kernel_size != W:
        x = x[:, :, : numH * kernel_size, :, numW * kernel_size]
    output_img = np.reshape(x, (N, C, numH, kernel_size, W))

    output_img = np.transpose(output_img, (0, 1, 2, 4, 3))
    output_img = np.reshape(output_img, (N * C, numH, numW, kernel_size, kernel_size))
    output_img = np.transpose(output_img, (0, 1, 2, 4, 3))
    output_img = np.reshape(output_img, (N, C, numH * numW, kernel_size * kernel_size))
    output_img = np.transpose(output_img, (0, 2, 1, 3))
    output_img = output_img.reshape(N, numH * numW, -1)

    return output_img


def fold(x, kernel_size, output_shape=(-1, -1)):
    """ipt"""

    N, C, L = x.shape
    org_C = L // (kernel_size[0] * kernel_size[1])
    org_H = output_shape[0]
    org_W = output_shape[1]
    numH = org_H // kernel_size[0]
    numW = org_W // kernel_size[1]
    output_img = np.reshape(x, (N, C, org_C, kernel_size[0], kernel_size[1]))
    output_img = np.transpose(output_img, (0, 2, 3, 1, 4))
    output_img = np.reshape(output_img, (N * org_C, kernel_size[0], numH, numW, kernel_size[1]))
    output_img = np.transpose(output_img, (0, 2, 1, 3, 4))
    output_img = np.reshape(output_img, (N, org_C, org_H, org_W))

    return output_img


def reconstruct_numpy(x, kernel_size, output_shape=(-1, -1), stride=-1):
    x = x.transpose([0, 3, 1, 2])
    x_shape_0 = x.shape[0]
    x = np.transpose(np.reshape(x, (x_shape_0, -1, 1)), (2, 0, 1))

    """ compute"""
    if isinstance(kernel_size, (list, tuple)):
        kernel_size = kernel_size
    else:
        kernel_size = [kernel_size, kernel_size]

    if stride == -1:
        stride = kernel_size[0]
    else:
        stride = stride

    NumBlock_x = (output_shape[0] - kernel_size[0]) // stride + 1
    NumBlock_y = (output_shape[1] - kernel_size[1]) // stride + 1
    large_shape = [NumBlock_x * kernel_size[0], NumBlock_y * kernel_size[1]]

    large_x = fold(x, kernel_size, large_shape)

    N, C, _, _ = large_x.shape
    leftup_idx_x = []
    leftup_idx_y = []
    for i in range(NumBlock_x):
        leftup_idx_x.append(i * kernel_size[0])
    for i in range(NumBlock_y):
        leftup_idx_y.append(i * kernel_size[1])
    fold_x = np.zeros(
        (N, C, (NumBlock_x - 1) * stride + kernel_size[0], (NumBlock_y - 1) * stride + kernel_size[1]), dtype=np.float32
    )
    for i in range(NumBlock_x):
        for j in range(NumBlock_y):
            fold_i = i * stride
            fold_j = j * stride
            org_i = leftup_idx_x[i]
            org_j = leftup_idx_y[j]
            fills = large_x[:, :, org_i : org_i + kernel_size[0], org_j : org_j + kernel_size[1]]
            t2 = fold_x[:, :, :fold_i, fold_j : fold_j + kernel_size[1]]
            zeros2 = np.zeros(t2.shape)
            concat1 = np.concatenate((zeros2, fills), axis=2)
            t3 = fold_x[:, :, fold_i + kernel_size[0] :, fold_j : fold_j + kernel_size[1]]
            zeros3 = np.zeros(t3.shape)
            concat2 = np.concatenate((concat1, zeros3), axis=2)
            t1 = fold_x[:, :, :, :fold_j]
            zeros1 = np.zeros(t1.shape)
            concat3 = np.concatenate((zeros1, concat2), axis=3)
            t4 = fold_x[:, :, :, fold_j + kernel_size[1] :]
            zeros4 = np.zeros(t4.shape)
            concat4 = np.concatenate((concat3, zeros4), axis=3)
            fold_x += concat4
    y = fold_x.transpose([0, 2, 3, 1])
    return y


def extract_patches_numpy(x, kernel_size, stride):
    x = x.transpose([0, 3, 1, 2])
    N, C, H, W = x.shape
    leftup_idx_x = []
    leftup_idx_y = []
    nh = (H - kernel_size) // stride + 1
    nw = (W - kernel_size) // stride + 1
    for i in range(nh):
        leftup_idx_x.append(i * stride)
    for i in range(nw):
        leftup_idx_y.append(i * stride)
    NumBlock_x = len(leftup_idx_x)
    NumBlock_y = len(leftup_idx_y)
    unf_x = np.zeros((N, C, NumBlock_x * kernel_size, NumBlock_y * kernel_size), dtype=np.float32)
    for i in range(NumBlock_x):
        for j in range(NumBlock_y):
            unf_i = i * kernel_size
            unf_j = j * kernel_size
            org_i = leftup_idx_x[i]
            org_j = leftup_idx_y[j]
            fills = x[:, :, org_i : org_i + kernel_size, org_j : org_j + kernel_size]
            zeros2 = np.zeros(unf_x[:, :, :unf_i, unf_j : unf_j + kernel_size].shape)
            concat1 = np.concatenate((zeros2, fills), axis=2)
            zeros3 = np.zeros(unf_x[:, :, unf_i + kernel_size :, unf_j : unf_j + kernel_size].shape)
            concat2 = np.concatenate((concat1, zeros3), axis=2)
            zeros1 = np.zeros(unf_x[:, :, :, :unf_j].shape)
            concat3 = np.concatenate((zeros1, concat2), axis=3)
            zeros4 = np.zeros(unf_x[:, :, :, unf_j + kernel_size :].shape)
            concat4 = np.concatenate((concat3, zeros4), axis=3)
            unf_x += concat4

    y = unfold(unf_x, kernel_size)
    return y


def ntlckk2nlthwc(x, n, t, h, w, c):
    data = x.reshape([n, t, -1, c, h, w])
    data = data.transpose([0, 2, 1, 4, 5, 3])  # n,l,t,h,w,c
    data = data.reshape([-1, t, h, w, c])  # nl,t,h,w,c
    return data


def cut_forward(x, patch_size, shave):
    n, t, h, w, c = x.shape
    x = x.reshape([n * t, h, w, c])
    x_unfold = extract_patches_numpy(x, kernel_size=patch_size, stride=patch_size - shave)  # nt, l, ckk

    x_unfold = ntlckk2nlthwc(x_unfold, n, t, patch_size, patch_size, c)  # nl,t,h,w,c
    return x_unfold


def cut_backward(y_h_cut_unfold, h_raw, w_raw, h_cut, w_cut, patch_size, shave, scale, axis="h"):
    output_shape = (
        (patch_size * scale, (w_raw - w_cut) * scale) if axis == "h" else ((h_raw - h_cut) * scale, patch_size * scale)
    )  # h w
    y_h_cut = reconstruct_numpy(
        y_h_cut_unfold,
        kernel_size=patch_size * scale,
        output_shape=output_shape,
        stride=patch_size * scale - shave * scale,
    )  # n, hr, wr, c
    if axis == "h":
        # nl, h, wp, c
        y_h_cut_unfold = y_h_cut_unfold[:, :, int(shave / 2 * scale) : patch_size * scale - int(shave / 2 * scale), :]

        output_shape = (patch_size * scale, (w_raw - w_cut - shave) * scale)
        y_h_cut_inter = reconstruct_numpy(
            y_h_cut_unfold,
            kernel_size=(patch_size * scale, patch_size * scale - shave * scale),
            output_shape=output_shape,
            stride=patch_size * scale - shave * scale,
        )  # n, hr, wr, c

        concat1 = np.concatenate((y_h_cut[:, :, : int(shave / 2 * scale), :], y_h_cut_inter), axis=2)
        y_h_cut = np.concatenate(
            (concat1, y_h_cut[:, :, (w_raw - w_cut) * scale - int(shave / 2 * scale) :, :]), axis=2
        )

    else:
        y_h_cut_unfold = y_h_cut_unfold[
            :, int(shave / 2 * scale) : patch_size * scale - int(shave / 2 * scale), :, :
        ]  # nl, h, wp, c

        output_shape = ((h_raw - h_cut - shave) * scale, patch_size * scale)
        y_h_cut_inter = reconstruct_numpy(
            y_h_cut_unfold,
            kernel_size=(patch_size * scale - shave * scale, patch_size * scale),
            output_shape=output_shape,
            stride=patch_size * scale - shave * scale,
        )  # n, hr, wr, c

        concat1 = np.concatenate((y_h_cut[:, : int(shave / 2 * scale), :, :], y_h_cut_inter), axis=1)
        y_h_cut = np.concatenate(
            (concat1, y_h_cut[:, (h_raw - h_cut) * scale - int(shave / 2 * scale) :, :, :]), axis=1
        )

    return y_h_cut
