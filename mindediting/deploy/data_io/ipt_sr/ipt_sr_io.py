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

import cv2
import numpy as np

from mindediting.deploy.data_io.base.image_io import image_post_transform, image_pre_transform
from mindediting.deploy.data_io.data_metaclass import DataIO
from mindediting.deploy.data_io.ipt_sr.ipt_data_process import (
    cut_backward,
    cut_forward,
    extract_patches_numpy,
    ntlckk2nlthwc,
    reconstruct_numpy,
)
from mindediting.deploy.utils.wrapper import func_time

PATCH_SIZE = 48
SHAVE = int(PATCH_SIZE / 4)
PATCH_SPLITS = None
H, W = 0, 0


def _extract_image_patches_ipt(data, num_patches_per_step=1):
    img = data
    if img.ndim != 4:
        raise Exception(f"Expect input data to have 4 dimensions, but got {data.ndim}.")

    if len(img.shape) == 4:
        img = img[None]  # n, t, h, w, c

    global H, W
    n, t, H, W, c = img.shape

    patch_size = PATCH_SIZE
    shave = SHAVE
    _h_cut = (H - patch_size) % (patch_size - shave)
    _w_cut = (W - patch_size) % (patch_size - shave)

    cuts = []
    # dealing with right-bottom corner
    x_hw_cut = img[:, :, (H - patch_size) :, (W - patch_size) :, :]
    cuts.append(x_hw_cut)

    # dealing with right/bottom border
    x_h_cut = img[..., (H - patch_size) :, :, :]  # n, t, h, w, c
    x_w_cut = img[..., (W - patch_size) :, :]
    cuts.append(cut_forward(x_h_cut, patch_size, shave))
    cuts.append(cut_forward(x_w_cut, patch_size, shave))

    # dealing with left/top border
    x_h_top = img[..., :patch_size, :, :]
    x_w_top = img[..., :patch_size, :]
    cuts.append(cut_forward(x_h_top, patch_size, shave))
    cuts.append(cut_forward(x_w_top, patch_size, shave))

    img = img.reshape([n * t, H, W, c])  # nt,h,w,c
    x_unfold = extract_patches_numpy(img, patch_size, stride=patch_size - shave)  # (nt, L, CKK)
    x_unfold = ntlckk2nlthwc(x_unfold, n, t, patch_size, patch_size, c)  # nl,t,h,w,c
    cuts.append(x_unfold)

    global PATCH_SPLITS
    PATCH_SPLITS = [cut.shape[0] for cut in cuts]

    cuts = np.concatenate(cuts, axis=0)

    num_patches = np.sum(PATCH_SPLITS)
    batch_pad = (num_patches // num_patches_per_step + 1) * num_patches_per_step - num_patches
    batch_pad = 0 if batch_pad == num_patches_per_step else batch_pad
    if batch_pad > 0:
        cuts.append(np.zeros([batch_pad, *cuts[0].shape[1:]], dtype=np.float32))
        PATCH_SPLITS.append(batch_pad)

    return np.concatenate(cuts, axis=0)


def _merge_patches_to_images_ipt(data, scale):
    data = np.array(data)

    split_mark = [sum(PATCH_SPLITS[: idx + 1]) for idx in range(len(PATCH_SPLITS) - 1)]

    patches_processed = np.split(data, split_mark, axis=0)
    y_hw_cut, y_h_cut, y_w_cut, y_h_top, y_w_top, y_unfold = patches_processed[:6]

    patch_size = PATCH_SIZE
    shave = SHAVE
    h, w = H, W
    h_cut = (h - patch_size) % (patch_size - shave)
    w_cut = (w - patch_size) % (patch_size - shave)

    h_raw = h
    w_raw = w

    y_h_cut = cut_backward(y_h_cut, h_raw, w_raw, h_cut, w_cut, patch_size, shave, scale, axis="h")
    y_w_cut = cut_backward(y_w_cut, h_raw, w_raw, h_cut, w_cut, patch_size, shave, scale, axis="w")

    y_h_top = cut_backward(y_h_top, h_raw, w_raw, h_cut, w_cut, patch_size, shave, scale, axis="h")
    y_w_top = cut_backward(y_w_top, h_raw, w_raw, h_cut, w_cut, patch_size, shave, scale, axis="w")

    output_shape = ((h - h_cut) * scale, (w - w_cut) * scale)
    y = reconstruct_numpy(
        y_unfold, kernel_size=patch_size * scale, output_shape=output_shape, stride=patch_size * scale - shave * scale
    )  # n, hr, wr, c

    y_unfold = y_unfold[
        :,
        int(shave / 2 * scale) : patch_size * scale - int(shave / 2 * scale),
        int(shave / 2 * scale) : patch_size * scale - int(shave / 2 * scale),
        :,
    ]

    output_shape = ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale)
    y_inter = reconstruct_numpy(
        y_unfold,
        kernel_size=patch_size * scale - shave * scale,
        output_shape=output_shape,
        stride=patch_size * scale - shave * scale,
    )  # n, hr, wr, c

    concat1 = np.concatenate(
        (
            y[:, : int(shave / 2 * scale), int(shave / 2 * scale) : (w - w_cut) * scale - int(shave / 2 * scale), :],
            y_inter,
        ),
        axis=1,
    )
    concat2 = np.concatenate(
        (
            concat1,
            y[
                :,
                (h - h_cut) * scale - int(shave / 2 * scale) :,
                int(shave / 2 * scale) : (w - w_cut) * scale - int(shave / 2 * scale),
                :,
            ],
        ),
        axis=1,
    )
    concat3 = np.concatenate((y[:, :, : int(shave / 2 * scale), :], concat2), axis=2)

    y = np.concatenate((concat3, y[:, :, (w - w_cut) * scale - int(shave / 2 * scale) :, :]), axis=2)

    # Remove the bottom square
    y = np.concatenate(
        (
            y[:, : y.shape[1] - int((patch_size - h_cut) / 2 * scale), :, :],
            y_h_cut[:, int((patch_size - h_cut) / 2 * scale + 0.5) :, :, :],
        ),
        axis=1,
    )

    # Remove the right square
    y_w_cat = np.concatenate(
        (
            y_w_cut[:, : y_w_cut.shape[1] - int((patch_size - h_cut) / 2 * scale), :, :],
            y_hw_cut[:, int((patch_size - h_cut) / 2 * scale + 0.5) :, :, :],
        ),
        axis=1,
    )
    y = np.concatenate(
        (
            y[:, :, : y.shape[2] - int((patch_size - w_cut) / 2 * scale), :],
            y_w_cat[:, :, int((patch_size - w_cut) / 2 * scale + 0.5) :, :],
        ),
        axis=2,
    )

    # Remove the top square
    y[:, : int((patch_size - h_cut) / 2 * scale + 0.5), : y_h_top.shape[2], :] = y_h_top[
        :, : int((patch_size - h_cut) / 2 * scale + 0.5), :, :
    ]

    # Remove the left square
    y[:, : y_w_top.shape[1], : int((patch_size - w_cut) / 2 * scale + 0.5), :] = y_w_top[
        :, :, : int((patch_size - w_cut) / 2 * scale + 0.5), :
    ]

    return y.astype(np.float32)


# video io
class IPTSRDataIO(DataIO):
    def set_input_model_shape(self, tlen, height, width):
        self.input_model_height = height
        self.input_model_width = width
        self.input_model_tlen = tlen

    def set_output_model_shape(self, height, width):
        self.output_model_height = height
        self.output_model_width = width

    def set_scale(self, scale):
        self.scale = scale
        self.target_height = None
        self.target_width = None

    def set_target_size(self, sample):
        frame_height, frame_width = sample.shape[:2]
        self.target_height = frame_height * self.scale
        self.target_width = frame_width * self.scale

    @func_time()
    def preprocess(self, input_data):
        lr = input_data[0]
        lr = image_pre_transform(lr)
        idx = np.ones(self.scale, np.int32)
        x_cuts = _extract_image_patches_ipt(lr)
        output_list = []
        for x_cut_item in x_cuts:
            output_list.append([x_cut_item[np.newaxis, :].transpose([0, 3, 1, 2]), idx])
        return output_list

    @func_time()
    def postprocess(self, input_data):
        cuts_list = []
        for input_item_list in input_data:
            for input_item in input_item_list:
                cuts_list.append(input_item[0])
        cuts_list = np.array(cuts_list)
        output_list = _merge_patches_to_images_ipt(cuts_list.transpose([0, 2, 3, 1]), self.scale)
        output_list = image_post_transform(output_list)
        return output_list

    def save_result(self, output_file, output_data):
        cv2.imwrite(output_file, output_data)
        return 0
