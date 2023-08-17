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

from mindediting.utils.dataset_utils import sub_mean


class bicubic:
    def __init__(self, seed=0):
        self.seed = seed
        self.rand_fn = np.random.RandomState(self.seed)

    @staticmethod
    def cubic(x):
        absx = np.abs(x)
        absx2 = absx**2
        absx3 = absx**3

        condition1 = (absx <= 1).astype(np.float32)
        condition2 = ((absx > 1) & (absx <= 2)).astype(np.float32)

        y = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return y

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        scale_reciprocal = 1 / scale
        if scale < 1:
            kernel_width = 4 * scale_reciprocal
        x0 = np.arange(start=1, stop=out_size[0] + 1).astype(np.float32)
        x1 = np.arange(start=1, stop=out_size[1] + 1).astype(np.float32)

        added = 0.5 * (1 - scale_reciprocal)
        u0 = x0 * scale_reciprocal + added
        u1 = x1 * scale_reciprocal + added
        kernel_width_half = kernel_width * 0.5
        left0 = np.floor(u0 - kernel_width_half)
        left1 = np.floor(u1 - kernel_width_half)

        width = np.ceil(kernel_width) + 2
        indice0 = np.expand_dims(left0, axis=1) + np.expand_dims(
            np.arange(start=0, stop=width).astype(np.float32), axis=0
        )
        indice1 = np.expand_dims(left1, axis=1) + np.expand_dims(
            np.arange(start=0, stop=width).astype(np.float32), axis=0
        )

        mid0 = np.expand_dims(u0, axis=1) - np.expand_dims(indice0, axis=0)
        mid1 = np.expand_dims(u1, axis=1) - np.expand_dims(indice1, axis=0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (np.expand_dims(np.sum(weight0, axis=2), 2))
        weight1 = weight1 / (np.expand_dims(np.sum(weight1, axis=2), 2))

        indice0 = np.expand_dims(np.minimum(np.maximum(1, indice0), in_size[0]), axis=0)
        indice1 = np.expand_dims(np.minimum(np.maximum(1, indice1), in_size[1]), axis=0)
        kill0 = np.equal(weight0, 0)[0][0]
        kill1 = np.equal(weight1, 0)[0][0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, hr, rain, lrx2, lrx3, lrx4, scales, filename, batchInfo):
        idx = self.rand_fn.randint(0, 6)
        if idx < 3:
            if idx == 0:
                scale = 0.5
                hr = lrx2
            elif idx == 1:
                scale = 1 / 3
                hr = lrx3
            elif idx == 2:
                scale = 0.25
                hr = lrx4
            hr = np.array(hr)
            [_, _, h, w] = hr.shape
            weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
            weight0 = np.asarray(weight0[0], dtype=np.float32)

            indice0 = np.asarray(indice0[0], dtype=np.float32).astype(np.long)
            weight0 = np.expand_dims(np.expand_dims(np.expand_dims(weight0, axis=0), axis=1), axis=4)
            out = hr[:, :, (indice0 - 1), :] * weight0
            out = np.sum(out, axis=3)
            A = np.transpose(out, (0, 1, 3, 2))

            weight1 = np.asarray(weight1[0], dtype=np.float32)
            weight1 = np.expand_dims(np.expand_dims(np.expand_dims(weight1, axis=0), axis=1), axis=4)
            indice1 = np.asarray(indice1[0], dtype=np.float32).astype(np.long)
            out = A[:, :, (indice1 - 1), :] * weight1
            out = np.round(255 * np.transpose(np.sum(out, axis=3), (0, 1, 3, 2))) / 255
            out = np.clip(np.round(out), 0, 255)
            lr = out
            lr = sub_mean(lr)
            hr = sub_mean(hr)
            lr = list(lr)
            hr = list(hr)
        else:
            if idx == 3:
                hr = np.array(hr)
                rain = np.array(rain)
                lr = np.clip((rain + hr), 0, 255)
                lr = sub_mean(lr)
                hr = sub_mean(hr)
                hr = list(hr)
                lr = list(lr)
            elif idx == 4:
                hr = np.array(hr)
                noise = np.random.randn(*hr.shape) * 30
                lr = np.clip(noise + hr, 0, 255)
                lr = sub_mean(lr)
                hr = sub_mean(hr)
                hr = list(hr)
                lr = list(lr)
            elif idx == 5:
                hr = np.array(hr)
                noise = np.random.randn(*hr.shape) * 50
                lr = np.clip(noise + hr, 0, 255)
                lr = sub_mean(lr)
                hr = sub_mean(hr)
                hr = list(hr)
                lr = list(lr)
        return lr, hr, [idx] * len(hr), filename


operation_train = bicubic().forward


def transforms_ipt_train(**kwargs):
    input_columns_train = ["HR", "Rain", "LRx2", "LRx3", "LRx4", "scales", "filename"]
    output_columns_train = ["LR", "HR", "idx", "filename"]
    return operation_train, input_columns_train, output_columns_train


def transforms_ipt_val(**kwargs):
    task_type = kwargs.get("task_type", "sr")
    if task_type == "sr":
        return transforms_ipt_val_sr(**kwargs)
    elif task_type == "derain":
        return transforms_ipt_val_derain(**kwargs)
    elif task_type == "denoise":
        return transforms_ipt_val_denoise(**kwargs)


def transforms_ipt_val_sr(**kwargs):
    input_columns_val = ["HR", "LR", "idx", "filename"]
    output_columns_val = ["LR", "HR"]

    def sr_task_operation(HR, LR, idx, filename, batchInfo=1):
        HR = [v.astype(np.float32) / 255.0 for v in HR]
        return LR, HR

    return sr_task_operation, input_columns_val, output_columns_val


def transforms_ipt_val_derain(**kwargs):
    input_columns_val = ["HR", "LR", "idx", "filename"]
    output_columns_val = ["LR", "HR"]

    def derain_task_operation(HR, LR, idx, filename, batchInfo=1):
        HR = [v.astype(np.float32) / 255.0 for v in HR]
        return LR, HR

    return derain_task_operation, input_columns_val, output_columns_val


def transforms_ipt_val_denoise(**kwargs):
    input_columns_val = ["HR", "LR", "idx", "filename"]
    output_columns_val = ["LR", "HR"]

    def derain_task_operation(HR, LR, idx, filename, batchInfo=1):
        original_png = [v.astype(np.float32) / 255.0 for v in HR]
        return LR, original_png

    return derain_task_operation, input_columns_val, output_columns_val
