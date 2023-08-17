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

import os

import cv2
import numpy as np
from mindspore import dataset as ds

from .dataset_base import DatasetBase

iso_list = [1600, 3200, 6400, 12800, 25600]
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]


def pack_gbrg_raw(raw):
    # pack GBRG Bayer raw to 4 channels
    black_level = 240
    white_level = 2**12 - 1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level - black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate(
        (im[1:H:2, 0:W:2, :], im[1:H:2, 1:W:2, :], im[0:H:2, 1:W:2, :], im[0:H:2, 0:W:2, :]), axis=2  # r  # gr  # b
    )  # gb
    return out


def load_cvrd_data(shift, noisy_level, scene_ind, frame_ind, xx, yy, args):
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]

    gt_name = os.path.join(
        args["input_path"],
        "indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff".format(
            scene_ind, scene_ind, iso_list[noisy_level], frame_list[frame_ind + shift]
        ),
    )
    gt_raw = cv2.imread(gt_name, -1)
    gt_raw_full = gt_raw
    gt_raw_patch = gt_raw_full[yy : yy + args["image_height"] * 2, xx : xx + args["image_width"] * 2]
    gt_raw_pack = np.expand_dims(pack_gbrg_raw(gt_raw_patch), axis=0)

    noisy_frame_index_for_current = np.random.randint(0, 10)
    input_name = os.path.join(
        args["input_path"],
        "indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy{}.tiff".format(
            scene_ind, scene_ind, iso_list[noisy_level], frame_list[frame_ind + shift], noisy_frame_index_for_current
        ),
    )
    noisy_raw = cv2.imread(input_name, -1)
    noisy_raw_full = noisy_raw
    noisy_patch = noisy_raw_full[yy : yy + args["image_height"] * 2, xx : xx + args["image_width"] * 2]
    input_pack = np.expand_dims(pack_gbrg_raw(noisy_patch), axis=0)
    return input_pack, gt_raw_pack


def load_eval_data(noisy_level, scene_ind, args):
    input_batch_list = []
    gt_raw_batch_list = []

    input_pack_list = []
    gt_raw_pack_list = []

    xx = 200
    yy = 200

    for shift in range(0, args.get("frame_num", 25)):
        # load gt raw
        frame_ind = 0
        input_pack, gt_raw_pack = load_cvrd_data(shift, noisy_level, scene_ind, frame_ind, xx, yy, args)
        input_pack_list.append(input_pack)
        gt_raw_pack_list.append(gt_raw_pack)

    input_pack_frames = np.concatenate(input_pack_list, axis=3)
    gt_raw_pack_frames = np.concatenate(gt_raw_pack_list, axis=3)

    input_batch_list.append(input_pack_frames)
    gt_raw_batch_list.append(gt_raw_pack_frames)

    input_batch = np.concatenate(input_batch_list, axis=0)
    gt_raw_batch = np.concatenate(gt_raw_batch_list, axis=0)

    in_data = input_batch.transpose((0, 3, 1, 2))
    gt_raw_data = gt_raw_batch.transpose((0, 3, 1, 2))

    return in_data, gt_raw_data


def generate_file_list(scene_list):
    file_num = 0
    data_name = []
    for scene_ind in scene_list:
        for iso in iso_list:
            for frame_ind in range(1, 8):
                gt_name = os.path.join("ISO{}/scene{}_frame{}_gt_sRGB.png".format(iso, scene_ind, frame_ind - 1))
                data_name.append(gt_name)
                file_num += 1

    random_index = np.random.permutation(file_num)
    data_random_list = []
    for i, idx in enumerate(random_index):
        data_random_list.append(data_name[idx])
    return data_random_list


def read_img(img_name, xx, yy, image_height, image_width):
    raw = cv2.imread(img_name, -1)
    raw_full = raw
    raw_patch = raw_full[yy : yy + image_height * 2, xx : xx + image_width * 2]  # 256 * 256
    raw_pack_data = pack_gbrg_raw(raw_patch)
    return raw_pack_data


def decode_data(dataset_dir, data_name, cfg):
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
    H = 1080
    W = 1920
    xx = np.random.randint(0, (W - cfg["image_width"] * 2 + 1) * 0.5) * 2
    yy = np.random.randint(0, (H - cfg["image_height"] * 2 + 1) * 0.5) * 2

    scene_ind = data_name.split("/")[1].split("_")[0]
    frame_ind = int(data_name.split("/")[1].split("_")[1][5:])
    iso_ind = data_name.split("/")[0]

    noisy_level_ind = iso_list.index(int(iso_ind[3:]))
    noisy_level = [a_list[noisy_level_ind], b_list[noisy_level_ind]]

    gt_name_list = []
    noisy_name_list = []
    xx_list = []
    yy_list = []
    image_height_list = []
    image_width_list = []
    for shift in range(0, cfg["num_frames"]):
        gt_name = os.path.join(
            dataset_dir,
            "indoor_raw_gt/indoor_raw_gt_{}/{}/{}/frame{}_clean_and_slightly_denoised.tiff".format(
                scene_ind, scene_ind, iso_ind, frame_list[frame_ind + shift]
            ),
        )

        noisy_frame_index_for_current = np.random.randint(0, 10)
        noisy_name = os.path.join(
            dataset_dir,
            "indoor_raw_noisy/indoor_raw_noisy_{}/{}/{}/frame{}_noisy{}.tiff".format(
                scene_ind, scene_ind, iso_ind, frame_list[frame_ind + shift], noisy_frame_index_for_current
            ),
        )

        gt_name_list.append(gt_name)
        noisy_name_list.append(noisy_name)

        xx_list.append(xx)
        yy_list.append(yy)

        image_height_list.append(cfg["image_height"])
        image_width_list.append(cfg["image_width"])

    gt_raw_data_list = list(map(read_img, gt_name_list, xx_list, yy_list, image_height_list, image_width_list))
    noisy_data_list = list(map(read_img, noisy_name_list, xx_list, yy_list, image_height_list, image_width_list))
    gt_raw_batch = np.concatenate(gt_raw_data_list, axis=2)
    noisy_raw_batch = np.concatenate(noisy_data_list, axis=2)

    return noisy_raw_batch, gt_raw_batch, noisy_level


class Crvd(DatasetBase):
    def __init__(self, dataset_dir, usage=None, **kwargs):
        DatasetBase.__init__(self)
        self.kwargs = kwargs
        self._index = 0
        self.dataset_dir = dataset_dir
        if usage == "train":
            scene_list = ["1", "2", "3", "4", "5", "6"]
        else:
            scene_list = ["7", "8"]
        self._data = generate_file_list(scene_list)

    def __getitem__(self, item):
        self.data_name = self._data[item]
        image, label, noisy_level = decode_data(self.dataset_dir, self.data_name, self.kwargs)
        self.image = image.transpose(2, 0, 1)
        self.label = label.transpose(2, 0, 1)
        self.noisy_level = noisy_level
        return self.image, self.label, self.noisy_level


class CrvdTest(object):
    def __init__(self, dataset_dir, usage=None, **kwargs):
        self.kwargs = kwargs
        self.scene_ind_list = kwargs.get("scene_ind", range(7, 9))  # 1~11
        self.noisy_level_list = kwargs.get("noisy_level", range(0, 5))  # 0~5
        self.in_data, self.gt_raw_data, self.noisy_level_data = self.get_data()

    def get_data(self):
        in_data_list = []
        gt_raw_data_list = []
        noisy_level_list = []
        for scene_ind in self.scene_ind_list:
            for noisy_level in self.noisy_level_list:
                in_data, gt_raw_data = load_eval_data(noisy_level, scene_ind, self.kwargs)
                in_data_list.append(in_data)
                gt_raw_data_list.append(gt_raw_data)
                noisy_level_list.append(np.full(in_data.shape[1], noisy_level))
        in_data_np = np.concatenate(in_data_list, axis=1)
        gt_raw_data_np = np.concatenate(gt_raw_data_list, axis=1)
        noisy_level_np = np.concatenate(noisy_level_list, axis=0)
        return in_data_np, gt_raw_data_np, noisy_level_np

    def __getitem__(self, time_ind):
        start = time_ind * 4
        end = (time_ind + 1) * 4
        ft1 = self.in_data[:, start:end, :, :]
        fgt = self.gt_raw_data[:, start:end, :, :]
        coeff_a = a_list[self.noisy_level_data[start]] / (2**12 - 1 - 240)
        coeff_b = b_list[self.noisy_level_data[start]] / (2**12 - 1 - 240) ** 2
        LR = ft1[0]
        HR = fgt[0]
        return HR, LR, np.array(coeff_a).astype(np.float32), np.array(coeff_b).astype(np.float32), time_ind

    def __len__(self):
        return self.in_data.shape[1] // 4


def create_crvd_dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    **kwargs
):
    if usage == "train":
        source = Crvd(dataset_dir, usage=usage, **kwargs)
        column_names = ["image", "label", "noisy_level"]
    else:
        source = CrvdTest(dataset_dir, usage=usage, **kwargs)
        column_names = ["HR", "LR", "coeff_a", "coeff_b", "idx"]
    dataset = ds.GeneratorDataset(
        source=source,
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        column_names=column_names,
        shuffle=shuffle,
    )
    return dataset
