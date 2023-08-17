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
from deploy.data_io.data_metaclass import DataIO

T_SIZE = 25

gt_list = []


def transfer(input_path, t_size=T_SIZE):
    frame_list = []

    for img in sorted(os.listdir(input_path)):
        tmp = cv2.imread(os.path.join(input_path, img))
        frame_list.append(tmp)

        tmp1 = cv2.imread(os.path.join("/home/work/gt", img))
        gt_list.append(tmp1)

    pair_list = []
    proc_len = len(frame_list) // t_size * t_size
    for i in range(0, proc_len):
        tmp_pair = []
        for j in range(i, i + t_size):
            tmp_pair.append(frame_list[j].astype(np.float32))
        tmp_pair = np.array(tmp_pair).astype(np.float32)
        tmp_pair = np.frombuffer(tmp_pair.tobytes(), np.float32)
        pair_list.append(tmp_pair)
    return pair_list, 64, 112


def process_result(results_list):
    frame_list = []
    for i, results in enumerate(results_list):
        for result in results:
            # tmp_res = copy.deepcopy(result)
            tmp_res = result
            t = tmp_res.shape[0]
            # tmp_res = 0.5 * (tmp_res[t // 4] + tmp_res[-1 - t // 4])
            tmp_res = tmp_res[0]
            # tmp_res = np.transpose(tmp_res, (1, 2, 0))
            # tmp_res = np.clip(tmp_res * 255, 0., 255.)
            # tmp_res = cv2.cvtColor(tmp_res, cv2.COLOR_RGB2BGR, cv2.CV_32F)
            tmp_res = tmp_res.astype(np.uint8)
            frame_list.append(tmp_res)
            # print(psnr(tmp_res, gt_list[i]))
    return frame_list


def psnr(img1, img2):
    mse_value = np.mean((img1 - img2) ** 2)
    if mse_value == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse_value))


def save(output_file, output_list, save_as_video=False, save_as_frames=True):
    if save_as_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = None
        for i, img in enumerate(output_list):
            if video_writer is None:
                video_writer = cv2.VideoWriter(output_file, fourcc, 25, (img.shape[1], img.shape[0]))
            video_writer.write(img.astype(np.uint8))
        video_writer.release()
    if save_as_frames:
        for i, img in enumerate(output_list):
            n = str(i)
            while len(n) < 4:
                n = "0" + n
            save_path_i = f"./output/frame_{n}.png"
            cv2.imwrite(save_path_i, img)
    return 0


class BasicDataIO(DataIO):
    def preprocess(self, input_file):
        input_pair_list, _, _ = transfer(input_file)
        return input_pair_list

    def postprocess(self, input_data):
        output_list = process_result(input_data)
        return output_list

    def save_result(self, output_file, output_data):
        return save(output_file, output_data)
