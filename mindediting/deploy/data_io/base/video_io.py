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
import ffmpeg
import imageio_ffmpeg
import numpy as np


def video_input(input_path, t_size=25):
    reader = imageio_ffmpeg.read_frames(input_path)
    meta = reader.__next__()
    width, height = meta["size"]
    nums_frames, duration = int(meta["fps"] * meta["duration"]), meta["duration"]
    frame_list = []
    for frame in reader:
        frame = np.frombuffer(frame, np.uint8).reshape([height, width, 3])
        frame = cv2.resize(frame, (480, 360))
        frame_list.append(frame)
    pair_list = []
    proc_len = len(frame_list) // t_size * t_size
    for i in range(0, proc_len, t_size):
        tmp_pair = []
        for j in range(i, min(i + t_size, proc_len)):
            tmp_pair.append(frame_list[j])
        tmp_pair = np.array(tmp_pair).astype(np.float32)
        tmp_pair = np.frombuffer(tmp_pair.tobytes(), np.float32)
        pair_list.append(tmp_pair)
    reader.close()
    return pair_list


# video io
class VideoReader:
    def __init__(self, input_path, windows_size=25, fixed_width=0, fixed_height=0, channel_order="rgb"):
        self.reader = cv2.VideoCapture(input_path)
        self.windows_size = windows_size
        self.width, self.height = (int(self.reader.get(3)), int(self.reader.get(4)))
        self.fps = float(self.reader.get(5))
        self.fixed_width, self.fixed_height = fixed_width, fixed_height
        if channel_order not in {"rgb", "bgr"}:
            raise ValueError(f"Invalid channel_order: {channel_order}")
        self.channel_order = channel_order

    def __del__(self):
        self.reader.release()

    def __getitem__(self, item):
        frame_list = []
        for _ in range(self.windows_size):
            ret, frame = self.reader.read()
            if not ret:
                break
            if self.channel_order == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.frombuffer(frame, np.uint8).reshape([self.height, self.width, 3])
            if self.fixed_width > 0 and self.fixed_height > 0:
                frame = cv2.resize(frame, (self.fixed_width, self.fixed_height), interpolation=cv2.INTER_AREA)
            frame_list.append(frame)
        return frame_list


class VideoWriter:
    def __init__(self, output_path, fixed_width=0, fixed_height=0, scale=1, codec="mp4v", fps=25, channel_order="rgb"):
        self.fixed_width, self.fixed_height = fixed_width * scale, fixed_height * scale

        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        if fixed_width <= 0 or fixed_height <= 0:
            raise ValueError(
                f"In video mode width and height parameters should not be zero, please, set the correct values."
            )
        self.video_writer = cv2.VideoWriter(output_path, self.fourcc, fps, (self.fixed_width, self.fixed_height))
        if channel_order not in {"rgb", "bgr"}:
            raise ValueError(f"Invalid channel_order: {channel_order}")
        self.channel_order = channel_order

    def __del__(self):
        self.video_writer.release()

    def write(self, input_data):
        if not isinstance(input_data, list):
            input_data = [input_data]
        for data_item in input_data:
            frame = data_item
            if self.channel_order == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame.astype(np.uint8))


def timestring_to_seconds(time_str):
    ftr = [3600, 60, 1]
    return sum([a * b for a, b in zip(ftr, map(float, time_str.split(":")))])


def get_video_info(filepath, video_type="normal"):
    """
    get video info
    :param filepath:
    :param video_type: normal / special
    :return:
    """
    probe = ffmpeg.probe(filepath)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    print(video_info, flush=True)
    width = int(video_info["width"])
    height = int(video_info["height"])

    if video_type == "normal":
        fps_str = video_info["avg_frame_rate"]
        duration = float(video_info["duration"])
    else:
        fps_str = video_info["r_frame_rate"]
        time_str = video_info["tags"]["DURATION"]  # 'tags': {'DURATION': '00:01:58.118000000'}
        duration = timestring_to_seconds(time_str)

    fps_list = fps_str.split("/")
    fps = float(fps_list[0]) / float(fps_list[1])
    print("fps", fps, "duration", duration)

    total_num_frames = int(fps * duration)
    if "side_data_list" in video_info:
        rotation = int(video_info["side_data_list"][0]["rotation"])
    else:
        rotation = 0
    print(
        "width",
        width,
        "height",
        height,
        "fps_str",
        fps_str,
        "total_num_frames(est.)",
        total_num_frames,
        "rotation",
        rotation,
    )
    return width, height, fps_str, total_num_frames, rotation
