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

import argparse
import os
from typing import List, Union

from deploy.data_io.base.image_io import ImageReader, ImageWriter
from deploy.data_io.base.video_io import VideoReader, VideoWriter
from deploy.tasks.task_manager import Pipeline
from deploy.tasks.task_metaclass import Task
from deploy.utils.config import parse_yaml
from munch import DefaultMunch


def parse_args():
    parser = argparse.ArgumentParser(description="pipeline for low level vision kit")
    parser.add_argument("-pf", "--pipeline_file", help="pipeline file path")
    args = parser.parse_args()
    return args


def get_tasks(cfg: dict) -> List[Task]:
    tasks_queue = []
    for sub_task_dict in cfg["tasks"]:
        cfg = DefaultMunch.fromDict(sub_task_dict)
        task = Task(cfg)
        if task.backend is None and task.data_io is None:
            raise Exception
        tasks_queue.append(task)

    return tasks_queue


def get_reader(cfg: dict):
    if cfg["task_type"] == "video":
        cfg = DefaultMunch.fromDict(cfg)
        return VideoReader(cfg.input_file, cfg.windows_size, cfg.fixed_width, cfg.fixed_height, cfg.color_space)
    elif cfg["task_type"] == "image":
        cfg = DefaultMunch.fromDict(cfg)
        return ImageReader(cfg.input_file, cfg.windows_size, cfg.fixed_width, cfg.fixed_height, cfg.color_space)


def get_writer(cfg: dict, reader: Union[ImageReader, VideoReader]):
    if cfg["task_type"] == "video":
        cfg = DefaultMunch.fromDict(cfg)
        output_fps = (cfg.fps_multiplier or 1.0) * getattr(reader, "fps", 25)
        return VideoWriter(
            cfg.output_file,
            cfg.fixed_width,
            cfg.fixed_height,
            cfg.up_scale,
            fps=output_fps,
            channel_order=cfg.color_space,
        )
    elif cfg["task_type"] == "image":
        cfg = DefaultMunch.fromDict(cfg)
        return ImageWriter(cfg.output_file, channel_order=cfg.color_space)


def deploy(cfg) -> None:
    if os.getenv("ASCEND_TOOLKIT_HOME") is None:
        toolkit_path = "/usr/local/Ascend/ascend-toolkit/latest/"
        if not os.path.isdir(toolkit_path):
            print("Please, set ASCEND_TOOLKIT_HOME environment variable. Exit.")
            return
        os.environ["ASCEND_TOOLKIT_HOME"] = toolkit_path
        print(f"Set ASCEND_TOOLKIT_HOME={toolkit_path}")
    task_list = get_tasks(cfg)
    reader = get_reader(cfg)
    writer = get_writer(cfg, reader)
    pipeline = Pipeline(task_list)

    for input_data in reader:
        if not input_data:
            break
        output_data_list = pipeline.run(input_data)
        if output_data_list is not None:
            writer.write(output_data_list)


if __name__ == "__main__":
    args = parse_args()
    default, helper, choices = parse_yaml(args.pipeline_file)
    cfg = DefaultMunch.fromDict(default)
    deploy(cfg)
