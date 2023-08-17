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

from deploy.tasks.task_metaclass import Task


class TaskHandler:
    def __init__(self, task: Task):
        self.next_handler = None
        self.task = task
        self.input_data_list = []

    def set_next_handler(self, handler):
        self.next_handler = handler

    def execute(self, input_data, **kwargs):
        if isinstance(input_data, list):
            self.input_data_list.extend(input_data)
        else:
            self.input_data_list.append(input_data)
        if len(self.input_data_list) >= self.task.once_process_frames:
            output_data_list = self.task.run(self.input_data_list, **kwargs)
            if self.task.frame_overlap > 0:
                self.input_data_list = self.input_data_list[-self.task.frame_overlap :]
            else:
                self.input_data_list = []
            if self.next_handler:
                return self.next_handler.execute(output_data_list, **kwargs)
            else:
                return output_data_list
        return None


class Pipeline(object):
    def __init__(self, task_list=None):
        if not task_list:
            raise Exception("There are no task to execute!")
        handler_list = []
        for task in task_list:
            handler_list.append(TaskHandler(task))
        for i in range(len(handler_list) - 1):
            handler_list[i].next_handler = handler_list[i + 1]
        self.handler = handler_list[0]

    def run(self, input_data, **kwargs):
        return self.handler.execute(input_data, **kwargs)
