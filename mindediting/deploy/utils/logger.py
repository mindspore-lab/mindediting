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

import logging
import os

from klass import Singleton


class Logger(metaclass=Singleton):
    def __init__(self, level=logging.INFO):
        self.log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        self._logger = logging.getLogger()
        self._logger.setLevel(level)
        self._logger.handlers.clear()
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.log_formatter)
        self._logger.addHandler(console_handler)
        # A flag to control whether to output to stdout.
        # Mainly used for distributed training, where only the root node will record the training.
        self._silence = False

    def add_log_file(self, log_file):
        if log_file is not None and log_file != "":
            real_path = os.path.split(os.path.realpath(log_file))[0]
            os.makedirs(real_path, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(self.log_formatter)
            self._logger.addHandler(file_handler)
            self._logger.info(f"Log file: {log_file}")

    @property
    def silence(self):
        return self._silence

    @silence.setter
    def silence(self, switch):
        self._silence = switch

    def info(self, message, force=False):
        if not self._silence or force:
            self._logger.info(message)

    def warn(self, message, force=False):
        if not self._silence or force:
            self._logger.warning(message)

    def error(self, message, force=False):
        if not self._silence or force:
            self._logger.error(message)

    def fatal(self, message, force=False):
        if not self._silence or force:
            self._logger.fatal(message)

    def debug(self, message, force=False):
        if not self._silence or force:
            self._logger.debug(message)


logger = Logger()
