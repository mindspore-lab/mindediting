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
import sys
import tempfile
import time
from datetime import datetime, timedelta
from subprocess import Popen, run
from typing import List

from mindediting.utils.tests import check_hbm_available, get_npu_names

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


class StatusCounter:
    def __init__(self, commands: List[str]) -> None:
        self.waiting_cmds = set(commands)
        self.running_cmds = set()
        self.done_cmds = set()
        self._max_cmd_len = len(max(commands, key=len))
        print(self)

    def __str__(self) -> str:
        line_separator = f"{'~' * (self._max_cmd_len + 10)}\n"
        str_out = line_separator
        str_out += (
            f"Status: {len(self.waiting_cmds)} waiting, {len(self.running_cmds)} running, {len(self.done_cmds)} done.\n"
        )
        for cmd in self.running_cmds:
            str_out += f"{cmd:{self._max_cmd_len}} | running \n"
        for cmd in self.waiting_cmds:
            str_out += f"{cmd:{self._max_cmd_len}} | waiting \n"
        str_out += line_separator
        return str_out

    def _list_to_str(self, cmd_list: List[str]) -> str:
        return " ".join(cmd_list)

    def process_started(self, command: List[str]) -> None:
        str_command = self._list_to_str(command)
        self.waiting_cmds.remove(str_command)
        self.running_cmds.add(str_command)
        print(self)

    def process_done(self, command: List[str]) -> None:
        str_command = self._list_to_str(command)
        self.running_cmds.remove(str_command)
        self.done_cmds.add(str_command)
        # print(self)


class TestSession:
    def __init__(self, command, stdout, stderr, env, dev_id, status_counter) -> None:
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        self.encoding = "utf-8"
        self.env = env
        self.allocated_devices = dev_id if isinstance(dev_id, (list, tuple)) else [dev_id]
        self.status_counter = status_counter

        self.process = None
        self.total_time = None
        self.__has_processed = False
        self.__has_test_passed = None

    def get_return_code(self) -> int:
        return self.process.poll()

    def start(self):
        self.process = Popen(self.command, stdout=self.stdout, stderr=self.stderr, encoding=self.encoding, env=self.env)
        self.total_time = time.time()
        self.status_counter.process_started(self.command)

    def has_processed(self) -> bool:
        return self.__has_processed

    def has_test_passed(self):
        """If return "None", then test has not been run."""
        return self.__has_test_passed

    def is_working(self) -> bool:
        return self.get_return_code() is None

    def process_session_result(self) -> None:
        self.process.kill()
        self.total_time = time.time() - self.total_time
        for dev_id in self.allocated_devices:
            if os.path.exists(f"rank_{dev_id}"):
                run(["rm", "-rf", f"rank_{dev_id}"], check=True)
        self.stdout.close()
        self.stderr.close()
        with open(self.stdout.name) as f:
            for line in f:
                print(line.rstrip())
        with open(self.stderr.name) as f:
            for line in f:
                print(line.rstrip())
        print("\n\n###############################################\n\n")
        self.__has_processed = True
        self.__has_test_passed = self.get_return_code() == 0
        self.status_counter.process_done(self.command)

    def __del__(self):
        run(["rm", self.stdout.name], check=True)
        run(["rm", self.stderr.name], check=True)


def run_tests(commands, num_devs):
    remaining_commands = commands[:]
    test_sessions = []
    device_is_free = [True for _ in range(num_devs)]
    status_counter = StatusCounter(remaining_commands)
    npu_names = get_npu_names()
    assert (
        len(npu_names) >= num_devs
    ), f"Please, check if {num_devs} cards are available at the system. {len(npu_names)}"

    time.sleep(10)

    for i in range(num_devs):
        if os.path.exists(f"rank_{i}"):
            run(["rm", "-rf", f"rank_{i}"], check=True)

    while True:
        for dev_id in range(num_devs):
            if device_is_free[dev_id] and remaining_commands:
                while check_hbm_available(npu_names[dev_id]) is False:
                    print(f"Waiting for memory freeing on device {dev_id}")
                    time.sleep(1)
                command = remaining_commands.pop().split(" ")
                my_env = os.environ.copy()
                my_env["DEVICE_ID"] = str(dev_id)
                my_env["RANK_ID"] = str(dev_id)
                test_session_name, _ = os.path.splitext(os.path.basename(command[1]))
                ts = TestSession(
                    command,
                    tempfile.NamedTemporaryFile(mode="w", delete=False, prefix=f"{test_session_name}_out_"),
                    tempfile.NamedTemporaryFile(mode="w", delete=False, prefix=f"{test_session_name}_err_"),
                    my_env,
                    dev_id,
                    status_counter,
                )
                test_sessions.append(ts)
                device_is_free[dev_id] = False
                start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                print(f'Test session "{test_session_name}" is started at {start_time}.')
                ts.start()

        for ts in test_sessions:
            if not ts.has_processed() and not ts.is_working():
                ts.process_session_result()
                for dev_id in ts.allocated_devices:
                    device_is_free[dev_id] = True

        if (not remaining_commands) and all(ts.has_processed() for ts in test_sessions):
            break
        time.sleep(10)
        sys.stdout.flush()

    max_cmd_len = len(max(commands, key=len))
    sum_return_codes = 0
    for ts in test_sessions:
        return_code = ts.get_return_code()
        sum_return_codes += return_code
        cmd = " ".join(ts.command)
        test_status = "PASSED" if return_code == 0 else "FAILED"
        msg = f"{cmd:{max_cmd_len}} : {test_status} in {ts.total_time:.2f}s ({timedelta(seconds=round(ts.total_time))})"
        print(msg)

    return sum_return_codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--type", type=str, help="run testcase type: ut or st or all", default="ut")
    parser.add_argument("--nums_of_cards", type=int, help="number of cards to run test", default=8)
    args, _ = parser.parse_known_args()
    abs_path = os.path.split(os.path.realpath(__file__))[0]
    ut_path = os.path.join(abs_path, "./ut")
    st_path = os.path.join(abs_path, "./st")

    ut_list = sorted(os.listdir(ut_path))
    st_list = sorted(os.listdir(st_path))

    ut_list = [os.path.join(abs_path, "./ut", item) for item in ut_list]
    st_list = [os.path.join(abs_path, "./st", item) for item in st_list]

    testcase_list = []
    if args.type.lower() == "all":
        testcase_list = ut_list + st_list
    elif args.type.lower() == "ut":
        testcase_list = ut_list
    elif args.type.lower() == "st":
        testcase_list = st_list

    testcase_list = [testcase for testcase in testcase_list if os.path.basename(testcase).startswith("test_")]
    testcase_list = [testcase for testcase in testcase_list if os.path.basename(testcase).endswith(".py")]
    testcase_list = [f"pytest {testcase} -v" for testcase in testcase_list]

    return_code = run_tests(testcase_list, args.nums_of_cards)
    sys.exit(return_code)
