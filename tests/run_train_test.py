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
from subprocess import run

from run_test import StatusCounter, TestSession, check_hbm_available, get_npu_names

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grep", default="tests/tt/")
    parser.add_argument("--no-run", action="store_true")
    return parser.parse_args()


def replace(s, what, by):
    new_s = ""
    for c in s:
        new_s += by if c in what else c
    return new_s


def run_tests(commands):
    remaining_commands = commands[:]
    test_sessions = []
    status_counter = StatusCounter(remaining_commands)
    npu_names = get_npu_names()
    dev_ids = list(range(len(npu_names)))

    time.sleep(10)

    for i in dev_ids:
        if os.path.exists(f"rank_{i}"):
            run(["rm", "-rf", f"rank_{i}"], check=True)

    while remaining_commands:
        while not all(check_hbm_available(npu_names[dev_id]) for dev_id in dev_ids):
            print(f"Waiting for memory freeing on devices")
            time.sleep(10)
        command = remaining_commands.pop().split(" ")
        my_env = os.environ.copy()
        test_session_name = replace(command[1], ("/", ".", "[", "]", ":", ".", "-"), "_")
        ts = TestSession(
            command,
            tempfile.NamedTemporaryFile(mode="w", delete=False, prefix=f"{test_session_name}_out_"),
            tempfile.NamedTemporaryFile(mode="w", delete=False, prefix=f"{test_session_name}_err_"),
            my_env,
            dev_ids,
            status_counter,
        )
        test_sessions.append(ts)
        start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f'Test session "{test_session_name}" is started at {start_time}.')
        ts.start()

        sys.stdout.flush()

        while not all(ts.has_processed() for ts in test_sessions):
            for ts in test_sessions:
                if not ts.has_processed() and not ts.is_working():
                    ts.process_session_result()
            time.sleep(10)

    max_cmd_len = len(max(commands, key=len))
    sum_return_codes = 0
    for ts in test_sessions:
        return_code = ts.get_return_code()
        sum_return_codes += return_code
        cmd = " ".join(ts.command)
        test_status = "PASSED" if return_code == 0 else "FAILED"
        msg = f"{cmd:{max_cmd_len}} : {test_status} in {ts.total_time:.2f}s ({timedelta(seconds=round(ts.total_time))})"
        print(msg)

    print(f"\t The test session has done. Memory check...")
    while not all(check_hbm_available(npu_names[dev_id]) for dev_id in dev_ids):
        print(f"Waiting for memory freeing on devices")
        time.sleep(10)

    return sum_return_codes


if __name__ == "__main__":
    args = parse_args()
    abs_path = os.path.split(os.path.realpath(__file__))[0]
    tt_path = os.path.join(abs_path, "./tt")
    testcase_list = (
        run(f"pytest tests/tt/ --collect-only -q", shell=True, capture_output=True).stdout.decode().strip().split("\n")
    )

    filtered_testcase_list = [testcase for testcase in testcase_list if args.grep in testcase]

    if filtered_testcase_list:
        filtered_testcase_list = [f"pytest {testcase} -v" for testcase in filtered_testcase_list]
        if args.no_run:
            sys.exit(0)
        return_code = run_tests(filtered_testcase_list)
        sys.exit(return_code)
    else:
        print("\n".join(testcase_list))
        sys.exit(1)
