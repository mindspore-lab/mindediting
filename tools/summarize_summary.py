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
import csv
import os
from collections import defaultdict
from subprocess import run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to profile dir when profiling *.om model")
    parser.add_argument("--csv", action="store_true", help="Display results as csv")
    parser.add_argument("--run-msprof", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_msprof:
        cmd = [
            "python3",
            os.path.join(os.environ["ASCEND_TOOLKIT_HOME"], "tools/msprof/analysis/msprof/msprof.py"),
            "export",
            "summary",
            "-dir",
            args.input,
        ]
        run(cmd, check=True)

    per_op_time = defaultdict(float)
    op_name_idx = 6
    time_op_idx = 9
    col_number = None
    with open(os.path.join(args.input, "device_0/summary/op_summary_0_1_1.csv")) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        for row in csv_reader:
            if col_number is None:
                col_number = len(row)
            else:
                assert col_number == len(row)
                per_op_time[row[op_name_idx]] += float(row[time_op_idx])

    sum = 0

    for k, v in per_op_time.items():
        sum += v

    print("----------")
    print("TOTAL", f"{sum / 1000 / 1000:.3f} sec.")
    print("----------")

    if args.csv:
        for k, v in sorted(per_op_time.items(), key=lambda x: x[1], reverse=True):
            print(f"{k}, {v:.3f}, {v / sum * 100:.3f}")
    else:
        from prettytable import PrettyTable

        x = PrettyTable()

        x.field_names = ["Op", "Total time (micro sec)", "Total time (%)"]

        for k, v in sorted(per_op_time.items(), key=lambda x: x[1], reverse=True):
            x.add_row([k, f"{v:.3f}", f"{v / sum * 100:.3f}"])

        print(x)


if __name__ == "__main__":
    main()
