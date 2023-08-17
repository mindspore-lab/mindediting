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
import ast
from pprint import pformat

import yaml


def merge_dicts(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass
            else:
                print(f"replaced '{'.'.join(path + [str(key)])}': {a[key]} by {b[key]}")
                a[key] = b[key]
        else:
            print(f"added {str(key)}: {b[key]}")
            a[key] = b[key]
    return a


class Config(object):
    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)
        self.cfg_dict = cfg_dict

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return pformat(self.cfg_dict)

    def get(self, k, default=None):
        return self.cfg_dict.get(k, default)


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
    parsers = argparse.ArgumentParser(description="[REPLACE THIS at config.py]", parents=[parser])
    choices = choices if choices else {}
    helper = helper if helper else {}
    for item in cfg:
        if not isinstance(cfg[item], dict) and not isinstance(cfg[item], list):
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parsers.add_argument(
                    "--" + item, type=ast.literal_eval, default=cfg[item], choices=choice, help=help_description
                )
            else:
                parsers.add_argument(
                    "--" + item, type=type(cfg[item]), default=cfg[item], choices=choice, help=help_description
                )
    args = parsers.parse_known_args()[0]
    return args


def parse_yaml(yaml_path):
    with open(yaml_path, "r") as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [i for i in cfgs]
            cfgs_len = len(cfgs)
            if cfgs_len == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            elif cfgs_len == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif cfgs_len == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
        except ValueError:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def get_config(config_file=""):
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", type=str, default=config_file, help="Config file path")
    path_args = parser.parse_known_args()[0]
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    final_config = Config(final_config)
    return final_config
