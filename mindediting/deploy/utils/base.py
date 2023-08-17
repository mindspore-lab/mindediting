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

"""
utils, especially for python utils
"""

from yacs.config import _VALID_TYPES, CfgNode, _assert_with_logging, _valid_type

"""
config usages
"""


def convert_to_dict(cfg_node, key_list):
    if not isinstance(cfg_node, CfgNode):
        _assert_with_logging(
            _valid_type(cfg_node),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES
            ),
        )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


"""
about loading ckpts from pytorch or npy funcs
"""


def check_graph_vars(graph_vars):
    # from src.utils.utils import load_with_graph_vars
    # graph_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope=self.cfg.model.scope)

    fo = open("graph_vars.txt", "w")
    for graph_var in graph_vars:
        layer_name = graph_var.name
        is_no_need_layer = no_need_layer(layer_name)
        shape = graph_var.get_shape().as_list()
        shape = tuple(shape)
        if is_no_need_layer:
            continue
        fo.write("{} {}\n".format(layer_name, shape))
    fo.close()
    exit()


def no_need_layer(name, netname=None):
    """
    is_no_need_layer = no_need_layer(v.name, npy_file)
    if is_no_need_layer:
        continue
    """
    miss_list = ["Adam", "spectr", "_power"]
    for m in miss_list:
        if m in name:
            return True
    if netname is not None:
        if ".npy" in netname:
            netname = netname[:-4]
        if netname not in name:
            return True
    return False
