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

from .transforms.basicvsr import transform_basicvsr_train, transform_basicvsr_val
from .transforms.basicvsr_plus_plus_light import (
    transform_basicvsr_plus_plus_light_train,
    transform_basicvsr_plus_plus_light_val,
)
from .transforms.ctsdg import transforms_ctsdg
from .transforms.emvd import transforms_emvd_train, transforms_emvd_val
from .transforms.fsrcnn import transform_fsrcnn
from .transforms.ifr_plus import transform_ifr_plus_train, transform_ifr_plus_val
from .transforms.ipt import transforms_ipt_train, transforms_ipt_val
from .transforms.mimo_unet import transforms_mimo_unet
from .transforms.mpfer import transform_mpfer
from .transforms.nafnet import transform_nafnet
from .transforms.noahtcv import transform_noahtcv_train, transform_noahtcv_val
from .transforms.rrdb import transform_rrdb_train, transform_rrdb_val
from .transforms.srdiff import transform_srdiff_train, transform_srdiff_val
from .transforms.ttvsr import transform_ttvsr_train, transform_ttvsr_val
from .transforms.tunable_edsr import transforms_tunable_edsr_train, transforms_tunable_edsr_val
from .transforms.tunable_nafnet import transforms_tunable_nafnet_train, transforms_tunable_nafnet_val
from .transforms.tunable_stylenet import transforms_tunable_stylenet_train, transforms_tunable_stylenet_val
from .transforms.tunable_swinir import transforms_tunable_swinir_train, transforms_tunable_swinir_val
from .transforms.vrt import transform_vrt_train, transform_vrt_val

__all__ = ["create_transform"]

transforms = {
    "ipt": {
        "train": transforms_ipt_train,
        "val": transforms_ipt_val,
    },
    "ctsdg": {
        "train": transforms_ctsdg,
        "val": transforms_ctsdg,
    },
    "mimo_unet": {
        "train": transforms_mimo_unet,
        "val": transforms_mimo_unet,
    },
    "noahtcv": {
        "train": transform_noahtcv_train,
        "val": transform_noahtcv_val,
    },
    "basicvsr": {
        "train": transform_basicvsr_train,
        "val": transform_basicvsr_val,
    },
    "basicvsr_plus_plus_light": {
        "train": transform_basicvsr_plus_plus_light_train,
        "val": transform_basicvsr_plus_plus_light_val,
    },
    "rrdb": {
        "train": transform_rrdb_train,
        "val": transform_rrdb_val,
    },
    "srdiff": {
        "train": transform_srdiff_train,
        "val": transform_srdiff_val,
    },
    "fsrcnn": {
        "train": transform_fsrcnn,
        "val": transform_fsrcnn,
    },
    "mpfer": {
        "train": transform_mpfer,
        "val": transform_mpfer,
    },
    "vrt": {
        "train": transform_vrt_train,
        "val": transform_vrt_val,
    },
    "ttvsr": {
        "train": transform_ttvsr_train,
        "val": transform_ttvsr_val,
    },
    "emvd": {
        "train": transforms_emvd_train,
        "val": transforms_emvd_val,
    },
    "nafnet": {
        "train": transform_nafnet,
        "val": transform_nafnet,
    },
    "tunable_nafnet": {
        "train": transforms_tunable_nafnet_train,
        "val": transforms_tunable_nafnet_val,
    },
    "tunable_edsr": {
        "train": transforms_tunable_edsr_train,
        "val": transforms_tunable_edsr_val,
    },
    "tunable_stylenet": {
        "train": transforms_tunable_stylenet_train,
        "val": transforms_tunable_stylenet_val,
    },
    "tunable_swinir": {
        "train": transforms_tunable_swinir_train,
        "val": transforms_tunable_swinir_val,
    },
    "ifr_plus": {
        "train": transform_ifr_plus_train,
        "val": transform_ifr_plus_val,
    },
}


def create_transform(model_name=None, split="train", **kwargs):
    if model_name not in transforms:
        raise ValueError(f"{model_name} transforms is not supported.")
    operation, input_columns, output_columns = transforms[model_name][split](**kwargs)
    return operation, input_columns, output_columns
