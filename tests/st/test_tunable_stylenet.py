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

import os

import mindspore
import numpy as np
import pytest
from common import init_test_environment

init_test_environment()
from mindediting.dataset.create_datasets import create_dataset
from mindediting.dataset.create_loaders import create_loader
from mindediting.dataset.create_transforms import create_transform
from mindediting.metrics.psnr import PSNR
from mindediting.models.tunable_image_style.tunable_stylenet import TunableStyleNet
from mindediting.utils.dataset_utils import write_image


@pytest.mark.parametrize("params", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
@pytest.mark.parametrize("data_root", ["/data/LLVT/Tunable_Conv/data/kodak"])
@pytest.mark.parametrize("ckpt_path", ["/data/LLVT/Tunable_Conv/ckpt/t_stylenet_ms.ckpt"])
def test_tunable_stylenet_val(data_root, params, ckpt_path, max_images=100, save_images=False):
    """
    Pick any values for these parameters. For example:
     (1.0, 0.0, 0.0) -> Mosaic style
     (0.0, 1.0, 0.0) -> Edtaonisl style
     (0.0, 0.0, 1.0) -> Kandinsky style
     (0.5, 0.0, 0.5) -> mixed Mosaic and Kandinsky style
    """
    px = np.expand_dims(params, axis=0).astype(np.float32)
    dataset_val = create_dataset(name="kodak", root=data_root, split="val", shuffle=False, nframes=None)
    val_operations, val_input_columns, val_output_columns = create_transform(
        model_name="tunable_stylenet",
        split="val",
        pipeline=None,
    )
    loader_val = create_loader(
        dataset=dataset_val,
        batch_size=1,
        operations=val_operations,
        input_columns=val_input_columns,
        output_columns=val_output_columns,
        split="val",
    )
    net = TunableStyleNet(img_channels=3, num_params=3, mode="mlp")
    ckpt_parm = mindspore.load_checkpoint(ckpt_path)
    mindspore.load_param_into_net(net, ckpt_parm)
    psnr = PSNR(reduction="avg", crop_border=0, input_order="CHW", convert_to=None)
    index = 0
    for data in loader_val:
        if index >= max_images:
            continue
        lr, hr = data
        sr = net(lr, mindspore.Tensor(px))
        psnr.update(sr, hr)
        if save_images:
            fname = dataset_val.source._data[index]["HR"].split(os.path.sep)
            write_image(
                fname[-1].replace(".png", f"_tuning_{px[0, 0]}_{px[0, 1]}_{px[0, 2]}.png"),
                np.concatenate([sr.asnumpy(), hr.asnumpy()], axis=-1),
                root="output/stylenet",
            )
        index += 1
    if params == (1.0, 0.0, 0.0):
        np.testing.assert_almost_equal(psnr.eval(), 14.232618154719104, decimal=2)
    elif params == (0.0, 1.0, 0.0):
        np.testing.assert_almost_equal(psnr.eval(), 17.10574857223054, decimal=2)
