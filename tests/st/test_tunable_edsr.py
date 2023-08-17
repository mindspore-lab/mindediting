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
from mindediting.models.tunable_image_denoise_deblur.tunable_edsr import TunableEDSR
from mindediting.utils.dataset_utils import write_image


@pytest.mark.parametrize("params", [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
@pytest.mark.parametrize("data_root", ["/data/LLVT/Tunable_Conv/data/kodak"])
@pytest.mark.parametrize("ckpt_path", ["/data/LLVT/Tunable_Conv/ckpt/t_edsr_ms.ckpt"])
@pytest.mark.parametrize("noise_stddev", [15.0])
@pytest.mark.parametrize("blur_stddev", [2.0])
def test_tunable_edsr_val(data_root, params, ckpt_path, noise_stddev, blur_stddev, max_images=100, save_images=False):
    """
    chose different params to get different PSNR result, i.e. modulate denoise with deblur for image.
    (1.0, 0.0) -> maximum denoising, minimum deblurring
    (1.0, 1.0) -> maximum denoising, maximum deblurring
    (0.0, 1.0) -> minimum denoising, maximum deblurring
    """
    assert all(0.0 <= p <= 1.0 for p in params)
    print(f" - params: {params}")
    print(f" - Noise stddev: {noise_stddev}")
    print(f" - Blur stddev: {blur_stddev}")
    px = np.expand_dims(params, axis=0).astype(np.float32)
    dataset_val = create_dataset(name="kodak", root=data_root, split="val", shuffle=False, nframes=None)
    val_operations, val_input_columns, val_output_columns = create_transform(
        model_name="tunable_edsr",
        split="val",
        pipeline=None,
        **{"noise_stddev": noise_stddev, "blur_stddev": blur_stddev},
    )
    loader_val = create_loader(
        dataset=dataset_val,
        batch_size=1,
        operations=val_operations,
        input_columns=val_input_columns,
        output_columns=val_output_columns,
        split="val",
    )
    net = TunableEDSR(img_channels=3, scale=1, num_params=2, mode="mlp")
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
        fname = dataset_val.source._data[index]["HR"].split(os.path.sep)
        index += 1
        if save_images:
            write_image(
                fname[-1].replace(".png", f"_noise{noise_stddev}_blur{blur_stddev}_tuning_{px[0, 0]}_{px[0, 1]}.png"),
                np.concatenate([lr.asnumpy(), sr.asnumpy(), hr.asnumpy()], axis=-1),
                root="output/edsr",
            )
    if params == (1.0, 0.0):
        np.testing.assert_almost_equal(psnr.eval(), 26.443301618895024, decimal=1)
    elif params == (1.0, 1.0):
        np.testing.assert_almost_equal(psnr.eval(), 25.71963703107902, decimal=1)
    elif params == (0.0, 1.0):
        np.testing.assert_almost_equal(psnr.eval(), 17.95147509361738, decimal=1)
