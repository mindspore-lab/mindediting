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
from mindediting.models.tunable_mutil_task.tunable_swinir import TunableSwinIR
from mindediting.utils.dataset_utils import upsample_image, write_image
from mindediting.utils.utils import is_ascend


@pytest.mark.parametrize("params", [(1.0, 0.0), (0.0, 1.0)])
@pytest.mark.parametrize("data_root", ["/data/LLVT/Tunable_Conv/data/kodak"])
@pytest.mark.parametrize("ckpt_path", ["/data/LLVT/Tunable_Conv/ckpt"])
@pytest.mark.parametrize("noise_stddev", [25.0])
def test_tunable_swinir_denoise_val(data_root, params, ckpt_path, noise_stddev, max_images=100, save_images=False):
    """
    chose different params to get different PSNR result, i.e. modulate denoise for image.
     (1.0, 0.0) -> maximum denoising
     (0.5, 0.5) -> medium  denoising
     (0.0, 1.0) -> minimum denoising
    """

    if is_ascend():
        pytest.skip("Too long on Ascend")

    assert all(0.0 <= p <= 1.0 for p in params)
    print(f" - params: {params}")
    print(f" - Noise stddev: {noise_stddev}")
    px = np.expand_dims(params, axis=0).astype(np.float32)
    dataset_val = create_dataset(name="kodak", root=data_root, split="val", shuffle=False, nframes=None)
    val_operations, val_input_columns, val_output_columns = create_transform(
        model_name="tunable_swinir", split="val", pipeline=None, **{"noise_stddev": noise_stddev}
    )
    loader_val = create_loader(
        dataset=dataset_val,
        batch_size=1,
        operations=val_operations,
        input_columns=val_input_columns,
        output_columns=val_output_columns,
        split="val",
    )

    net = TunableSwinIR(
        img_channels=3,
        window_size=8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        mlp_ratio=2.0,
        resi_connection="1conv",
        num_params=2,
        mode="mlp",
    )
    ckpt_path = os.path.join(ckpt_path, f"t_swinir_dn{int(noise_stddev)}_ms.ckpt")
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
                fname[-1].replace(".png", f"_denoise{noise_stddev}_tuning_{px[0, 0]}_{px[0, 1]}.png"),
                np.concatenate([lr.asnumpy(), sr.asnumpy(), hr.asnumpy()], axis=-1),
                root="output/swinir",
            )
        index += 1
    if params == (1.0, 0.0):
        np.testing.assert_almost_equal(psnr.eval(), 32.92225794689168)
    elif params == (0.0, 1.0):
        np.testing.assert_almost_equal(psnr.eval(), 21.295782764473135)


@pytest.mark.parametrize("params", [(1.0, 0.0), (0.0, 1.0)])
@pytest.mark.parametrize("data_root", ["/data/LLVT/Tunable_Conv/data/kodak"])
@pytest.mark.parametrize("ckpt_path", ["/data/LLVT/Tunable_Conv/ckpt/"])
@pytest.mark.parametrize("scale", [4])
def test_tunable_swinir_x4sr_val(data_root, params, ckpt_path, scale, max_images=100, save_images=False):
    """
    chose different params to get different PSNR result, i.e. modulate x4sr for image.
     (1.0, 0.0) -> maximize accuracy
     (0.5, 0.5) -> mixed accuracy and perceptual quality
     (0.0, 1.0) -> maximize perceptual quality
    """

    if is_ascend():
        pytest.skip("Too long on Ascend")
    sr_factor = scale
    assert all(0.0 <= p <= 1.0 for p in params)
    print(f" - params: {params}")
    px = np.expand_dims(params, axis=0).astype(np.float32)
    dataset_val = create_dataset(name="kodak", root=data_root, split="val", shuffle=False, nframes=None)
    val_operations, val_input_columns, val_output_columns = create_transform(
        model_name="tunable_swinir", split="val", pipeline=None, **{"scale": scale}
    )
    loader_val = create_loader(
        dataset=dataset_val,
        batch_size=1,
        operations=val_operations,
        input_columns=val_input_columns,
        output_columns=val_output_columns,
        split="val",
    )
    net = TunableSwinIR(
        img_channels=3,
        window_size=8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        mlp_ratio=2.0,
        resi_connection="1conv",
        num_params=2,
        mode="mlp",
        upsampler="pixelshuffle",
        upscale=4,
    )
    ckpt_path = os.path.join(ckpt_path, f"t_swinir_sr{sr_factor}_ms.ckpt")
    ckpt_parm = mindspore.load_checkpoint(ckpt_path)
    mindspore.load_param_into_net(net, ckpt_parm)
    # Cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation.
    psnr = PSNR(reduction="avg", crop_border=sr_factor, input_order="CHW", convert_to="y")
    index = 0
    for data in loader_val:
        if index >= max_images:
            continue
        lr, hr = data
        sr = net(lr, mindspore.Tensor(px))
        psnr.update(sr, hr)
        if save_images:
            fname = dataset_val.source._data[index]["HR"].split(os.path.sep)
            hr_height, hr_width = hr.shape[2:]
            lr = upsample_image(lr.asnumpy(), (hr_width, hr_height))
            write_image(
                fname[-1].replace(".png", f"_sr{sr_factor}_tuning_{px[0, 0]}_{px[0, 1]}.png"),
                np.concatenate([lr, sr.asnumpy(), hr.asnumpy()], axis=-1),
                root="output/swinir",
            )
        index += 1
    if params == (1.0, 0.0):
        np.testing.assert_almost_equal(psnr.eval(), 29.36201954605669)
    elif params == (0.0, 1.0):
        np.testing.assert_almost_equal(psnr.eval(), 26.63876710371075)
