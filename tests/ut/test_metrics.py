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
import mindspore as ms
import numpy as np
import pytest
from common import init_test_environment

init_test_environment()

from mindediting.metrics.dists import DISTS
from mindediting.metrics.lpips import LPIPS
from mindediting.metrics.mae import MAE
from mindediting.metrics.ms_ssim import MS_SSIM
from mindediting.metrics.niqe import NIQE
from mindediting.metrics.psnr import PSNR
from mindediting.metrics.snr import SNR
from mindediting.metrics.ssim import SSIM


@pytest.mark.parametrize("convert_to", [None, "y"])
def test_calculate_psnr(convert_to):
    sr = np.ones((3, 16, 16), dtype=np.float32)
    hr = np.ones((3, 16, 16), dtype=np.float32) * 0.5
    psnr = PSNR(reduction="avg", crop_border=0, input_order="CHW", convert_to=convert_to)
    psnr.update(sr, hr)
    result = psnr.eval()
    if convert_to is None:
        np.testing.assert_almost_equal(result, 6.054729189491574, decimal=3)
    elif convert_to == "y":
        np.testing.assert_almost_equal(result, 7.376651194702327, decimal=3)


@pytest.mark.parametrize("convert_to", [None, "y"])
def test_calculate_ssim(convert_to):
    sr = np.ones((3, 16, 16))
    hr = np.ones((3, 16, 16)) * 0.5
    ssim = SSIM(reduction="avg", crop_border=0, input_order="CHW", convert_to=convert_to)
    ssim.update(sr, hr)
    result = ssim.eval()
    if convert_to is None:
        np.testing.assert_almost_equal(result, 0.8018927660608104, decimal=3)
    elif convert_to == "y":
        np.testing.assert_almost_equal(result, 0.8326566335668579, decimal=3)


@pytest.mark.parametrize("convert_to", [None, "y"])
def test_calculate_ms_ssim(convert_to):
    sr = np.ones((3, 16, 16))
    hr = np.ones((3, 16, 16)) * 0.5
    ms_ssim = MS_SSIM(reduction="avg", crop_border=0, input_order="CHW", convert_to=convert_to)
    ms_ssim.update(sr, hr)
    result = ms_ssim.eval()
    if convert_to is None:
        np.testing.assert_almost_equal(result, 0.9710074155719598, decimal=3)
    elif convert_to == "y":
        np.testing.assert_almost_equal(result, 0.9758526582082461, decimal=3)


@pytest.mark.parametrize("convert_to", [None, "y"])
def test_calculate_mae(convert_to):
    sr = np.ones((3, 16, 16))
    hr = np.ones((3, 16, 16)) * 0.5
    mae = MAE(reduction="avg", crop_border=0, input_order="CHW", convert_to=convert_to)
    mae.update(sr, hr)
    result = mae.eval()
    if convert_to is None:
        np.testing.assert_almost_equal(result, 0.49803921580314636, decimal=3)
    elif convert_to == "y":
        np.testing.assert_almost_equal(result, 0.42772772908210754, decimal=3)


@pytest.mark.parametrize("convert_to", [None, "y"])
def test_calculate_snr(convert_to):
    sr = np.ones((3, 16, 16))
    hr = np.ones((3, 16, 16)) * 0.5
    snr = SNR(reduction="avg", crop_border=0, input_order="CHW", convert_to=convert_to)
    snr.update(sr, hr)
    result = snr.eval()
    if convert_to is None:
        np.testing.assert_almost_equal(result, 6.054729189491574, decimal=3)
    elif convert_to == "y":
        np.testing.assert_almost_equal(result, 6.667205095291138, decimal=3)


@pytest.mark.parametrize("convert_to", [None, "y"])
@pytest.mark.parametrize("model_path", ["/data/LLVT/IQA/niqe_pris_params.npz"])
def test_calculate_niqe(convert_to, model_path):
    sr = np.random.random((3, 320, 480))
    hr = np.random.random((3, 320, 480))
    niqe = NIQE(reduction="avg", crop_border=0, input_order="CHW", convert_to=convert_to, model_path=model_path)
    niqe.update(sr, hr)
    result = niqe.eval()
    if convert_to is None:
        np.testing.assert_almost_equal(result, 12.214226470396106, decimal=1)
    elif convert_to == "y":
        np.testing.assert_almost_equal(result, 10.409364596695527, decimal=1)


@pytest.mark.parametrize("convert_to", [None, "y"])
@pytest.mark.parametrize("vgg16_model_path", ["/data/LLVT/IQA/vgg16.ckpt"])
@pytest.mark.parametrize("alpha_beta_model_path", ["/data/LLVT/IQA/dists_alpha_beta.ckpt"])
def test_calculate_dists(convert_to, vgg16_model_path, alpha_beta_model_path):
    sr = np.ones((3, 16, 16))
    hr = np.ones((3, 16, 16)) * 0.5
    dists = DISTS(
        reduction="avg",
        crop_border=0,
        input_order="CHW",
        convert_to=convert_to,
        vgg16_model_path=vgg16_model_path,
        alpha_beta_model_path=alpha_beta_model_path,
    )
    dists.update(sr, hr)
    result = dists.eval()
    if convert_to is None:
        np.testing.assert_almost_equal(result, 0.2759544849395752, decimal=2)
    elif convert_to == "y":
        np.testing.assert_almost_equal(result, 0.24408942461013794, decimal=2)


@pytest.mark.parametrize("convert_to", [None, "y"])
@pytest.mark.parametrize(
    "args",
    [
        ["vgg", "/data/LLVT/IQA/vgg16.ckpt", "/data/LLVT/IQA/lpips_vgg.ckpt"],
        ["squeeze", "/data/LLVT/IQA/squeezenet1_1.ckpt", "/data/LLVT/IQA/lpips_squeeze.ckpt"],
        ["alex", "/data/LLVT/IQA/alexnet.ckpt", "/data/LLVT/IQA/lpips_alex.ckpt"],
    ],
)
def test_calculate_lpips(convert_to, args):
    net, pnet_model_path, lpips_model_path = args
    sr = np.ones((3, 64, 64))
    hr = np.ones((3, 64, 64)) * 0.5
    lpips = LPIPS(
        reduction="avg",
        crop_border=0,
        input_order="CHW",
        convert_to=convert_to,
        net=net,
        pnet_model_path=pnet_model_path,
        lpips_model_path=lpips_model_path,
    )
    lpips.update(sr, hr)
    result = lpips.eval()
    if convert_to is None:
        if net == "vgg":
            np.testing.assert_almost_equal(result, 0.49301424622535706, decimal=3)
        elif net == "squeeze":
            np.testing.assert_almost_equal(result, 0.32175976037979126, decimal=3)
        elif net == "alex":
            np.testing.assert_almost_equal(result, 0.36084216833114624, decimal=3)
    elif convert_to == "y":
        if net == "vgg":
            np.testing.assert_almost_equal(result, 0.3839491009712219, decimal=3)
        elif net == "squeeze":
            np.testing.assert_almost_equal(result, 0.26618117094039917, decimal=3)
        elif net == "alex":
            np.testing.assert_almost_equal(result, 0.38287752866744995, decimal=3)
