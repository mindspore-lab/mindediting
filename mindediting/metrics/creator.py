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

from .dists import DISTS
from .lpips import LPIPS
from .mae import MAE
from .ms_ssim import MS_SSIM
from .niqe import NIQE
from .psnr import PSNR
from .snr import SNR
from .ssim import SSIM


def create_metrics(metric_cfg):
    """
    Creates a dictionary of metrics based on the data from the config.

    Args:
        metric_cfg (mindediting.utils.config.Config): Config with the name
            of the metric and its parameters.

    Returns:
        dict: Dictionary with metric name and metric class.
    """
    available_metrics = {
        "PSNR": PSNR,
        "SSIM": SSIM,
        "MAE": MAE,
        "SNR": SNR,
        "MS_SSIM": MS_SSIM,
        "NIQE": NIQE,
        "LPIPS": LPIPS,
        "DISTS": DISTS,
    }
    metrics = {}
    for metric_name, params in metric_cfg.cfg_dict.items():
        if metric_name not in available_metrics:
            raise ValueError(
                f"Metric {metric_name} is not supported yet. Implemented metrics: {list(available_metrics.keys())}."
            )

        metrics[metric_name] = available_metrics[metric_name](**params)
    return metrics
