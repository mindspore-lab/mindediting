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

import numpy as np

from mindediting.metrics.base_metrics import BaseMetric


class SNR(BaseMetric):
    """Signal-to-Noise Ratio.
    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    Args:
        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'CHW'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.
    Metrics:
        - SNR (float): Signal-to-Noise Ratio
    """

    def calculate_metrics(self, pred, gt):
        """Calculate PSNR (Peak Signal-to-Noise Ratio).
        Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        Args:
            gt (ndarray): Images with range [0, 255].
            pred (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edges of an image. These
                pixels are not involved in the PSNR calculation. Default: 0.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                Default: 'HWC'.
            convert_to (str): Whether to convert the images to other color models.
                If None, the images are not altered. When computing for 'Y',
                the images are assumed to be in BGR order. Options are 'Y' and
                None. Default: None.
            channel_order (str): The channel order of image. Default: 'rgb'.
        Returns:
            float: SNR result.
        """
        pred, gt = self.preprocess(pred=pred, gt=gt)
        signal = ((pred) ** 2).mean()
        noise = ((pred - gt) ** 2).mean()
        result = 10.0 * np.log10(signal / noise)
        return result
