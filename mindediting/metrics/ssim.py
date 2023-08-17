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
import cv2
import numpy as np

from mindediting.metrics.base_metrics import BaseMetric


class SSIM(BaseMetric):
    def _ssim(self, pred, gt):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        pred = pred.astype(np.float64)
        gt = gt.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        sigma1_sq = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(gt**2, -1, window)[5:-5, 5:-5] - mu2_sq
        mu1_mu2 = mu1 * mu2
        sigma12 = cv2.filter2D(pred * gt, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def calculate_metrics(self, pred, gt):
        """Calculate SSIM (structural similarity).

        Ref:
        Image quality assessment: From error visibility to structural similarity

        For three-channel images, SSIM is calculated for each channel and then averaged.

        Args:
            pred (ndarray): Images with range [0, 255].
            gt (ndarray): Images with range [0, 255].
        Returns:
            float: ssim result.
        """
        pred, gt = self.preprocess(pred=pred, gt=gt)
        if isinstance(self.convert_to, str) and self.convert_to.lower() == "y":
            pred = np.expand_dims(pred, axis=2)
            gt = np.expand_dims(gt, axis=2)
        ssims = []
        for i in range(pred.shape[2]):
            ssims.append(self._ssim(pred[..., i], gt[..., i]))
        return np.array(ssims).mean()
