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


class MAE(BaseMetric):
    def calculate_metrics(self, pred, gt, mask=None):
        """Process an image.
        Args:
            pred (np.ndarray): Pred image.
            gt (np.ndarray): GT image.
            mask (np.ndarray): Mask of evaluation.
        Returns:
            result (np.ndarray): MAE result.
        """
        pred, gt = self.preprocess(pred=pred, gt=gt)
        pred = pred / 255.0
        gt = gt / 255.0

        diff = pred - gt
        diff = abs(diff)

        if mask is not None:
            diff *= mask  # broadcast for channel dimension
            scale = np.prod(diff.shape) / np.prod(mask.shape)
            result = diff.sum() / (mask.sum() * scale + 1e-12)
        else:
            result = diff.mean()

        return result
