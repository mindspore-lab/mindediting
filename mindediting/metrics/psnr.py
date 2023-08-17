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


class PSNR(BaseMetric):
    def calculate_metrics(self, pred, gt):
        pred, gt = self.preprocess(pred=pred, gt=gt)
        mse_value = np.mean((pred - gt) ** 2)
        return 20.0 * np.log10(255.0 / (np.sqrt(mse_value) + 1e-9))
