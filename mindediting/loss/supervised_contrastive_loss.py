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
import mindspore.nn as nn
import mindspore.ops as ops


class SupConLoss(nn.Cell):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
        self.eye = ops.Eye()
        self.tile = ops.Tile()
        self.scatter = ops.ScatterNd()
        self.oneslike = ops.OnesLike()
        self.exp = ops.Exp()
        self.matmul = ops.MatMul()
        self.div = ops.Div()
        self.transpose = ops.Transpose()
        self.l2normalize = ops.L2Normalize(axis=2)
        self.unstack = ops.Unstack(axis=1)
        self.unstack2 = ops.Unstack(axis=1)
        self.concat = ops.Concat(axis=0)
        self.maxes = ops.ArgMaxWithValue(axis=1, keep_dims=True)
        self.reducesum = ops.ReduceSum(keep_dims=True)
        self.log = ops.Log()
        self.reshape = ops.Reshape()
        self.reducemean = ops.ReduceMean()

    def construct(self, x):
        feature_contrast = ()
        x = self.l2normalize(x)
        batch_size = x.shape[0]
        mask = self.eye(batch_size, batch_size, ms.float32)
        temp_feature = self.concat(self.unstack(x[:, :190, :]))
        feature_contrast += (temp_feature,)
        contrast_count = x.shape[1]
        split_num = contrast_count // 190
        for num in range(split_num):
            temp_feature = self.concat(self.unstack2(x[:, 190 * (num + 1) : 190 * (num + 2), :]))
            feature_contrast += (temp_feature,)
        feature_contrast = self.concat(feature_contrast)
        if self.contrast_mode == "all":
            anchor_count = contrast_count
            anchor_feature = feature_contrast
        else:
            anchor_count = 1
            anchor_feature = x[:, 0]
        anchor_dot_contrast = self.div(
            self.matmul(anchor_feature, self.transpose(feature_contrast, (1, 0))), self.temperature
        )
        logits_max = self.maxes(anchor_dot_contrast)[1]
        logits = anchor_dot_contrast - logits_max
        mask = self.tile(mask, (anchor_count, contrast_count))
        logits_mask = 1 - self.eye(mask.shape[0], mask.shape[1], ms.float32)
        exp_logits = self.exp(logits) * logits_mask
        log_prob = logits - self.log(self.reducesum(exp_logits, 1) + 1e-8)
        mask = mask * logits_mask
        mean_log_prob_pos = self.reducesum((mask * log_prob), 1) / self.reducesum(mask, 1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = self.reducemean(self.reshape(loss, (anchor_count, batch_size)))
        return loss
