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


import mindspore.nn as nn
from mindspore._c_expression import Cell_, MixedPrecisionType


class NotSetPrecisionCell(nn.Cell):
    def _set_mixed_precision_type_recursive(self, mixed_type):
        Cell_.set_mixed_precision_type(self, MixedPrecisionType.NOTSET)
        for cell in self.cells():
            cell._set_mixed_precision_type_recursive(MixedPrecisionType.NOTSET)
