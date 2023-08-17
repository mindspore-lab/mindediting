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

class DataIO:
    def set_input_model_shape(self, **kwargs):
        pass

    def set_output_model_shape(self, **kwargs):
        pass

    def set_scale(self, scale):
        pass

    def preprocess(self, input_data):
        pass

    def postprocess(self, input_data):
        pass

    def save_result(self, output_file, output_data):
        pass
