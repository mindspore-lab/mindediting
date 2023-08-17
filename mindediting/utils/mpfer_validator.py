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


class MpferValidator:
    def __init__(
        self,
        net,
        eval_network,
        loss_fn,
        metrics,
        input_indices=[6, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        valid_scenes=[0, 9, 10, 23, 24, 52, 56, 62, 63, 73],
        position=0,
        ref_height=496,
        ref_width=800,
        h0=0,
        h1=496,
        w0=0,
        w1=800,
    ) -> None:
        # print("In mpfer validator init!")
        self.input_indices = input_indices
        self.valid_scenes = valid_scenes
        self.position = position
        self.ref_height = ref_height
        self.ref_width = ref_width
        self.h0 = h0
        self.h1 = h1
        self.w0 = w0
        self.w1 = w1

        self.model = eval_network
        self.metrics = metrics

    def eval(self, loader_val, dataset_sink_mode, callbacks):
        for i in range(len(callbacks)):
            callbacks[i].on_eval_begin(None)

        corners = [0, self.ref_height, 0, self.ref_width]
        psnr_array = np.zeros((len(self.valid_scenes), len(self.input_indices)))
        for scene_idx, curr_item in enumerate(loader_val):
            for i in range(len(callbacks)):
                callbacks[i].on_eval_step_begin(None)
            input_imgs, scene, imgs, heights, poses_ms, intrinsics_ms, ref_pose_ms, ref_intrinsics_ms, depths_ms = (
                curr_item[0].astype(ms.float32),
                curr_item[1].asnumpy(),
                curr_item[2].asnumpy(),
                curr_item[3].asnumpy(),
                curr_item[4].astype(ms.float32),
                curr_item[5].astype(ms.float32),
                curr_item[6].astype(ms.float32),
                curr_item[7].astype(ms.float32),
                curr_item[8].astype(ms.float32),
            )

            scene = scene[0]
            batch_ms = dict()
            batch_ms["input_imgs"] = input_imgs[..., self.h0 : self.h1, self.w0 : self.w1]
            batch_ms["corners"] = corners

            batch_ms["poses"] = poses_ms
            batch_ms["intrinsics"] = intrinsics_ms
            batch_ms["ref_pose"] = ref_pose_ms
            batch_ms["ref_intrinsics"] = ref_intrinsics_ms
            batch_ms["depths"] = depths_ms

            self.model.set_train(False)
            output_ms = self.model(batch_ms)
            output_ms = np.clip(output_ms.asnumpy(), 0, 1)

            for batch_idx in range(heights.shape[0]):
                for view_idx, view in enumerate(self.input_indices):
                    h_min = min(self.h1, heights[batch_idx][view_idx])
                    for k in self.metrics:
                        self.metrics[k].update(
                            imgs[batch_idx, view_idx, :, self.h0 : h_min, self.w0 : self.w1],
                            output_ms[batch_idx, view_idx, :, : (h_min - self.h0), :],
                        )

            for i in range(len(callbacks)):
                callbacks[i].on_eval_step_end(None)

        for i in range(len(callbacks)):
            callbacks[i].on_eval_end(None)
