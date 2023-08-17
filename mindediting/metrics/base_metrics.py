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

from mindediting.metrics.utils import TensorSyncer, tensor2img
from mindediting.utils.dataset_utils import bgr2ycbcr
from mindediting.utils.device_adapter import get_device_num

version = float(".".join(ms.__version__.split(".")[:2]))
if version >= 2.0:
    from mindspore import Metric
else:
    from mindspore.nn import Metric


class BaseMetric(Metric):
    def __init__(
        self,
        reduction="avg",
        crop_border=0,
        input_order="HWC",
        convert_to=None,
        process_middle_image=(False, False),
    ):
        """
        Args:
            reduction [str]: choice from ["avg", "sum"].
            crop_border (int): Cropped pixels in each edges of an image. These
                pixels are not involved in the SSIM calculation. Default: 0.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                Default: 'HWC'.
            convert_to (str): Whether to convert the images to other color models.
                If None, the images are not altered. When computing for 'Y',
                the images are assumed to be in BGR order. Options are 'Y' and
                None. Default: None.
            process_middle_image [(bool, bool)]: Video data with intermediate frame set to (True,True), default to (False,False).
        """
        super(BaseMetric, self).__init__()
        self.crop_border = crop_border
        self.input_order = input_order
        self.convert_to = convert_to
        self.process_middle_image = process_middle_image

        self.all_reduce_sum = None
        if get_device_num() > 1:
            self.all_reduce_sum = TensorSyncer(_type="sum")
        self.clear()
        self.reduction = reduction.lower()
        self.metric_list = []

    def _accumulate(self, value):
        if isinstance(value, (list, tuple)):
            self._count += len(value)
            self._accuracy_value += sum(value)
        else:
            self._accuracy_value += value
            self._count += 1

    def eval(self, sync=True):
        """
        sync: True, return metric value merged from all mindspore-processes
        sync: False, return metric value in this single mindspore-processes
        """
        if self._count == 0:
            raise RuntimeError("self._count == 0")
        if self.all_reduce_sum is not None and sync:
            data = ms.Tensor([self._accuracy_value, self._count], ms.dtype.float32)
            data = self.all_reduce_sum(data)
            acc_value, count = self._convert_data(data).tolist()
        else:
            acc_value, count = self._accuracy_value, self._count
        if self.reduction == "sum":
            return acc_value
        if self.reduction == "avg":
            return acc_value / count
        raise RuntimeError(f"_DistMetric._type={self.reduction} is not support")

    def to_numpy(self, x):
        if isinstance(x, ms.Tensor):
            x = x.asnumpy()
        return x

    def update(self, sr, hr):
        sr = self.to_numpy(sr)
        hr = self.to_numpy(hr)

        # Check if Vimeo90K-T case is using where GT consists of one image
        if self.process_middle_image[0]:
            assert sr.ndim == 5, "Central frame can be picked from frame sequences only."
            t = sr.shape[1]
            if self.process_middle_image[1]:
                sr = 0.5 * (sr[:, t // 4] + sr[:, -1 - t // 4])
                hr = hr[:, t // 4]
            else:
                sr = sr[:, t // 2]
                hr = hr[:, t // 2]

        metrics = []
        if sr.ndim == 5:  # a sequence: (n, t, c, h, w)
            for i in range(0, sr.shape[0]):
                per_frame_metrics = []
                for j in range(0, sr.shape[1]):
                    output_i = tensor2img(sr[i, j, :, :, :], self.input_order)
                    gt_i = tensor2img(hr[i, j, :, :, :], self.input_order)
                    val = self.calculate_metrics(pred=output_i, gt=gt_i)
                    per_frame_metrics.append(val)
                metrics.append(np.mean(np.array(per_frame_metrics)))
        elif sr.ndim == 4:  # images: (n, c, h, w), for Vimeo-90K-T
            for i in range(0, sr.shape[0]):
                output_img = tensor2img(sr[i, :, :, :], self.input_order)
                gt_img = tensor2img(hr[i, :, :, :], self.input_order)
                val = self.calculate_metrics(pred=output_img, gt=gt_img)
                metrics.append(val)
        elif sr.ndim == 3:  # an image: (c, h, w)
            output_img = tensor2img(sr, self.input_order)
            gt_img = tensor2img(hr, self.input_order)
            val = self.calculate_metrics(pred=output_img, gt=gt_img)
            metrics.append(val)
        else:
            raise ValueError(f"Supported only 5D, 4D and 3D inputs but got {sr.ndim}D input")
        metrics = [i for i in np.array(metrics) if not (np.isnan(i) or np.isinf(i))]
        self._accumulate(metrics)

    def clear(self):
        self._accuracy_value = 0.0
        self._count = 0

    def __del__(self):
        print(self.__class__.__name__, self.eval())

    def preprocess(self, pred, gt):
        assert pred.shape == gt.shape, f"Image shapes are different: {pred.shape}, {gt.shape}."
        if self.input_order != "HWC" and self.input_order != "CHW":
            raise ValueError(f'Wrong input_order {self.input_order}. Supported input_orders are "HWC" and "CHW"')
        pred = pred[..., None] if pred.ndim == 2 else pred
        gt = gt[..., None] if gt.ndim == 2 else gt

        num_channels = pred.shape[-1]
        assert num_channels in [4, 3, 1], f"{pred.shape} is not the 'HWC' shape of the image."
        pred, gt = pred.astype(np.float32), gt.astype(np.float32)
        is_bgr = num_channels == 3
        if is_bgr:
            if isinstance(self.convert_to, str) and self.convert_to.lower() == "y":
                reciprocal = 1 / 255
                pred = bgr2ycbcr(pred * reciprocal, y_only=True) * 255.0
                gt = bgr2ycbcr(gt * reciprocal, y_only=True) * 255.0
            elif self.convert_to is not None:
                raise ValueError('Wrong color model. Supported values are "Y" and None.')

        if self.crop_border != 0:
            pred = pred[
                self.crop_border : -self.crop_border,
                self.crop_border : -self.crop_border,
                None,
            ]
            gt = gt[
                self.crop_border : -self.crop_border,
                self.crop_border : -self.crop_border,
                None,
            ]
        return pred, gt
