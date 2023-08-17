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

"""callbacks"""

import datetime
import os
import time
from pathlib import Path
from typing import Union

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import numpy as mnp
from mindspore import save_checkpoint
from mindspore.ops import Concat
from mindspore.train.callback import Callback, RunContext, _CallbackManager, _InternalCallbackParam
from PIL import Image

from .device_adapter import get_rank_id

_S_IWRITE = 128


class LossTimeMonitor(Callback):
    """loss time monitor"""

    def __init__(self, cfg):
        super().__init__()
        self.log_step_time = time.time()
        self.per_print_times = cfg.train_params.log_frequency_step

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cb_params.cur_step_num += 1
        if cb_params.cur_step_num % self.per_print_times == 0:
            loss_g, loss_d = cb_params.net_outputs
            loss_g = round(loss_g, 3)
            loss_d = round(loss_d, 3)

            time_used = time.time() - self.log_step_time
            per_step_time = round(1e3 * time_used / self.per_print_times, 2)

            date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(
                f"{date_time} iter: {cb_params.cur_step_num}, loss_g: {loss_g}, "
                f"loss_d: {loss_d}, step time: {per_step_time} ms",
                flush=True,
            )

            self.log_step_time = time.time()


class CTSDGModelCheckpoint(Callback):
    """CTSDG ModelCheckpoint"""

    def __init__(self, cfg):
        super().__init__()
        self.save_checkpoint_steps = cfg.train_params.save_checkpoint_steps
        self.keep_checkpoint_max = cfg.train_params.keep_checkpoint_max
        self.checkpoint_filelist = []
        self.save_path = cfg.train_params.ckpt_save_dir
        self.total_steps = cfg.train.total_steps
        Path(self.save_path).mkdir(exist_ok=True, parents=True)

    def _remove_ckpoint_file(self, file_name, is_g):
        """Remove the specified checkpoint file from this checkpoint manager
        and also from the directory."""
        try:
            os.chmod(file_name, _S_IWRITE)
            os.remove(file_name)
            if is_g:
                self.checkpoint_filelist.remove(file_name)
        except OSError:
            print(f"OSError, failed to remove the older ckpt file {file_name}.", flush=True)
        except ValueError:
            print(f"ValueError, failed to remove the older ckpt file {file_name}.", flush=True)

    def _remove_oldest(self):
        """remove oldest checkpoint file"""
        ckpoint_files = sorted(self.checkpoint_filelist, key=os.path.getmtime)
        file_to_remove = Path(ckpoint_files[0])
        name_g = file_to_remove.name
        name_d = name_g.replace("generator_", "discriminator_")
        file_name_g = file_to_remove.as_posix()
        file_name_d = (file_to_remove.parent / name_d).as_posix()
        self._remove_ckpoint_file(file_name_g, True)
        self._remove_ckpoint_file(file_name_d, False)

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step % self.save_checkpoint_steps == 0 or cur_step == self.total_steps:
            g_name = os.path.join(self.save_path, f"generator_{cur_step:06d}.ckpt")
            save_checkpoint(cb_params.train_network_g, g_name)
            d_name = os.path.join(self.save_path, f"discriminator_{cur_step:06d}.ckpt")
            save_checkpoint(cb_params.train_network_d, d_name)
            self.checkpoint_filelist.append(g_name)

        if len(self.checkpoint_filelist) > self.keep_checkpoint_max:
            self._remove_oldest()


class CTSDGCallbackManager:
    """ctsdg callback manager"""

    def __init__(self, cfg, model_g, model_d):
        ckpt_cb = CTSDGModelCheckpoint(cfg)
        ltm_cb = LossTimeMonitor(cfg)

        cb_params = _InternalCallbackParam()
        cb_params.train_network_g = model_g
        cb_params.train_network_d = model_d
        cb_params.cur_step_num = cfg.train.start_iter
        self.cb_params = cb_params
        self.run_context = RunContext(self.cb_params)
        cbs = []
        if get_rank_id() == 0:
            cbs = [ltm_cb, ckpt_cb]
        self.cb_manager = _CallbackManager(cbs)
        self.cb_manager.begin(self.run_context)

    def __call__(self, losses):
        self.cb_params.net_outputs = losses
        self.cb_manager.step_end(self.run_context)


def get_callbacks(cfg, model_g, model_d, finetune):
    """get callbacks"""
    cb_manager = CTSDGCallbackManager(cfg, model_g, model_d)
    print("==============================", flush=True)
    if finetune:
        print("Start finetune", flush=True)
    else:
        print(f"Start training", flush=True)
    return cb_manager


class EvalCallback:
    def __init__(self, calc_psnr, calc_ssim, output_path, total, test_batch_size, verbose_step):
        self.psnr_dict = {
            "0-20%": [],
            "20-40%": [],
            "40-60%": [],
        }
        self.ssim_dict = {
            "0-20%": [],
            "20-40%": [],
            "40-60%": [],
        }
        self.calc_psnr = calc_psnr
        self.calc_ssim = calc_ssim
        if output_path:
            self.save_dir = Path(output_path)
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True)
        else:
            self.save_dir = None

        self.total = total
        self.test_batch_size = test_batch_size
        self.verbose_step = verbose_step

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        """
        Map tensor values from [-1, 1] to [0, 1]

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        x = (x + 1.0) * 0.5
        x = mnp.clip(x, 0, 1)
        return x

    @staticmethod
    def save_img(img: Tensor, path: Union[str, Path]):
        """
        Normalize and save output image

        Args:
            img: Tensor image with values in [0, 1] range
            path: Image save path

        Returns:
            None
        """
        if img.ndim == 4:
            img = img[0]
        img_np = mnp.clip((img * 255) + 0.5, 0, 255).transpose((1, 2, 0)).astype(mstype.uint8).asnumpy()
        Image.fromarray(img_np).save(path)

    @staticmethod
    def print_metric(name, metric_dict):
        print(name + ":", flush=True)
        for k, v in metric_dict.items():
            if v:
                metric_value = float(Concat()(v).mean())
                print(f"{k}: {metric_value:.3f}", flush=True)

    def __call__(self, ground_truth, mask, net_output):
        output_comp = ground_truth * mask + net_output * (1 - mask)
        output_comp = self.postprocess(output_comp)

        ground_truth_post = self.postprocess(ground_truth)

        psnr_value = self.calc_psnr(output_comp, ground_truth_post)
        ssim_value = self.calc_ssim(output_comp, ground_truth_post)

        part = 1 - mask.sum() / mask.size

        if part <= 0.2:
            self.psnr_dict["0-20%"].append(psnr_value)
            self.ssim_dict["0-20%"].append(ssim_value)
        elif 0.2 < part <= 0.4:
            self.psnr_dict["20-40%"].append(psnr_value)
            self.ssim_dict["20-40%"].append(ssim_value)
        elif 0.4 < part <= 0.6:
            self.psnr_dict["40-60%"].append(psnr_value)
            self.ssim_dict["40-60%"].append(ssim_value)

        if self.save_dir:
            for i in range(self.test_batch_size):
                self.pic_index += 1
                self.save_img(output_comp[i, ...], self.save_dir / f"{self.pic_index:05d}.png")
        else:
            self.pic_index += self.test_batch_size

        if self.pic_index % self.verbose_step == 0:
            end = time.time()
            pic_cost = (end - self.start_time) / self.verbose_step
            time_left = (self.total - self.pic_index) * pic_cost
            print(
                f"Processed images: {self.pic_index} of {self.total}, "
                f"Fps: {1 / pic_cost:.2f}, "
                f"Image time: {pic_cost:.2f} sec, "
                f"Time left: ~{time_left:.2f} sec.",
                flush=True,
            )
            self.start_time = time.time()

            self.print_metric("PSNR", self.psnr_dict)
            self.print_metric("SSIM", self.ssim_dict)

    def start_eval(self):
        print(f"Total number of images in dataset: {self.total}", flush=True)
        self.start_time = time.time()
        self.pic_index = 0

    def end_eval(self):
        print()
        print("TOTAL:")
        self.print_metric("PSNR", self.psnr_dict)
        self.print_metric("SSIM", self.ssim_dict)
