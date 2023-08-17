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

import datetime
import json
import os
import time

import mindspore
import numpy as np
from mindspore import Profiler
from mindspore.train.callback import Callback
from PIL import Image
from tqdm import tqdm

from mindediting import create_validator
from mindediting.dataset.create_loaders import create_data_loader
from mindediting.deploy.utils.config import Config
from mindediting.loss import create_loss
from mindediting.metrics import create_metrics
from mindediting.metrics.utils import tensor2img
from mindediting.utils.local_adapter import get_device_num, get_rank_id
from mindediting.utils.utils import get_rank_mentioned_filename


def save_metrics(metrics, path):
    metrics_values = {k: float(v.eval()) for k, v in metrics.items()}
    with open(path, "w") as write_file:
        json.dump(metrics_values, write_file)


def val_py_eval(cfg, net, eval_network, save_metrics_path):
    # create val data loader
    if hasattr(cfg, "val_dataset"):
        loader_val = create_data_loader(
            cfg.model.name, cfg.val_dataset, cfg.test_pipeline, "val", cfg.val_dataset.eval_batch_size
        )
    else:
        loader_val = create_data_loader(
            cfg.model.name, cfg.dataset, cfg.test_pipeline, "val", cfg.dataset.eval_batch_size
        )
    cfg.steps_per_epoch = loader_val.get_dataset_size()

    # create loss
    loss = create_loss(loss_name=cfg.loss.name, **cfg.loss.cfg_dict)
    # define metrics
    metrics = create_metrics(cfg.metric)

    # define Callback
    callbacks = [mindspore.TimeMonitor()]
    if get_rank_id() == 0:
        callbacks.append(EvalProgressBar(cfg.steps_per_epoch))
    if hasattr(cfg, "val_params") and getattr(cfg.val_params, "img_save_directory", False):
        os.makedirs(cfg.val_params.img_save_directory, exist_ok=True)
        callbacks.append(EvalSaveImgs(cfg.val_params.img_save_directory, cfg.val_params.save_bgr))

    # define validator
    validator = create_validator(
        getattr(cfg, "validator", Config({"name": "default"})),
        net,
        eval_network,
        loss,
        metrics,
    )
    print(" validating...")
    validator.eval(loader_val, dataset_sink_mode=cfg.dataset.dataset_sink_mode, callbacks=callbacks)

    if save_metrics_path:
        if get_device_num() > 1:
            save_metrics_path = get_rank_mentioned_filename(save_metrics_path)
        save_metrics(metrics, save_metrics_path)

    return {k: float(v.eval()) for k, v in metrics.items()}


class EvalAsInValPyCallBack(Callback):
    """
    eval callback
    """

    def __init__(self, cfg, net, eval_net=None, summary_writer=None):
        self.cfg = cfg
        self.net = net
        self.eval_net = eval_net
        self.by_epoch = getattr(self.cfg.train_params, "eval_by_epoch", True)
        self.eval_frequency = self.cfg.train_params.eval_frequency
        self.step = 0
        self.summary_writer = summary_writer

    def _add_metrics_summary(self, metrics, cur_step):
        if self.summary_writer:
            for k, v in metrics.items():
                self.summary_writer.add_scalar("val/" + k, v, cur_step)
            self.summary_writer.flush()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if self.by_epoch and cb_params.cur_epoch_num % self.eval_frequency == 0:
            self.net.set_train(False)
            metrics = val_py_eval(self.cfg, self.net, self.eval_net, "")
            self._add_metrics_summary(metrics, cb_params.cur_step_num)
            self.net.set_train(True)

    def on_train_step_end(self, run_context):
        self.step += 1
        if not self.by_epoch and self.step % self.eval_frequency == 0:
            cb_params = run_context.original_args()
            self.net.set_train(False)
            metrics = val_py_eval(self.cfg, self.net, self.eval_net, "")
            self._add_metrics_summary(metrics, cb_params.cur_step_num)
            self.net.set_train(True)


def to_scalar(value):
    if isinstance(value, mindspore.Tensor):
        value = value.asnumpy()
    if isinstance(value, np.ndarray):
        value = value.item()
    return value


class TrainingMonitor(Callback):
    def __init__(self, epochs, steps_per_epoch, print_frequency=100, summary_writer=None):
        super(TrainingMonitor, self).__init__()
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.print_frequency = print_frequency
        self.start_time = time.time()
        self.last_timestamp = None
        self.step_time_sum = 0
        self.summary_writer = summary_writer

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        batch_num = cb_params.batch_num
        cur_epoch_num = cb_params.cur_epoch_num
        cur_step_num = cb_params.cur_step_num
        cur_step_in_epoch = (cur_step_num - 1) % batch_num + 1
        current_timestamp = time.time()
        train_duration = current_timestamp - self.start_time
        if self.last_timestamp is not None:
            step_time = current_timestamp - self.last_timestamp
            self.last_timestamp = current_timestamp
        else:
            step_time = 0
            self.last_timestamp = current_timestamp
        self.step_time_sum += step_time
        if cur_step_num > 0:
            step_time_avg = self.step_time_sum / cur_step_num
        else:
            step_time_avg = 0
        eta = step_time_avg * (self.total_steps - cur_step_num)
        if (
            cur_step_in_epoch % self.print_frequency == 0
            or cur_step_in_epoch == 1
            or cur_step_in_epoch == self.steps_per_epoch
        ):
            if isinstance(loss, (list, tuple)) and len(loss) == 3:
                loss = loss[0].asnumpy().item()
            train_duration = self.convert_time(train_duration)
            eta = self.convert_time(eta)
            output_str = (
                f"epoch {cur_epoch_num}/{self.epochs} step {cur_step_in_epoch}/{self.steps_per_epoch}, "
                f"loss = {loss}, duration_time = {train_duration}, "
                f"step_time_avg = {step_time_avg:.2f} secs, eta = {eta}"
            )
            if get_rank_id() == 0:
                print(output_str)
                if self.summary_writer:
                    self.summary_writer.add_scalar(
                        "loss", to_scalar(loss), global_step=to_scalar(cb_params.cur_step_num)
                    )
                    self.summary_writer.flush()

    def convert_time(self, t):
        m = int(t) // 60
        s = int(t - m * 60)
        h = m // 60
        m -= int(h * 60)
        d = 0
        if h > 23:
            d = h // 24
            h -= int(d * 24)
        out_str = f"{self.get_time_str(h)}:{self.get_time_str(m)}:{self.get_time_str(s)}"
        if d > 0:
            out_str = f"{d} day(s) " + out_str
        return out_str

    @staticmethod
    def get_time_str(t):
        if len(str(t)) > 1:
            return t
        else:
            return f"0{t}"


class EvalProgressBar(mindspore.Callback):
    def __init__(self, total_steps=None):
        super().__init__()
        self.pbar = None
        self.total_steps = total_steps

    def on_eval_begin(self, run_context):
        self.pbar = tqdm(total=self.total_steps)

    def on_eval_end(self, run_context):
        if self.pbar is not None:
            self.pbar.close()

    def on_eval_step_end(self, run_context):
        self.pbar.update(1)


class EvalSaveImgs(mindspore.Callback):
    """
    Callback to save the output image of the model.
    """

    def __init__(self, save_img_path=None, bgr=False):
        super().__init__()
        self.save_img_path = save_img_path
        self.bgr = bgr
        self.counter = 0

    def on_eval_begin(self, run_context):
        self.counter = 0

    def on_eval_step_end(self, run_context):
        if self.save_img_path:
            pred_imgs = run_context._original_args.net_outputs[-2]
            if pred_imgs.ndim == 5:
                n, d, c, h, w = pred_imgs.shape
                pred_imgs = pred_imgs.reshape(n * d, c, h, w)
            elif pred_imgs.ndim == 3:
                pred_imgs = pred_imgs.unsqueeze(0)
            assert pred_imgs.ndim == 4
            for i in range(pred_imgs.shape[0]):
                pred_name = f"{self.counter:06d}_pred.png"
                self._save_img(pred_imgs[i], pred_name)
                self.counter += 1

    def _save_img(self, image, image_name):
        image = tensor2img(image).asnumpy().squeeze()
        if self.bgr:
            image = image[..., (2, 1, 0)]
        image = Image.fromarray(image, "RGB")
        save_path = os.path.join(self.save_img_path, image_name)
        image.save(save_path)


class ProfileCallback(Callback):
    def __init__(
        self, start, stop, output_path="./profile", by_epoch=True, add_datetime_suffix=False, exit_after=False
    ):
        super().__init__()
        self.start = start
        self.stop = stop
        if add_datetime_suffix:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path + now
        self.profiler = Profiler(start_profile=False, output_path=output_path)
        self.by_epoch = by_epoch
        self.step_counter = 0
        self.exit_after = exit_after

    def _profiler_stop(self):
        print("Finish profiling")
        self.profiler.stop()
        self.profiler.analyse()
        if self.exit_after:
            exit(0)

    def _profiler_start(self):
        print("Start profiling")
        self.profiler.start()

    def on_train_step_begin(self, run_context):
        if not self.by_epoch:
            if self.start == self.step_counter:
                self._profiler_start()

    def on_train_step_end(self, run_context):
        if not self.by_epoch:
            if self.stop == self.step_counter:
                self._profiler_stop()
        self.step_counter += 1

    def on_eval_step_begin(self, run_context):
        if not self.by_epoch:
            if self.start == self.step_counter:
                self._profiler_start()

    def on_eval_step_end(self, run_context):
        if not self.by_epoch:
            if self.stop == self.step_counter:
                self._profiler_stop()
        self.step_counter += 1

    def on_train_epoch_begin(self, run_context):
        if self.by_epoch:
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num
            if epoch_num == self.start:
                self._profiler_start()

    def on_train_epoch_end(self, run_context):
        if self.by_epoch:
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num
            if epoch_num == self.stop:
                self._profiler_stop()
