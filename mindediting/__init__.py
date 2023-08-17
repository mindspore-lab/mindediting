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

import sys
from collections import defaultdict

import mindspore
import numpy as np
from mindspore import nn

from mindediting.models.image_inpainting.ctsdg.trainer import (
    CTSDGTrainer,
    DTrainOneStepCell,
    DWithLossCell,
    GTrainOneStepCell,
    GWithLossCell,
)
from mindediting.models.mutil_task.ipt import IPT, IPTWithLossCell
from mindediting.models.mutil_task.ipt_trainer import IPTTrainOneStepWithLossScaleCell
from mindediting.models.video_denoise.emvd import EMVDWithLossCell
from mindediting.utils.cast_net_and_loss import cast_net_and_loss, prepare_model_kwargs
from mindediting.utils.ipt_validator import IPTValidator
from mindediting.utils.mpfer_validator import MpferValidator
from mindediting.utils.tiling_val import TilingValidator
from mindediting.utils.utils import cast_module, is_ascend

_module_to_trainers = defaultdict(set)
_trainer_to_module = {}
_trainer_entrypoints = {}

_module_to_validators = defaultdict(set)
_validator_to_module = {}
_validator_entrypoints = {}


def register_trainer(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split(".")
    module_name = module_name_split[-1] if len(module_name_split) else ""

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _trainer_entrypoints[model_name] = fn
    _trainer_to_module[model_name] = module_name
    _module_to_trainers[module_name].add(model_name)
    return fn


def register_validator(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split(".")
    module_name = module_name_split[-1] if len(module_name_split) else ""

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _validator_entrypoints[model_name] = fn
    _validator_to_module[model_name] = module_name
    _module_to_validators[module_name].add(model_name)
    return fn


def base_trainer(net, loss, optimizer, cfg):
    loss_scale_manager = mindspore.DynamicLossScaleManager() if cfg.system.device_target == "Ascend" else None
    trainer = mindspore.train.model.Model(
        net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager, amp_level=cfg.loss.amp_level
    )
    return trainer


@register_trainer
def basicvsr(net, loss, optimizer, cfg):
    return base_trainer(net, loss, optimizer, cfg)


@register_trainer
def basicvsr_plus_plus_light(net, loss, optimizer, cfg):
    return base_trainer(net, loss, optimizer, cfg)


@register_trainer
def nafnet(net, loss, optimizer, cfg):
    return base_trainer(net, loss, optimizer, cfg)


@register_trainer
def ipt(net, loss, optimizer, cfg):
    loss_scale_manager = nn.DynamicLossScaleUpdateCell(
        loss_scale_value=cfg.loss.init_loss_scale, scale_factor=2, scale_window=1000
    )
    net_with_loss = IPTWithLossCell(net, loss, use_con=cfg.loss.con_loss)
    net_with_loss = IPTTrainOneStepWithLossScaleCell(net_with_loss, optimizer, loss_scale_manager)
    trainer = mindspore.train.model.Model(net_with_loss)
    return trainer


@register_trainer
def ttvsr(net, loss, optimizer, cfg):
    return base_trainer(net, loss, optimizer, cfg)


@register_trainer
def mimo_unet(net, loss, optimizer, cfg):
    model_kwargs = {}
    if cfg.system.device_target == "Ascend":
        # Not fully tested with CPU and GPU
        cast_net_and_loss(cfg, net, loss)
        model_kwargs = prepare_model_kwargs(cfg, net, loss)
    trainer = mindspore.train.model.Model(net, loss_fn=loss, optimizer=optimizer, **model_kwargs)
    return trainer


@register_trainer
def fsrcnn(net, loss, optimizer, cfg):
    model_kwargs = {}
    if cfg.system.device_target == "Ascend":
        # Not fully tested with CPU and GPU
        cast_net_and_loss(cfg, net, loss)
        model_kwargs = prepare_model_kwargs(cfg, net, loss)
    trainer = mindspore.train.model.Model(net, loss_fn=loss, optimizer=optimizer, **model_kwargs)
    return trainer


@register_trainer
def ctsdg(net, loss, optimizer, cfg):
    if cfg.model.is_train_finetune:
        cfg.train.total_steps = cfg.train.finetune_iter
    else:
        cfg.train.total_steps = cfg.train.train_iter
    generator, discriminator, vgg16_feature_extractor = (
        net.generator,
        net.discriminator,
        net.vgg16_feature_extractor,
    )
    generator_w_loss = GWithLossCell(generator, discriminator, vgg16_feature_extractor, cfg)
    discriminator_w_loss = DWithLossCell(discriminator)
    generator_t_step = GTrainOneStepCell(generator_w_loss, optimizer.get("generator"))
    discriminator_t_step = DTrainOneStepCell(discriminator_w_loss, optimizer.get("discriminator"))
    trainer = CTSDGTrainer(generator_t_step, discriminator_t_step, cfg, cfg.model.is_train_finetune)
    return trainer


@register_trainer
def noahtcv(net, loss, optimizer, cfg):
    return base_trainer(net, loss, optimizer, cfg)


@register_trainer
def emvd(net, loss, optimizer, cfg):
    loss_scale_manager = mindspore.DynamicLossScaleManager() if cfg.system.device_target == "Ascend" else None
    net_with_loss = EMVDWithLossCell(backbone=net, loss_fn=loss, frame_num=cfg.dataset.num_frames)
    trainer = mindspore.train.model.Model(
        net_with_loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager, amp_level=cfg.loss.amp_level
    )
    return trainer


@register_trainer
def vrt(net, loss, optimizer, cfg):
    class RelativeBiasToTableCallback(mindspore.ModelCheckpoint):
        def __init__(self, a) -> None:
            super().__init__(a._prefix, a._directory, a._config)

        def on_train_step_end(self, run_context):
            cb_params = run_context.original_args()
            cb_params.train_network.network._backbone.relative_position_bias_to_table()

    class VrtTrainer(mindspore.train.model.Model):
        def train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1, initial_epoch=0):
            checkpoint_callback = [
                callback for callback in callbacks if isinstance(callback, mindspore.ModelCheckpoint)
            ]
            if checkpoint_callback:
                callbacks = [RelativeBiasToTableCallback(checkpoint_callback[0])] + callbacks
            super().train(
                epoch,
                train_dataset,
                callbacks=callbacks,
                dataset_sink_mode=dataset_sink_mode,
                sink_size=sink_size,
                initial_epoch=initial_epoch,
            )

    cast_net_and_loss(cfg, net, loss)
    model_kwargs = prepare_model_kwargs(cfg, net, loss)
    trainer = VrtTrainer(net, loss_fn=loss, optimizer=optimizer, **model_kwargs)
    return trainer


@register_trainer
def rrdb(net, loss, optimizer, cfg):
    loss_scale_manager = mindspore.DynamicLossScaleManager() if cfg.system.device_target == "Ascend" else None
    trainer = mindspore.train.model.Model(
        net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale_manager, amp_level=cfg.loss.amp_level
    )
    return trainer


@register_trainer
def srdiff(net, loss, optimizer, cfg):
    loss_scale_manager = mindspore.DynamicLossScaleManager() if cfg.system.device_target == "Ascend" else None
    trainer = mindspore.train.model.Model(
        net, optimizer=optimizer, loss_scale_manager=loss_scale_manager, amp_level=cfg.loss.amp_level
    )
    return trainer


@register_trainer
def rvrt(net, loss, optimizer, cfg):
    return vrt(net, loss, optimizer, cfg)


@register_trainer
def rvrt_light(net, loss, optimizer, cfg):
    return vrt(net, loss, optimizer, cfg)


@register_trainer
def ifr_plus(net, loss, optimizer, cfg):
    cast_net_and_loss(cfg, net, loss)
    model_kwargs = prepare_model_kwargs(cfg, net, loss)
    trainer = mindspore.train.model.Model(net, loss_fn=loss, optimizer=optimizer, **model_kwargs)
    return trainer


def create_trainer_by_name(model_name, net, loss, optimizer, cfg):
    model_name = model_name.lower()
    if model_name not in _trainer_entrypoints:
        raise Exception
    trainer = _trainer_entrypoints[model_name](net, loss, optimizer, cfg)
    return trainer


@register_validator
def default(net, eval_network, loss, metrics):
    validator = mindspore.train.model.Model(net, eval_network=eval_network, loss_fn=loss, metrics=metrics)
    return validator


@register_validator
def tiling(net, eval_network, loss, metrics, **kwargs):
    validator = TilingValidator(net, eval_network=eval_network, loss_fn=loss, metrics=metrics, **kwargs)
    return validator


@register_validator
def mpfer(net, eval_network, loss, metrics, **kwargs):
    validator = MpferValidator(net, eval_network=eval_network, loss_fn=loss, metrics=metrics, **kwargs)
    return validator


@register_validator
def ipt_validator(net, eval_network, loss, metrics, **kwargs):
    validator = IPTValidator(net, eval_network=eval_network, loss_fn=loss, metrics=metrics, **kwargs)
    return validator


@register_validator
def om_default(eval_network, metrics, **kwargs):
    class OmDefaultValidator:
        def __init__(self, net, metrics):
            self.net = net
            self.metrics = metrics

        def eval(self, loader_val, dataset_sink_mode, callbacks):
            for i in range(len(callbacks)):
                callbacks[i].on_eval_begin(None)

            for inputs in loader_val:
                if len(inputs) and not isinstance(inputs[0], np.ndarray):
                    inputs = [x.asnumpy() for x in inputs]
                if len(inputs) == 2:
                    lq, hq = inputs
                    model_inputs = [lq]
                elif len(inputs) == 3:
                    lq, hq, lq_up = inputs
                    model_inputs = [lq, lq_up]
                else:
                    raise ValueError(f"Supported only 2 and 3 inputs but got {len(inputs)}")

                for i in range(len(callbacks)):
                    callbacks[i].on_eval_step_begin(None)

                hr = self.net([model_inputs])
                while isinstance(hr, list):
                    hr = hr[0]

                for i in range(len(callbacks)):
                    callbacks[i].on_eval_step_end(None)
                for k in self.metrics:
                    self.metrics[k].update(hr, hq)

            for i in range(len(callbacks)):
                callbacks[i].on_eval_end(None)

    validator = OmDefaultValidator(eval_network, metrics)
    return validator


def create_validator(validator_cfg, net, eval_network, loss, metrics):
    validator_params = dict(validator_cfg.cfg_dict)
    validator_name = validator_params.pop("name")
    if validator_name not in _validator_entrypoints:
        raise Exception

    validator = _validator_entrypoints[validator_name](
        net=net, eval_network=eval_network, loss=loss, metrics=metrics, **validator_params
    )
    return validator
