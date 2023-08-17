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

from mindediting.scheduler.scheduler_factory import create_scheduler

_module_to_schedulers = defaultdict(set)
_scheduler_to_module = {}
_scheduler_entrypoints = {}


def register_scheduler(fn):
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
    _scheduler_entrypoints[model_name] = fn
    _scheduler_to_module[model_name] = module_name
    _module_to_schedulers[module_name].add(model_name)

    return fn


def default_scheduler(cfg):
    ignore_params = getattr(cfg.scheduler, "ignore_params", None)
    lr_scheduler = create_scheduler(
        cfg.steps_per_epoch,
        scheduler=cfg.scheduler.name,
        lr=cfg.scheduler.base_lr,
        min_lr=cfg.scheduler.min_lr,
        decay_epochs=max(cfg.train_params.epoch_size - cfg.scheduler.warmup_epochs, 0),
        warmup_epochs=min(cfg.scheduler.warmup_epochs, cfg.train_params.epoch_size),
        warmup_factor=cfg.scheduler.warmup_factor,
        warmup_base_lr=cfg.scheduler.warmup_base_lr,
        num_epochs=cfg.train_params.epoch_size,
    )
    optimizer_params = [p for p in cfg.net.get_parameters()]
    if ignore_params:
        optimizer_params = [p for p in optimizer_params if ignore_params not in p.name]
    optimizer_params = [{"params": optimizer_params, "lr": lr_scheduler, "weight_decay": cfg.optimizer.weight_decay}]
    return optimizer_params, cfg.optimizer.learning_rate


def default_scheduler_with_two_param_groups(cfg):
    net_lr = create_scheduler(
        cfg.steps_per_epoch,
        scheduler=cfg.scheduler.name,
        lr=cfg.scheduler.base_lr,
        min_lr=cfg.scheduler.min_lr,
        decay_epochs=max(cfg.train_params.epoch_size - cfg.scheduler.warmup_epochs, 0),
        warmup_epochs=min(cfg.scheduler.warmup_epochs, cfg.train_params.epoch_size),
        warmup_factor=cfg.scheduler.warmup_factor,
        warmup_base_lr=cfg.scheduler.warmup_base_lr,
        num_epochs=cfg.train_params.epoch_size,
    )

    extra_lr_scheduler = create_scheduler(
        cfg.steps_per_epoch,
        scheduler=cfg.extra_scheduler.name,
        lr=cfg.scheduler.base_lr * cfg.extra_scheduler.lr_mul,
        min_lr=cfg.scheduler.min_lr,
        decay_epochs=max(cfg.train_params.epoch_size - cfg.extra_scheduler.warmup_epochs, 0),
        warmup_epochs=min(cfg.extra_scheduler.warmup_epochs, cfg.train_params.epoch_size),
        warmup_factor=cfg.extra_scheduler.warmup_factor,
        warmup_base_lr=cfg.extra_scheduler.warmup_base_lr,
        num_epochs=cfg.train_params.epoch_size,
    )
    # creat optimizer
    net_params = []
    extra_params_group = []
    for p in cfg.net.get_parameters():
        if p.requires_grad:
            if any(p.name.startswith(x) for x in cfg.extra_scheduler.param_prefixes):
                extra_params_group.append(p)
            else:
                net_params.append(p)
    optimizer_params = [
        {"params": net_params, "lr": net_lr, "weight_decay": cfg.optimizer.weight_decay},
        {"params": extra_params_group, "lr": extra_lr_scheduler, "weight_decay": cfg.optimizer.weight_decay},
    ]
    return optimizer_params, cfg.optimizer.learning_rate


@register_scheduler
def basicvsr(cfg):
    return default_scheduler_with_two_param_groups(cfg)


@register_scheduler
def basicvsr_plus_plus_light(cfg):
    return default_scheduler(cfg)


@register_scheduler
def nafnet(cfg):
    return default_scheduler(cfg)


@register_scheduler
def ipt(cfg):
    return default_scheduler(cfg)


@register_scheduler
def ttvsr(cfg):
    return default_scheduler_with_two_param_groups(cfg)


@register_scheduler
def vrt(cfg):
    return default_scheduler_with_two_param_groups(cfg)


@register_scheduler
def rvrt(cfg):
    return basicvsr(cfg)


@register_scheduler
def rvrt_light(cfg):
    return basicvsr_plus_plus_light(cfg)


@register_scheduler
def mimo_unet(cfg):
    return default_scheduler(cfg)


@register_scheduler
def fsrcnn(cfg):
    return default_scheduler(cfg)


@register_scheduler
def ctsdg(cfg):
    optimizer_params = {
        "params_generator": {"params": cfg.net.generator.trainable_params(), "lr": cfg.optimizer.gen_lr_train},
        "params_discriminator": {
            "params": cfg.net.discriminator.trainable_params(),
            "lr": cfg.optimizer.dis_lr_multiplier * cfg.optimizer.gen_lr_train,
        },
    }
    return optimizer_params, cfg.optimizer.learning_rate


@register_scheduler
def noahtcv(cfg):
    lr_0 = [cfg.scheduler.base_lr for i in range(cfg.steps_per_epoch * cfg.train_params.epoch_size // 4)]
    lr_1 = create_scheduler(
        int(cfg.steps_per_epoch * 0.75),
        scheduler=cfg.scheduler.name,
        lr=cfg.scheduler.base_lr,
        min_lr=cfg.scheduler.min_lr,
        decay_epochs=max(cfg.train_params.epoch_size - cfg.scheduler.warmup_epochs, 0),
        warmup_epochs=min(cfg.scheduler.warmup_epochs, cfg.train_params.epoch_size),
        warmup_factor=cfg.scheduler.warmup_factor,
        num_epochs=cfg.train_params.epoch_size,
    )

    net_lr = lr_0 + lr_1
    cfg.optimizer.learning_rate = net_lr
    return cfg.net.trainable_params(), cfg.optimizer.learning_rate


@register_scheduler
def emvd(cfg):
    return default_scheduler(cfg)


@register_scheduler
def ifr_plus(cfg):
    return default_scheduler(cfg)


@register_scheduler
def rrdb(cfg):
    return default_scheduler(cfg)


@register_scheduler
def srdiff(cfg):
    return default_scheduler(cfg)


def create_scheduler_by_name(model_name, cfg):
    if model_name not in _scheduler_entrypoints:
        raise Exception
    model = _scheduler_entrypoints[model_name](cfg)
    return model


if __name__ == "__main__":
    model_name_test = "VRT"
    model_test = create_scheduler_by_name(model_name_test)
    print("model", model_test)
