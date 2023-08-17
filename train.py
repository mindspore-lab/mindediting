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

import argparse
import os
import shutil

import mindspore
from mindspore import CheckpointConfig
from tensorboardX import SummaryWriter

from mindediting import create_trainer_by_name
from mindediting.dataset.create_loaders import create_data_loader
from mindediting.deploy.utils.config import Config, merge, parse_cli_to_yaml, parse_yaml
from mindediting.loss import create_loss
from mindediting.models import create_model_by_name
from mindediting.optim import create_optimizer
from mindediting.scheduler import create_scheduler_by_name
from mindediting.utils.callbacks import EvalAsInValPyCallBack, ProfileCallback, TrainingMonitor
from mindediting.utils.device_adapter import get_device_id, get_device_num
from mindediting.utils.init_utils import init_env


def train(cfg, profile):
    init_env(cfg)
    # create dataset
    rank_id = get_device_id()
    group_size = get_device_num()

    # create train data loader
    loader_train = create_data_loader(
        cfg.model.name,
        cfg.dataset,
        cfg.train_pipeline,
        "train",
        cfg.dataset.batch_size,
    )

    # create model
    cfg.mode = "train"
    net, eval_network = create_model_by_name(model_name=cfg.model.name, cfg=cfg)
    cfg.net, cfg.steps_per_epoch = net, loader_train.get_dataset_size()

    # create loss
    loss = create_loss(loss_name=cfg.loss.name, **cfg.loss.cfg_dict)

    # create learning rate schedule
    optimizer_params, learning_rate = create_scheduler_by_name(model_name=cfg.model.name, cfg=cfg)

    # create optimizer
    loss_scale = 1.0 if cfg.loss.amp_level == "O0" else cfg.loss.loss_scale
    optimizer = create_optimizer(
        params=optimizer_params,
        lr=learning_rate,
        opt=cfg.optimizer.name,
        loss_scale=loss_scale,
        **{"beta1": cfg.optimizer.beta1, "beta2": cfg.optimizer.beta2}
    )

    # define callbacks
    save_checkpoint_steps = cfg.train_params.save_epoch_frq * loader_train.get_dataset_size()
    config_ck = CheckpointConfig(
        save_checkpoint_steps=save_checkpoint_steps, keep_checkpoint_max=cfg.train_params.keep_checkpoint_max
    )
    summary_writer = None
    if rank_id == 0:
        summary_writer = SummaryWriter(os.path.join(cfg.train_params.ckpt_save_dir, "summary"))
    callbacks = [mindspore.TimeMonitor()]
    if cfg.train_params.need_val:
        callbacks.append(EvalAsInValPyCallBack(cfg, net, eval_network, summary_writer=summary_writer))
    if profile:
        callbacks.append(ProfileCallback(**cfg.train_params.profile.cfg_dict))
    if rank_id == 0:
        callbacks.append(
            TrainingMonitor(
                cfg.train_params.epoch_size,
                cfg.steps_per_epoch,
                print_frequency=cfg.train_params.print_frequency,
                summary_writer=summary_writer,
            )
        )
        callbacks.append(
            mindspore.ModelCheckpoint(
                prefix=cfg.model.name + "_" + cfg.dataset.dataset_name,
                directory=cfg.train_params.ckpt_save_dir,
                config=config_ck,
            )
        )

    # define trainer
    trainer = create_trainer_by_name(cfg.model.name, net, loss, optimizer, cfg)
    initial_epoch = cfg.train_params.get("initial_epoch", 0)
    print(" training...")
    trainer.train(
        cfg.train_params.epoch_size,
        loader_train,
        dataset_sink_mode=cfg.dataset.dataset_sink_mode,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
    )

    if summary_writer:
        summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", type=str, help="Config file path", required=True)
    parser.add_argument("--profile", action="store_true", help="Enables profile callback.")
    known_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(known_args.config_path)
    args = parse_cli_to_yaml(
        parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=known_args.config_path
    )
    final_config = merge(args, default)
    final_config = Config(final_config)
    if get_device_id() == 0:
        os.makedirs(final_config.train_params.ckpt_save_dir, exist_ok=True)
        try:
            shutil.copy(known_args.config_path, final_config.train_params.ckpt_save_dir)
        except shutil.SameFileError:
            pass
    train(final_config, known_args.profile)
