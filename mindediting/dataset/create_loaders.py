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

"""
Create dataloader
"""
__all__ = ["create_loader", "create_data_loader"]

from .create_datasets import create_dataset
from .create_transforms import create_transform


def create_loader(
    dataset,
    batch_size,
    split="val",
    operations=None,
    input_columns=None,
    output_columns=None,
    num_parallel_workers=4,
    max_rowsize=16,
    python_multiprocessing=False,
):
    drop_remainder = True if split == "train" else False
    dataset = dataset.project(columns=input_columns)

    if operations:
        dataset = dataset.batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            num_parallel_workers=num_parallel_workers,
            max_rowsize=max_rowsize,
            per_batch_map=operations,
            input_columns=input_columns,
            output_columns=output_columns,
            python_multiprocessing=python_multiprocessing,
        )
    else:
        dataset = dataset.batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
            python_multiprocessing=python_multiprocessing,
        )
    return dataset


def create_data_loader(model_name, cfg, pipeline_cfg, split, batch_size):
    cfg_dict = cfg.cfg_dict
    shuffle = split == "train"

    if pipeline_cfg is None or isinstance(pipeline_cfg, list):
        transforms_pipeline_type = model_name
    else:
        transforms_pipeline_type = pipeline_cfg.name
        pipeline_cfg = pipeline_cfg.pipeline

    operations, input_columns, output_columns = create_transform(
        model_name=transforms_pipeline_type, split=split, pipeline=pipeline_cfg, **cfg_dict
    )

    if cfg_dict.get("transform_while_batching", True):
        dataset = create_dataset(
            name=cfg.dataset_name,
            root=cfg.input_path,
            split=split,
            shuffle=shuffle,
            nframes=cfg.get("num_frames", None),
            **cfg_dict
        )

        loader = create_loader(
            dataset=dataset,
            batch_size=batch_size,
            operations=operations,
            input_columns=input_columns,
            output_columns=output_columns,
            split=split,
            num_parallel_workers=cfg_dict.get("num_parallel_workers", 4),
            max_rowsize=cfg_dict.get("max_rowsize", 16),
            python_multiprocessing=cfg_dict.get("python_multiprocessing", False),
        )
    else:
        dataset = create_dataset(
            name=cfg.dataset_name,
            root=cfg.input_path,
            split=split,
            shuffle=shuffle,
            nframes=cfg.get("num_frames", None),
            transforms=operations,
            column_names=output_columns,
            **cfg_dict
        )

        loader = create_loader(
            dataset=dataset,
            batch_size=batch_size,
            input_columns=output_columns,
            split=split,
        )
    return loader
