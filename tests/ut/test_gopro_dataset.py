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

import math

import pytest
from common import init_test_environment

init_test_environment()

from mindediting.dataset.create_datasets import create_dataset

DATASET_NAME = "gopro"
ROOT_PATH = "/data/LLVT/MIMO-UNet/data/GOPRO_Large/"
BASE_TRAIN_VIDEO_DATASET_LEN = 22
BASE_TRAIN_IMAGE_DATASET_LEN = 2103


@pytest.mark.parametrize("split", ["val", "train"])
def test_can_create_dataset(split):
    dataset = create_dataset(name=DATASET_NAME, root=ROOT_PATH, split=split)
    assert dataset.source_len > 0


def test_can_not_create_dataset_with_wrong_split():
    split = "test"
    with pytest.raises(ValueError):
        create_dataset(name=DATASET_NAME, root=ROOT_PATH, split=split)


@pytest.mark.parametrize("every_nth", [1, 7, 10])
def test_can_get_every_nth_video_mode(every_nth):
    dataset = create_dataset(name=DATASET_NAME, root=ROOT_PATH, split="train", video_mode=True, every_nth=every_nth)
    assert dataset.source_len == math.ceil(BASE_TRAIN_VIDEO_DATASET_LEN / every_nth)


@pytest.mark.parametrize("every_nth", [1, 100])
def test_can_get_every_nth_image_mode(every_nth):
    dataset = create_dataset(name=DATASET_NAME, root=ROOT_PATH, split="train", video_mode=False, every_nth=every_nth)
    assert dataset.source_len == math.ceil(BASE_TRAIN_IMAGE_DATASET_LEN / every_nth)
