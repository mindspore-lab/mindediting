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
Create dataset by name
"""

from typing import Optional

import mindspore.dataset as ds
from mindspore.dataset import DIV2KDataset, Places365Dataset

from ..utils.local_adapter import get_device_id, get_device_num, get_rank_id
from .src.cbsd68 import create_cbsd68_dataset
from .src.crvd import create_crvd_dataset
from .src.davis2017 import create_davis2017_dataset
from .src.deep_video_deblurring import create_dvd_dataset
from .src.df2k import create_df2k_dataset
from .src.div2k import create_div2k_dataset
from .src.gopro import create_gopro_dataset
from .src.imagenet import create_imagenet_dataset
from .src.kodak import create_kodak_dataset
from .src.ldv_v2 import create_ldv_video_enhancement_dataset
from .src.mai21denoise import create_mai21denoise_dataset
from .src.mask_datasets import create_mask_dataset
from .src.mpfer_datasets import create_space_dataset
from .src.rain100 import create_rain100_dataset
from .src.reds import create_reds_dataset
from .src.reds_deblur import create_reds_deblur_dataset
from .src.set8 import create_set8_dataset
from .src.sidd import create_sidd_dataset
from .src.srf_datasets import create_srf_dataset
from .src.t91image import create_h5_t91image_dataset
from .src.udm10 import create_udm10_dataset
from .src.vid4 import create_vid4_dataset
from .src.vimeo90k import (
    create_vimeo_blur_dataset,
    create_vimeo_deblocking_dataset,
    create_vimeo_denoising_dataset,
    create_vimeo_super_resolution_dataset,
    create_vimeo_super_resolution_test_dataset,
    create_vimeo_triplet,
)

ds.config.set_enable_watchdog(False)
__all__ = ["create_dataset"]

_MINDSPORE_BASIC_DATASET = dict(
    div2k=create_div2k_dataset,
    df2k=create_df2k_dataset,
    places365=Places365Dataset,
    gopro=create_gopro_dataset,
    cbsd68=create_cbsd68_dataset,
    davis2017=create_davis2017_dataset,
    set8=create_set8_dataset,
    dvd=create_dvd_dataset,
    mai21denoise=create_mai21denoise_dataset,
    rain100=create_rain100_dataset,
    reds=create_reds_dataset,
    reds_deblur=create_reds_deblur_dataset,
    set5=create_srf_dataset,
    space=create_space_dataset,
    set14=create_srf_dataset,
    bsd100=create_srf_dataset,
    urban100=create_srf_dataset,
    udm10=create_udm10_dataset,
    vid4=create_vid4_dataset,
    vimeo_blur=create_vimeo_blur_dataset,
    vimeo_deblocking=create_vimeo_deblocking_dataset,
    vimeo_denoising=create_vimeo_denoising_dataset,
    vimeo_super_resolution=create_vimeo_super_resolution_dataset,
    vimeo_super_resolution_test=create_vimeo_super_resolution_test_dataset,
    vimeo_triplet=create_vimeo_triplet,
    paris_street_view=create_mask_dataset,
    places2=create_mask_dataset,
    celeba=create_mask_dataset,
    imagenet=create_imagenet_dataset,
    t91image=create_h5_t91image_dataset,
    crvd=create_crvd_dataset,
    sidd=create_sidd_dataset,
    kodak=create_kodak_dataset,
    ldv_v2=create_ldv_video_enhancement_dataset,
)


def create_dataset(
    name: str = "",
    root: str = "./",
    split: str = "train",
    shuffle: bool = True,
    num_samples: Optional[bool] = None,
    num_parallel_workers: Optional[int] = 1,
    **kwargs,
):
    r"""Creates dataset by name.

    Args:
        name: dataset name like MNIST, CIFAR10, ImageNeT, ''. '' means a customized dataset. Default: ''.
        root: dataset root dir. Default: './'.
        split: data split: '' or split name string (train/val/test), if it is '', no split is used.
            Otherwise, it is a subfolder of root dir, e.g., train, val, test. Default: 'train'.
        shuffle: whether to shuffle the dataset. Default: True.
        num_samples: Number of elements to sample (default=None, which means sample all elements).
            This argument can only be specified when `num_shards` is also specified.
        num_parallel_workers: Number of workers to read the data (default=None, set in the config).
        download: whether to download the dataset. Default: False

    Note:
        For custom datasets and imagenet, the dataset dir should follow the structure like:
        .dataset_name/
        ├── split1/
        │  ├── class1/
        │  │   ├── 000001.jpg
        │  │   ├── 000002.jpg
        │  │   └── ....
        │  └── class2/
        │      ├── 000001.jpg
        │      ├── 000002.jpg
        │      └── ....
        └── split2/
           ├── class1/
           │   ├── 000001.jpg
           │   ├── 000002.jpg
           │   └── ....
           └── class2/
               ├── 000001.jpg
               ├── 000002.jpg
               └── ....

    Returns:
        Dataset object
    """

    if num_samples is None:
        sampler = None
    elif num_samples > 0:
        if shuffle:
            sampler = ds.RandomSampler(replacement=False, num_samples=num_samples)
        else:
            sampler = ds.SequentialSampler(num_samples=num_samples)
        shuffle = None  # shuffle and sampler cannot be set at the same in mindspore datatset API
    else:
        sampler = None

    name = name.lower()
    if name in _MINDSPORE_BASIC_DATASET:
        dataset_class = _MINDSPORE_BASIC_DATASET[name]
        dataset = dataset_class(
            dataset_dir=root,
            usage=split,
            num_samples=num_samples,
            num_parallel_workers=num_parallel_workers,
            shuffle=shuffle,
            sampler=sampler,
            num_shards=get_device_num(),
            shard_id=get_rank_id() % get_device_num(),
            **kwargs,
        )
    else:
        raise ValueError(f"{name} dataset is not supported.")
    return dataset


if __name__ == "__main__":
    create_dataset(name="gopro", root="/home/ma-user/work/code/projects/LLVT_projects/LLVT", download=False)
