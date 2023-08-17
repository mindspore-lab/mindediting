# Contents

- [VRT Description](#vrt-description)
- [Model-architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [VRT Description](#contents)

Video restoration (e.g., video super-resolution) aims to restore high-quality frames from low-quality frames. Different from single image restoration, video restoration generally requires to utilize temporal information from multiple adjacent but usually misaligned video frames. Existing deep methods generally tackle with this by exploiting a sliding window strategy or a recurrent architecture, which either is restricted by frame-by-frame restoration or lacks long-range modelling ability. In this paper, authors propose a Video Restoration Transformer (VRT) with parallel frame prediction and long-range temporal dependency modelling abilities. Experimental results on three tasks, including video super-resolution, video deblurring and video denoising, demonstrate that VRT outperforms the state-of-the-art methods by large margins (up to 2.16 dB) on nine benchmark datasets.

Currently Video Super Resolution is supported only.

[Paper](https://arxiv.org/abs/2201.12288): VRT: A Video Restoration Transformer.

[Reference github repository (evaluation only)](https://github.com/JingyunLiang/VRT)

[Reference github repository (both training and evaluation)](https://github.com/cszn/KAIR)

# [Model architecture](#contents)

VRT is composed of multiple scales, each of which consists of two kinds of modules: temporal mutual self attention (TMSA) and parallel warping. TMSA divides the video into small clips, on which mutual attention is applied for joint motion estimation, feature alignment and feature fusion, while self-attention is used for feature extraction. To enable cross-clip interactions, the video sequence is shifted for every other layer. Besides, parallel warping is used to further fuse information from neighboring frames by parallel feature warping.

# [Dataset](#contents)

## Dataset used

This work uses the [REDS sharp & sharp_bicubic](https://seungjunnah.github.io/Datasets/reds.html) (266 videos, 266000 frames: train + val except REDS4) for training, and REDS4 (4 videos, 400 frames: 000, 011, 015, 020 of REDS) for testing.

## Dataset preprocessing

Use `src/scripts/regroup_reds_dataset.py` to regroup and rename REDS val set
For better I/O speed, use `src/scripts/create_lmdb.py` to convert `.png` datasets to `.lmdb` datasets.

# [Environmental requirements](#contents)

## GPU

- Hardware (GPU)
    - Prepare hardware environment with GPU processor
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- Additional python packages:
    - Install additional packages manually or using `pip install -r requirements.txt` command in the model directory.

## Ascend 910

- Hardware (Ascend)
    - Prepare hardware environment with Ascend 910 (cann_6.0.0, euler_2.8, py_3.7)
- Framework
    - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) or later

# [Performance](#contents)

## [Training Performance](#contents)

| Parameters                              | VRT (8xGPU)                           | VRT (8xGPU)                            |
|-----------------------------------------|---------------------------------------|----------------------------------------|
| Model Version                           | 001_train_vrt_videosr_bi_reds_6frames | 002_train_vrt_videosr_bi_reds_12frames |
| Resources                               | 8x Nvidia A100                        | 8x Nvidia A100                         |
| Uploaded Date                           | 12 / 22 / 2022 (month/day/year)       | N/A                                    |
| MindSpore Version                       | 1.9.0                                 | 1.9.0                                  |
| Dataset                                 | REDS                                  | REDS                                   |
| Training Parameters                     | batch_size=8, 300,000 steps           | batch_size=8, 300,000 steps            |
| Optimizer                               | Adam                                  | Adam                                   |
| Speed                                   | 2.65 s/step                           | 5.70 s/step                            |
| Total time                              | 9d 15h                                | N/A                                    |
| Checkpoint for Fine tuning              | 157 MB (.ckpt file)                   | 200 MB (.ckpt file)                    |
| GPU memory consumption                  | 29.6 GB                               | 66.8 GB                                |

| Parameters                              | VRT (8xAscend)                        |
|-----------------------------------------|---------------------------------------|
| Model Version                           | 001_train_vrt_videosr_bi_reds_6frames |
| Resources                               | 8x Ascend 910                         |
| Uploaded Date                           | N/A                                   |
| MindSpore Version                       | 2.0.0.20221118                        |
| Dataset                                 | REDS                                  |
| Training Parameters                     | batch_size=8, 300,000 steps           |
| Optimizer                               | Adam                                  |
| Speed                                   | 5.60 s/step                           |
| Total time                              | N/A                                   |

In order to decrease memory consumption one can reduce value of `batch_size` and/or `num_frame`. However it will pobably lead to quality degradataion.

## [Evaluation Performance](#contents)

| Parameters                  | 1xGPU                                 | 1xAscend                              |
|-----------------------------|---------------------------------------|---------------------------------------|
| Model Version               | 001_train_vrt_videosr_bi_reds_6frames | 001_train_vrt_videosr_bi_reds_6frames |
| Resources                   | 1x Nvidia A100                        | 1x Ascend 910                         |
| MindSpore Version           | 1.9.0                                 | 2.0.0.20221118                        |
| Datasets                    | REDS                                  | REDS                                  |
| Batch_size                  | 1                                     | 1                                     |
| num_frame_testing           | 40                                    | 40                                    |
| PSNR metric                 | 31.63                                 | 31.62                                 |
| GPU memory consumption      | 57.6 GB                               | N/A                                   |
| Speed                       | 12.82 s/call                          | 18.12 s/call             |

In order to decrease memory consumption one can reduce value of `num_frame_testing`. However it will pobably lead to quality degradataion.
