# Contents

- [IFR+ Description](#ifrplus-description)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [IFR+ Description](#contents)

## IFR+ Description

Improved version of [IFRNet](https://arxiv.org/abs/2205.14620) model: combines both promising approaches like optical flow based backward warping and kernel-based image refining. The proposed network architecture allowed to beat SOTA on Vimeo-triplet dataset.

## IFRNet Description (CVPR2022)

Prevailing video frame interpolation algorithms, that generate the intermediate frames from consecutive inputs, typically rely on complex model architectures with heavy parameters or large delay, hindering them from diverse real-time applications. In this work, we devise an efficient encoder-decoder based network, termed IFRNet, for fast intermediate frame synthesizing. It first extracts pyramid features from given inputs, and then refines the bilateral intermediate flow fields together with a powerful intermediate feature until generating the desired output. The gradually refined intermediate feature can not only facilitate intermediate flow estimation, but also compensate for contextual details, making IFRNet do not need additional synthesis or refinement module. To fully release its potential, we further propose a novel task-oriented optical flow distillation loss to focus on learning the useful teacher knowledge towards frame synthesizing. Meanwhile, a new geometry consistency regularization term is imposed on the gradually refined intermediate features to keep better structure layout. Experiments on various benchmarks demonstrate the excellent performance and fast inference speed of proposed approaches.

[Paper](https://arxiv.org/abs/2205.14620): IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation

[Original github repository](https://github.com/ltkong218/ifrnet)

# [Dataset](#contents)

This work uses the [Vimeo-triplet](http://toflow.csail.mit.edu/) dataset. The dataset consists of 73,171 3-frame sequences with a fixed resolution of 448 x 256, extracted from 15K selected video clips from [Vimeo-90K](http://toflow.csail.mit.edu/). This dataset is designed for temporal frame interpolation.

Download [link](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip).

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
    - [MindSpore Ascend 1.9.0](https://www.mindspore.cn/install) or later

## [Performance](#contents)

### [Training Performance](#contents)

| Parameters                 | IFR+ (4xGPU)                    |
|----------------------------|---------------------------------|
| Model Version              | IFR+                            |
| Resources                  | 4x Nvidia V100                  |
| Uploaded Date              | 05 / 26 / 2023 (month/day/year) |
| MindSpore Version          | 2.0.0.RC1                       |
| Dataset                    | Vimeo-triplet                   |
| Training Parameters        | batch_size=6, 100 epochs        |
| Optimizer                  | AdamW                           |
| Speed                      | 1.89 s/step                     |
| Total time                 | 4d 17h 11m                      |
| Checkpoint for Fine tuning | 413.0 MB (.ckpt file)           |

### [Evaluation Performance](#contents)

| Parameters             | IFR+ (1xNPU, CANN)              |
|------------------------|---------------------------------|
| Model Version          | IFR+                            |
| Resources              | 1x Ascend 910A                  |
| Uploaded Date          | 05 / 26 / 2023 (month/day/year) |
| MindSpore Version      | 2.0.0.20221118                  |
| Datasets               | Vimeo-triplet                   |
| Batch_size             | 1                               |
| Inference speed, s     | 0.15 (448x256)                  |
| Inference speed, s     | 2.00 (1920x1024)                |
| PSNR metric (exported) | 36.54                           |
| SSIM metric (exported) | 0.9710                          |
