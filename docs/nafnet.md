# Contents

- [NAFNet Description](#nafnet-description)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [NAFNet Description](#contents)

Although there have been significant advances in the field of image restoration recently, the system complexity of the state-of-the-art (SOTA) methods is increasing as well, which may hinder the convenient analysis and comparison of methods. In this paper, we propose a simple baseline that exceeds the SOTA methods and is computationally efficient. To further simplify the baseline, we reveal that the nonlinear activation functions, e.g. Sigmoid, ReLU, GELU, Softmax, etc. are not necessary: they could be replaced by multiplication or removed. Thus, we derive a Nonlinear Activation Free Network, namely NAFNet, from the baseline. SOTA results are achieved on various challenging benchmarks, e.g. 33.69 dB PSNR on GoPro (for image deblurring), exceeding the previous SOTA 0.38 dB with only 8.4% of its computational costs; 40.30 dB PSNR on SIDD (for image denoising), exceeding the previous SOTA 0.28 dB with less than half of its computational costs.

[Paper](https://arxiv.org/abs/2204.04676): Simple Baselines for Image Restoration

[Original github repository](https://github.com/megvii-research/NAFNet)

# [Dataset](#contents)

This work uses the [GoPro](https://seungjunnah.github.io/Datasets/gopro.html) dataset. The dataset consists of T2103 train and 1111 test images with a resolution of 1280 x 720.

Download [link](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing).

# [Environmental requirements](#contents)

## Ascend 910

- Hardware (Ascend)
    - Prepare hardware environment with Ascend 910 (cann_6.0.0, euler_2.8, py_3.7)
- Framework
    - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) or later

# [Performance](#contents)

## [Training Performance](#contents)

| Parameters                 | NAFNet deblur (8xNPU)           |
|----------------------------|---------------------------------|
| Model Version              | NAFNet                          |
| Resources                  | 1x Ascend 910A                  |
| Uploaded Date              | 06 / 14 / 2023 (month/day/year) |
| MindSpore Version          | 1.9.0                           |
| Dataset                    | GoPro                           |
| Training Parameters        | batch_size=16, 10000 epochs     |
| Optimizer                  | Adam                            |
| Speed                      | 780 ms/step                     |
| Total time                 | 2d 12h 31m                      |
| Checkpoint for Fine tuning | 274.9 MB (.ckpt file)           |

## [Evaluation Performance](#contents)

| Parameters         | NAFNet deblur (1xNPU, CANN)     |
|--------------------|---------------------------------|
| Model Version      | NAFNet                          |
| Resources          | 1x Ascend 910A                  |
| Uploaded Date      | 06 / 14 / 2023 (month/day/year) |
| MindSpore Version  | 1.9.0                           |
| Datasets           | GoPro                           |
| Batch_size         | 1                               |
| Inference speed, s | 0.041 (1280x720)                |
| PSNR metric        | 30.48                           |
| SSIM metric        | 0.9041                          |
