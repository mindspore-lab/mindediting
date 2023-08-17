# 内容

- [内容](#内容)
- [VRT 简介](#vrt-简介)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [模型介绍](#模型介绍)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo 首页](#modelzoo-首页)

# [VRT 简介](#内容)

视频恢复(如视频超分辨率)旨在从低质量帧中恢复高质量帧。与单一图像恢复不同，视频恢复通常需要利用多个相邻但通常不对齐的视频帧的时间信息。现有的深度方法通常通过利用滑动窗口策略或循环架构来解决这个问题，这要么受到逐帧恢复的限制，要么缺乏远程建模能力。本文作者提出了一种具有并行帧预测和长程时间依赖建模能力的视频恢复Transformer (VRT)。在视频超分辨率、视频去模糊和视频去噪等三个任务上的实验结果表明，VRT在9个基准数据集上的表现优于最先进的方法(高达2.16 dB)。

目前只支持视频超分辨率。

[论文](https://arxiv.org/abs/2201.12288): VRT: A Video Restoration Transformer.

[参考 github 仓库 (仅评估)](https://github.com/JingyunLiang/VRT)

[参考 github 仓库  (训练和评估)](https://github.com/cszn/KAIR)

# [模型架构](#内容)

VRT由多个尺度组成，每个尺度由两种模块组成:时间相互自我注意(TMSA)和并行扭曲。TMSA将视频分割成小片段，在小片段上利用相互注意进行关节运动估计、特征对齐和特征融合，同时利用自注意进行特征提取。为了实现跨剪辑交互，视频序列将为每一层移动。此外，还采用了平行规整，通过平行特征规整进一步融合相邻帧的信息。

# [数据集](#内容)

## 使用数据集

本工作使用 [REDS sharp & sharp_bicubic](https://seungjunnah.github.io/Datasets/reds.html) (266个视频，266000帧:除REDS4外的train + val)进行训练， 并使用REDS4(4个视频，400帧:000,011,015,020 of REDS)进行测试。

## 数据集预处理

使用`src/scripts/regroup_reds_dataset.py `重新分组并重命名REDS 验证集。
为了获得更好的I/O速度，使用`src/scripts/create_lmdb.py `将` .png `数据集转换为`.lmdb`数据集。

# [环境要求](#内容)

## GPU

- 硬件 (GPU)
    - 准备GPU处理器硬件
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 具体请参见以下资源:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 其他python包:
    - 手动或在模型目录下使用` pip Install -r requirements.txt `命令安装其他包。

## Ascend 910

- 硬件 (Ascend)
    - 准备 Ascend 910 硬件环境 (cann_6.0.0, euler_2.8, py_3.7)
- 框架
    - [MindSpore Ascend 1.9.0](https://www.mindspore.cn/install) or 更新

# [模型介绍](#内容)

## [性能](#内容)

### 训练性能

| 参数           | VRT (8xGPU)                           | VRT (8xGPU)                            |
| -------------- | ------------------------------------- | -------------------------------------- |
| 模型参数       | 001_train_vrt_videosr_bi_reds_6frames | 002_train_vrt_videosr_bi_reds_12frames |
| 资源           | 8x Nvidia A100                        | 8x Nvidia A100                         |
| 上传日期       | 12 / 22 / 2022 (month/day/year)       | N/A                                    |
| MindSpore 版本 | 1.9.0                                 | 1.9.0                                  |
| 数据集         | REDS                                  | REDS                                   |
| 训练参数       | batch_size=8, 300,000 steps           | batch_size=8, 300,000 steps            |
| 优化器         | Adam                                  | Adam                                   |
| 速度           | 2.65 s/step                           | 5.70 s/step                            |
| 总耗时         | 9d 15h                                | N/A                                    |
| 微调参数文件   | 157 MB (.ckpt file)                   | 200 MB (.ckpt file)                    |
| GPU内存消耗    | 29.6 GB                               | 66.8 GB                                |

| 参数           | VRT (8xAscend)                        |
| -------------- | ------------------------------------- |
| 模型版本       | 001_train_vrt_videosr_bi_reds_6frames |
| 资源           | 8x Ascend 910                         |
| 上传日期       | N/A                                   |
| MindSpore 版本 | 2.0.0.20221118                        |
| 数据集         | REDS                                  |
| 训练参数       | batch_size=8, 300,000 steps           |
| 优化器         | Adam                                  |
| 速度           | 5.60 s/step                           |
| 总耗时         | N/A                                   |

为了减少内存消耗，可以减小`batch_size `和/或`num_frame`的值。然而，这可能会导致质量下降。

### [评估性能](#内容)

| 参数                | 1xGPU                                 | 1xAscend                              |
|-----------------------------|---------------------------------------|---------------------------------------|
| 模型版本           | 001_train_vrt_videosr_bi_reds_6frames | 001_train_vrt_videosr_bi_reds_6frames |
| 资源                   | 1x Nvidia A100                        | 1x Ascend 910                         |
| MindSpore 版本           | 1.9.0                                 | 2.0.0.20221118                        |
| 数据集                    | REDS                                  | REDS                                  |
| 一次处理数据数目          | 1                                     | 1                                     |
| 测试帧数         | 40                                    | 40                                    |
| PSNR 指标                 | 31.63                                 | 31.62                                 |
| GPU内存消耗 | 57.6 GB                               | N/A                                   |
| 速度                       | 12.82 s/call                          | 18.12 s/call             |

为了减少内存消耗，可以降低`num_frame_testing`的值。然而，这可能会导致质量下降。

# [ModelZoo 首页](#内容)

请查询官方网站[首页](https://gitee.com/mindspore/models)
