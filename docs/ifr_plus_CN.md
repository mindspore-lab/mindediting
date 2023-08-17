# 内容

- [IFR+ 简介](#ifrplus-description)
- [数据集](#dataset)
- [环境要求](#environmental-requirements)
- [性能](#performance)
  - [训练性能](#training-performance)
  - [评估性能](#evaluation-performance)

# [IFR+ 简介](#contents)

## IFR+ 简介

[IFRNet](https://arxiv.org/abs/2205.14620) 模型的改进版本：结合了两种有前途的方法，如基于光流的向后翘曲和基于核的图像精炼。提出的网络架构允许在Vimeo-triplet数据集上击败SOTA。

## IFRNet 简介 (CVPR2022)

当前流行的视频帧插值算法从连续输入生成中间帧，通常依赖于具有重参数或大延迟的复杂模型架构，阻碍了它们在各种实时应用中的应用。在这项工作中，我们设计了一个高效的基于编码器-解码器的网络，称为IFRNet，用于快速中间帧合成。它首先从给定的输入中提取金字塔特征，然后结合强大的中间特征对双边中间流场进行细化，直到生成所需的输出。逐步细化的中间特征不仅可以方便中间流量估计，还可以补偿上下文细节，使IFRNet不需要额外的综合或细化模块。为了充分发挥它的潜力，我们进一步提出了一种新的面向任务的光流蒸馏损失，专注于学习对框架合成有用的教师知识。同时，对逐渐细化的中间特征加入新的几何一致性正则化项，以保持更好的结构布局。在各种基准测试上的实验证明了所提方法的优异性能和快速推理速度。

[论文](https://arxiv.org/abs/2205.14620): IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation

[原始 github 仓库](https://github.com/ltkong218/ifrnet)

# [数据集](#contents)

本文使用[Vimeo-triplet](http://toflow.csail.mit.edu/)数据集。该数据集由73,171个固定分辨率为448 x 256的3帧序列组成，提取自[Vimeo-90K](http://toflow.csail.mit.edu/)上的15K选定视频片段。该数据集是为时间帧插值而设计的。

下载 [链接](http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip).

# [环境要求](#contents)

## GPU

- 硬件 (GPU)
  - 准备带有GPU处理器的硬件环境
- 框架
  - [MindSpore](https://www.mindspore.cn/install)
- 详细信息请参见以下资源：
  - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 其他python包：
  - 手动安装其他包，或者在模型目录下使用 `pip install -r requirements.txt` 命令。

## Ascend 910

- 硬件 (Ascend)
  - 使用Ascend 910准备硬件环境 (cann_6.0.0, euler_2.8, py_3.7)
- 框架
  - [MindSpore Ascend 1.9.0](https://www.mindspore.cn/install) 或 更新版本

## [性能](#contents)

### [训练性能](#contents)

| 参数           | IFR+ (4xGPU)                    |
| -------------- | ------------------------------- |
| 模型版本       | IFR+                            |
| 资源           | 4x Nvidia V100                  |
| 上传日期       | 05 / 26 / 2023 (month/day/year) |
| MindSpore 版本 | 2.0.0.RC1                       |
| 数据集         | Vimeo-triplet                   |
| 训练参数       | batch_size=6, 100 epochs        |
| 优化器         | AdamW                           |
| 速度           | 1.89 s/step                     |
| 总时间         | 4d 17h 11m                      |
| 微调参数文件   | 413.0 MB (.ckpt file)           |

### [评估性能](#contents)

| 参数           | IFR+ (1xNPU, CANN)              |
| -------------- | ------------------------------- |
| 模型版本       | IFR+                            |
| 资源           | 1x Ascend 910A                  |
| 上传日期       | 05 / 26 / 2023 (month/day/year) |
| MindSpore 版本 | 2.0.0.20221118                  |
| 数据集         | Vimeo-triplet                   |
| Batch_size     | 1                               |
| 推理速度/s     | 0.15 (448x256)                  |
| 推理速度/s     | 2.00 (1920x1024)                |
| PSNR 指标      | 36.54                           |
| SSIM 指标      | 0.9710                          |
