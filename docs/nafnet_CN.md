# 内容

- [NAFNet 简介](#nafnet-description)
- [数据集](#dataset)
- [环境要求](#environmental-requirements)
- [性能](#performance)
  - [训练性能](#training-performance)
  - [评估性能](#evaluation-performance)

# [NAFNet 简介](#contents)

虽然最近在图像复原领域取得了重大进展，但最先进（SOTA）方法的系统复杂性也在增加，这可能会阻碍方法的方便分析和比较。在本文中，我们提出了一个简单的基线，它超过了SOTA方法，并且在计算上是有效的。为了进一步简化基线，我们揭示了非线性激活函数，如Sigmoid、ReLU、GELU、Softmax等是不必要的：它们可以被乘法取代或删除。因此，我们从基线导出了一个非线性激活自由网络，即NAFNet。SOTA结果是在各种具有挑战性的基准上实现的，例如GoPro上的33.69 dB PSNR（用于图像去模糊），超过了以前的SOTA 0.38 dB，仅占其计算成本的8.4%;SIDD上的40.30 dB PSNR（用于图像去噪），超过了以前的SOTA 0.28 dB，计算成本不到一半。

[论文](https://arxiv.org/abs/2204.04676): Simple Baselines for Image Restoration

[原始 github 仓库](https://github.com/megvii-research/NAFNet)

# [数据集](#contents)

此任务使用 [GoPro](https://seungjunnah.github.io/Datasets/gopro.html) 数据集. 数据集包括T2103列车和1111测试图像，分辨率为1280 x 720。

下载 [链接](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing).

# [环境要求](#contents)

## Ascend 910

- 硬件 (Ascend)
  - 使用Ascend 910准备硬件环境 (cann_6.0.0, euler_2.8, py_3.7)
- 框架
  - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) 或 更新版本

# [性能](#contents)

## [训练性能](#contents)

| 参数           | NAFNet deblur (8xNPU)           |
| -------------- | ------------------------------- |
| 模型版本       | NAFNet                          |
| 资源           | 1x Ascend 910A                  |
| 上传日期       | 06 / 14 / 2023 (month/day/year) |
| MindSpore 版本 | 1.9.0                           |
| 数据集         | GoPro                           |
| 训练参数       | batch_size=16, 10000 epochs     |
| 优化器         | Adam                            |
| 速度           | 780 ms/step                     |
| 总时间         | 2d 12h 31m                      |
| 微调参数文件   | 274.9 MB (.ckpt file)           |

## [评估性能](#contents)

| 参数           | NAFNet deblur (1xNPU, CANN)     |
| -------------- | ------------------------------- |
| 模型版本       | NAFNet                          |
| 资源           | 1x Ascend 910A                  |
| 上传日期       | 06 / 14 / 2023 (month/day/year) |
| MindSpore 版本 | 1.9.0                           |
| 数据集         | GoPro                           |
| Batch_size     | 1                               |
| 推理速度/s     | 0.041 (1280x720)                |
| PSNR 指标      | 30.48                           |
| SSIM 指标      | 0.9041                          |
