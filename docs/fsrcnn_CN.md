# [模型](#内容)

## FSRCNN 简介

作为一种成功应用于图像超分辨率(SR)的深度模型，超分辨率卷积神经网络(super-resolution Convolutional Neural Network, SRCNN)[1,2]在速度和复原质量上都表现出了优于以往手工制作模型的性能。然而，较高的计算成本仍然阻碍了它的实际应用，要求实时性能(24帧/秒)。在本文中，我们以加速当前SRCNN为目标，提出了一种紧凑的沙漏形CNN结构，以实现更快更好的SRCNN。我们主要从三个方面对SRCNN结构进行了重新设计。首先，我们在网络的末端引入反褶积层，然后直接从原始低分辨率图像(不需要插值)学习到高分辨率图像的映射。其次，我们通过在映射之前缩小输入特征维度，然后再展开来重新制定映射层。第三，我们采用更小的过滤器尺寸，但更多的映射层。该模型的复原速度提高了40倍以上，复原质量甚至更高。此外，我们还介绍了一些参数设置，这些参数设置可以在通用CPU上实现实时性能，同时仍然保持良好的性能。针对不同提升因子的快速训练和测试，提出了相应的迁移策略。

[论文](https://arxiv.org/pdf/1608.00367.pdf): Accelerating the Super-Resolution Convolutional Neural Network

# [数据集](#内容)

91-image，Set5数据集转换为HDF5可以从下面的链接下载。

| 数据集   | 倍数 | 类型  | 链接                                                         |
| -------- | ---- | ----- | ------------------------------------------------------------ |
| 91-image | 2    | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
| 91-image | 3    | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
| 91-image | 4    | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
| Set5     | 2    | Eval  | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) |
| Set5     | 3    | Eval  | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0) |
| Set5     | 4    | Eval  | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0) |

# 模型实现

[参考github仓库 (PyTorch)](https://github.com/yjn870/FSRCNN-pytorch)

# 训练性能

为了比较结果，我们在这里和报告中的所有地方使用scale = 4。

| 模型                | 设备类型 | 规格        | PSNR (dB) | 训练时间(secs per epoch) |
| ------------------- | -------- | ----------- | --------- | ------------------------ |
| FSRCNN-Torch-paper  | GPU      | ?           | 30.55     | ?                        |
| FSRCNN-Torch-github | GPU      | ?           | 30.50     | ?                        |
| FSRCNN-Torch        | GPU      | P100        | 30.44     | 210                      |
| FSRCNN-MS           | GPU      | P100        | 30.15     | 320                      |
| FSRCNN-MS-Ascend    | Ascend   | Ascend-910A | 30.18     | 360                      |

原始(论文和github)和我们的实现之间的PSNR差异很小。所以结果被复制了。

# 推理性能

## 推理数据集信息

|                  |      |
| ---------------- | ---- |
| 数据集名         | Set5 |
| 一次处理数据数目 | 8    |
| 级别             | x4   |
| 实验数 (its)     | 1000 |
| 预热 (its)       | 100  |

## 计时

| 模型             | 设备类型 | 规格        | 推理时间 (ms, median, per it) |
| ---------------- | -------- | ----------- | ----------------------------- |
| FSRCNN-Torch     | GPU      | P100        | 6                             |
| FSRCNN-MS        | GPU      | P100        | 13                            |
| FSRCNN-MS-Ascend | Ascend   | Ascend-910A | 20                            |
