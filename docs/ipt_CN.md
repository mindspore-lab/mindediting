<TOC>

# 预训练图像处理Transformer (IPT)

该仓库是CVPR 2021论文 "Pre-Trained Image Processing Transformer" 的正式实现。

我们研究了底层的计算机视觉任务(如去噪、超分辨率和去雨)，并开发了一种新的预训练模型，即图像处理转换器(IPT)。为了最大限度地挖掘转换器的能力，我们提出利用著名的ImageNet基准来生成大量损坏的图像对。IPT模型是在这些多头多尾图像上训练的。此外，还引入了对比学习，以适应不同的图像处理任务。因此，经过微调的预训练模型可以有效地用于预期的任务。由于只有一个预训练的模型，IPT在各种低级基准测试中优于当前最先进的方法。

如果你觉得我们的工作对你的研究或出版有用，请引用我们的工作:
[1] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen
Gao. **"Pre-trained image processing transformer"**. <i>**CVPR 2021**.</i> [[arXiv](https://arxiv.org/abs/2012.00364)]

    @inproceedings{chen2020pre,
      title={Pre-trained image processing transformer},
      author={Chen, Hanting and Wang, Yunhe and Guo, Tianyu and Xu, Chang and Deng, Yiping and Liu, Zhenhua and Ma, Siwei and Xu, Chunjing and Xu, Chao and Gao, Wen},
      booktitle={CVPR},
      year={2021}
     }

## 模型架构

### IPT整体网络架构如下图所示:

![architecture](image/ipt.png)

## 数据集

基准测试数据集可在以下下载:

超分辨率:

Set5,
[Set14](https://sites.google.com/site/romanzeyde/research-interests),
[B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),
Urban100.

去噪:

[CBSD68](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).

去雨：

[Rain100L](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)

结果图像转换为YCbCr色彩空间。PSNR仅在Y通道上计算。

## 需求

### 硬件 (Ascend)

> 准备 Ascend 硬件环境。

### 框架

> [MindSpore>=1.9](https://www.mindspore.cn/install/en)

### 欲了解更多信息，请查看下面的资源

[MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
[MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)



## 性能

### 推理性能

所有任务的结果如下所示。

超分辨率的结果：

| 倍数 | Set5  | Set14 | B100  | Urban100 |
| ---- | ----- | ----- | ----- | -------- |
| ×2   | 38.33 | 34.49 | 32.46 | 33.74    |
| ×3   | 34.86 | 30.85 | 29.38 | 29.50    |
| ×4   | 32.71 | 29.03 | 27.84 | 27.24    |

去噪结果:

| 噪声水平 | CBSD68 | Urban100 |
| -------- | ------ | -------- |
| 30       | 32.35  | 33.99    |
| 50       | 29.93  | 31.49    |

去雨:

| 任务   | Rain100L |
| ------ | -------- |
| Derain | 42.08    |

超分辨率(x4)任务推理速度:

| 平台                                                         | 设备类型 | 规格        | set5数据集上图像的推理时间(seconds)  |
| ------------------------------------------------------------ | -------- | ----------- | ------------------------------------ |
| IPT-Torch                                                    | GPU      | V100        | 1.33, 0.69, 0.59, 0.69, 1.09         |
| IPT-MS (mindspore 1.7)                                       | GPU      | V100        | 39.29, 13.45, 0.69, 0.71, 14.01      |
| IPT-MS-Ascend (mindspore 1.9, Graph Mode)                    | Ascend   | Ascend-910A | 419.53, 254.83, 9.49, 22.68, 275.21  |
| IPT-MS-Ascend (mindspore 1.9, Pynative Mode)                 | Ascend   | Ascend-910A | 635.53, 195.69, 14.94, 15.07, 271.17 |
| IPT-MS-Ascend (mindspore 1.9, Graph Mode, warmup 50 exps)    | Ascend   | Ascend-910A | 2.74, 2.68, 2.47, 2.38, 2.56         |
| IPT-MS-Ascend (mindspore 1.9, Pynative Mode, warmup 50 exps) | Ascend   | Ascend-910A | 4.43, 4.10, 4.44, 4.03, 4.11         |

训练速度:
*在mindspore中，动态形状需要额外的成本。特别是在两幅大小差异较大的图像上。

训练速度:

| 平台          | 设备类型 | 规格        | 每个步骤的训练时间(seconds) |
| ------------- | -------- | ----------- | --------------------------- |
| IPT-MS-Ascend | Ascend   | Ascend-910A | 0.159 (batch size: 32)      |
