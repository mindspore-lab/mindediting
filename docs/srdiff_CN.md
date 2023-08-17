# 内容

- [SRDiff 简介](#srdiff-description)
- [模型架构](#model-architecture)
- [数据集](#dataset)
- [环境要求](#environmental-requirements)
- [性能](#performance)
  - [训练性能](#training-performance)
  - [评估性能](#evaluation-performance)

# [SRDiff 简介](#contents)

单幅图像超分辨率(SISR)旨在重建高分辨率(HR)图像给定的低分辨率(LR)图像，这是一个不适定问题，因为一张LR图像对应多个HR图像。近年来，基于学习的SISR方法有了很大的发展性能优于传统算法，但存在过平滑、模态崩溃或大模型足迹问题分别针对面向psnr、gan驱动和基于流的方法。为了解决这些问题，我们提出了一种新的单幅图像超分辨扩散方法概率模型(SRDiff)，这是第一个基于扩散的SISR模型。SRDiff是用一种变分界的优化方法对数据的似然性和可提供通过逐步将高斯噪声转化为一种可实现多样化和现实的SR预测通过马尔可夫链对LR输入进行条件反射的超分辨率(SR)图像。此外,我们在整个框架中引入残差预测以加快收敛速度。我们在面部和一般基准(CelebA和DIV2K数据集)上进行了广泛的实验：

1. SRDiff可以在丰富的细节中生成不同的SR结果仅给定一个LR输入时的性能;

2) SRDiff易于训练，占地面积小;
3) SRDiff可以实现包括潜在空间插值在内的灵活图像处理以及内容融合。

[论文](https://arxiv.org/abs/2104.14951): SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models.

[参考 github 仓库](https://github.com/LeiaLi/SRDiff)

# [模型架构](#contents)

SRDiff由编码器RRDB和噪声预测网UNet组成，扩散步数为100。该模型以低分辨率图像为输入，通过双三次插值对图像进行放大。

# [数据集](#contents)

## 使用数据集

该任务使用 [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (数据来自 NTIRE 2017)和 [Flickr2K](https://www.kaggle.com/datasets/hliang001/flickr2k) 数据集.
`DIV2K` 数据集包括训练子集(800张图像)和验证子集(100张图像),
`Flick2K` 由2650张用于训练的图像组成。这两个数据集一起命名为 `DF2K`.

## 数据集预处理

官方数据集包括高分辨率地面真值样本和低分辨率地面真值样本。该模型对图像中的小块进行训练。要预处理补丁，请使用 `src/dataset/div2k_preprocess.py` 脚本，从高和低分辨率的样本。

准备数据集:

1. 下载并解压缩数据集
2. 运行预处理脚本:

```commandline
python div2k_preprocess.py \
  --root ${PATH_TO_ROOT_DIRECTORY_OF_DATASET} \
  --dataset div2k \
  --output ${OUTPUT_PATH}/div2k
```

```commandline
python div2k_preprocess.py \
  --root ${PATH_TO_ROOT_DIRECTORY_OF_DATASET} \
  --dataset flickr2k \
  --output ${OUTPUT_PATH}/flickr2k
```

有关详细信息，请运行:

```commandline
python div2k_preprocess.py --help
```

## 数据集组织方式

建议使用数据集设置目录，如下所示。但是，文件的另一种结构是可能的，但需要更改配置文件。要设置默认文件结构，请执行以下操作:

```text
.
└─ data
  ├─ div2k
  │  ├─ train
  │  │  ├─ X1
  │  │  │  ├─ image_0000.png
  │  │  │  ├─ image_0001.png
  │  │  │  ├─ ...
  │  │  └─ X4
  │  │     ├─ image_0000.png
  │  │     ├─ image_0001.png
  │  │     ├─ ...
  │  └─ val
  │     ├─ X1
  │     │  ├─ image_0000.png
  │     │  ├─ image_0001.png
  │     │  ├─ ...
  │     └─ X4
  │        ├─ image_0000.png
  │        ├─ image_0001.png
  │        ├─ ...
  │
  └─ flickr2k
     └─ train
        ├─ X1
        │  ├─ image_0000.png
        │  ├─ image_0001.png
        │  ├─ ...
        └─ X4
           ├─ image_0000.png
           ├─ image_0001.png
           ├─ ...
```

# [环境要求](#contents)

## GPU

- 硬件 (GPU)
  - 准备带有GPU处理器的硬件环境
- 框架
  - [MindSpore](https://www.mindspore.cn/install)
- 详细信息请参见以下参考资料:
  - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 其他python包:
  - 手动安装其他包，或者在模型目录下使用 `pip install -r requirements.txt` 命令。

## Ascend 910

- 硬件 (Ascend)
  - 使用Ascend 910准备硬件环境 (cann_5.1.2, euler_2.8.3, py_3.7)
- 框架
  - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) 或 更新版本

# [性能](#contents)

## [训练性能](#contents)

| 参数           | SRDiff (1xNPU)                  |
| -------------- | ------------------------------- |
| 模型版本       | SRDiff                          |
| 资源           | 1x Ascend 910                   |
| 上传日期       | 05 / 24 / 2023 (month/day/year) |
| MindSpore 版本 | 2.0.0.20221118                  |
| 数据集         | DIV2K                           |
| 训练参数       | batch_size=64, 60 epochs        |
| 优化器         | Adam                            |
| 速度           | 720 ms/step                     |
| 总时间         | 22h                             |
| 微调参数文件   | 575 MB (.ckpt file)             |

## [评估性能](#contents)

| 参数           | SRDiff (1xNPU, CANN)            |
| -------------- | ------------------------------- |
| 模型版本       | SRDiff                          |
| 资源           | 1x Ascend 910                   |
| 上传日期       | 05 / 24 / 2023 (month/day/year) |
| MindSpore 版本 | 2.0.0.20221118                  |
| 数据集         | DIV2K                           |
| Batch_size     | 1                               |
| 推理速度/s     | 0.96 (per 1 image, 320x270)     |
| PSNR 指标      | 28.78                           |
| LPIPS 指标     | 0.06                            |
