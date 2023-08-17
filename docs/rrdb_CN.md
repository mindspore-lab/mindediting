# 内容

- [RRDB 简介](#rrdb-description)
- [模型架构](#model-architecture)
- [数据集](#dataset)
- [环境要求](#environmental-requirements)
- [性能](#performance)
  - [训练性能](#training-performance)
  - [评估性能](#evaluation-performance)

# [RRDB 简介](#contents)

残差密集块(RRDB)将多层残差网络与密集块相结合连接。根据观察，更多的层次和连接总是可以提高在性能方面，所提出的RRDB采用了比原始RRDB更深入、更复杂的结构残块。具体来说，建议的RRDB具有残差中的残差结构，其中残差学习在不同层次上使用，所以网络容量变得更高得益于密集的联系。通常，RRDB在生成模型中用作编码器但由于该模型作为一个SISR模型进行预训练，可以作为一个独立的模型使用。

[论文](https://arxiv.org/abs/1809.00219): ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks

[参考 github 仓库](https://github.com/LeiaLi/SRDiff)

# [模型架构](#contents)

RRDB非常简单，由残差中残差密集块序列组成。当前的实现使用这些块在中间层和与上采样阶段不同的是，上采样阶段采用双线性插值最近的一个。

# [数据集](#contents)

## 使用数据集

该任务使用 [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (数据来自 NTIRE 2017)和 [Flickr2K](https://www.kaggle.com/datasets/hliang001/flickr2k) 数据集.
`DIV2K` 数据集包括训练子集(800张图像)和验证子集(100张图像),
`Flick2K` **由2650张用于训练的图像组成。**这两个数据集一起命名为 `DF2K`.

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

建议使用数据集设置目录，如下所示。但是，文件的另一种结构是可能的，但需要在配置文件中更改。要设置默认文件结构，请执行以下操作:

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

| 参数           | RRDB (1xNPU)                    |
| -------------- | ------------------------------- |
| 模型版本       | RRDB                            |
| 资源           | 1x Ascend 910                   |
| 上传日期       | 05 / 24 / 2023 (month/day/year) |
| MindSpore 版本 | 2.0.0.20221118                  |
| 数据集         | DIV2K                           |
| 训练参数       | batch_size=128, 220 epochs      |
| 优化器         | Adam                            |
| 速度           | 500 ms/step                     |
| 总时间         | 36h                             |
| 微调参数文件   | 97 MB (.ckpt file)              |

## [评估性能](#contents)

| 参数           | RRDB (1xNPU, CANN)              |
| -------------- | ------------------------------- |
| 模型版本       | RRDB                            |
| 资源           | 1x Ascend 910                   |
| 上传日期       | 05 / 24 / 2023 (month/day/year) |
| MindSpore 版本 | 2.0.0.20221118                  |
| 数据集         | DIV2K                           |
| Batch_size     | 1                               |
| 推理速度/s     | 1.3 (per 1 image, 1920x1080)    |
| PSNR 指标      | 30.73                           |
| SSIM 指标      | 0.845                           |
