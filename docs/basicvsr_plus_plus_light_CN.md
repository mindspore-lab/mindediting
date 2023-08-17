# 内容

- [BasicVSR++Light ](#basicvsr-description)[简介](#contents)
- [模型架构](#model-architecture)
- [多任务](#tasks)
- [数据集](#datasets)
- [环境要求](#environmental-requirements)
- [性能](#performance)
  - [训练性能](#training-performance)
  - [评估性能](#evaluation-performance)

# [BasicVSR++Light 简介](#contents)

循环结构是视频超分辨率任务的流行框架选择。最先进的方法BasicVSR采用具有特征对齐的双向传播有效地利用来自整个输入视频的信息。在本研究中，我们重新设计基本VSR通过提出二阶网格传播和流引导变形对准。我们表明，通过增强的传播和对齐来增强循环框架的能力，人们可以更有效地利用未对齐视频帧中的时空信息。新组件在类似的计算约束下提高了性能。特别是，我们的模型BasicVSR++在PSNR上比BasicVSR高出0.82 dB，具有相似的PSNR的参数。除了视频超分辨率外，BasicVSR++也很好地推广到其他视频恢复任务，如压缩视频增强。.

[论文](https://arxiv.org/abs/2104.13371): BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

[参考 github 仓库](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/basicvsr_plusplus)

# [模型架构](#contents)

这是BasicVSR++模型的轻量级实现，并进行了以下优化对于MindSpore:

- 不包括SPyNet
- 可变形卷积被5个剩余块取代。

# [多任务](#contents)

此模型用于以下任务:

- 视频超分辨率 (VSR)
- 视频去噪 (VDN)
- 视频去模糊 (VDB)
- 视频增强 (VE)

# [数据集](#contents)

## 视频超分

### 使用数据集

超分任务使用 [Vimeo90K](http://toflow.csail.mit.edu/) 数据集.
下载 [链接](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).

### 数据集预处理

官方数据集由高分辨率的地面真值样本组成。要获得低分辨率的样本，应该使用 `src/dataset/preprocess.py` 生成注释文件的脚本，并通过bicbic缩小了4倍从基础真值的插值样本。

准备数据集:

1. 下载并解压缩数据集
2. 执行预处理脚本:

```commandline
python vimeo_preprocess.py \
  --train-annotation ${DATASET_ROOT_DIR}/vimeo_septuplet/sep_trainlist.txt \
  --test-annotation ${DATASET_ROOT_DIR}vimeo_septuplet/sep_testlist.txt \
  --images-root ${DATASET_ROOT_DIR}vimeo_septuplet/sequences \
  --output-dir ${DATASET_ROOT_DIR}vimeo_septuplet/BIx4 \
  --generate-lq
```

有关更多详细信息，请运行:

```commandline
python vimeo_preprocess.py --help
```

预处理脚本使用双三次插值的实现，就像MatLab中的那样，它具有对结果的影响很大。

### 数据集组织方式

建议使用数据集设置目录，如下所示。但是，文件的另一种结构是可能的，但需要在配置文件中更改。要设置默认文件结构，请执行以下操作:

```bash
cd ${CODE_ROOT_DIR}
mkdir data && cd data
ln -s ${DATASET_ROOT_DIR} vimeo90k
```

```text
.
└─ data
  └─ vimeo90k
    ├─ sequences
    │  ├─ 00001
    │  │  ├─ 0001
    │  │  │  ├─ im1.png
    │  │  │   ...
    │  │  │  └─ im7.png
    │  │  ├─ ...
    │  │  ...
    │  ├─ 00002
    │  │  ├─ 0001
    │  │  │  ├─ im1.png
    │  │  │   ...
    │  │  │  └─ im7.png
    │  │  ├─ ...
    │  │  ...
    │  ...
    └─ BIx4
       ├─ 00001
       │  ├─ 0001
       │  │  ├─ im1.png
       │  │   ...
       │  │  └─ im7.png
       │  ├─ ...
       │  ...
       ├─ 00002
       │  ├─ 0001
       │  │  ├─ im1.png
       │  │   ...
       │  │  └─ im7.png
       │  ├─ ...
       │  ...
       ...
```

## 视频去噪

### 使用数据集

去噪任务使用 [DAVIS 2017](https://davischallenge.org/) 数据集480p分辨率.
下载 [链接](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip)

### 数据集组织方式

```text
.
└─ data
  └─ DAVIS
    ├─ Annotations_unsupervised
    |  └─ 480p
    │     ├─ bear
    │     │  ├─ 00000.png
    |     │   ...
    |     ...
    ├─ ImageSets
    |  └─ 2017
    │     ├─ train.txt
    |     └─ val.txt
    ├─ JPEGImages
    |  └─ 480p
    │     ├─ bear
    │     │  ├─ 00000.png
    |     │   ...
    |     ...
    ├─ README.md
    └─ SOURCES.md
```

## 视频去模糊

### 使用数据集

去模糊使用 [REDS](https://seungjunnah.github.io/Datasets/reds.html) 数据集.
下载 [链接](https://seungjunnah.github.io/Datasets/reds.html)

### 数据集组织方式

```text
.
└─ data
  └─ REDS
    ├─ train
    │  ├─ train_blur
    │  │  ├─ 000
    │  │  │  ├─ 00000000.png
    |  │  │   ...
    │  │  │  └─ 00000099.png
    │  │  └─ ...
    |  └─ train_sharp
    │     ├─ 000
    │     │  ├─ 00000000.png
    |     │   ...
    │     │  └─ 00000099.png
    |     └─ ...
    ├─ val
    │  ├─ val_blur
    │  │  ├─ 000
    │  │  │  ├─ 00000000.png
    |  │  │   ...
    │  │  │  └─ 00000099.png
    │  │  └─ ...
    |  └─ val_sharp
    │     ├─ 000
    │     │  ├─ 00000000.png
    |     │   ...
    │     │  └─ 00000099.png
    │     └─ ...
    ├─ REDS_train.txt
    └─ REDS_test.txt
```

## 视频增强

### 使用数据集

视频增强任务使用 [LDV 2.0](https://github.com/RenYang-home/LDV_dataset) 数据集.
下载 [链接](https://github.com/RenYang-home/LDV_dataset#ldv-20-ldv-10--95-videos--335-videos)

### 数据集预处理

准备数据集:

1. 下载并解压缩数据集
2. 运行预处理脚本 `run.sh`

### 数据集组织方式

```text
.
└─ data
  └─ LDVv2
    ├─ test_gt
    |  ├─ 001
    │  |  ├─ f001.png
    │  |   ...
    |  └─ ...
    ├─ test_lq
    |  ├─ 001
    │  |  ├─ f001.png
    │  |   ...
    |  └─ ...
    ├─ train_gt
    |  ├─ 001
    │  |  ├─ f001.png
    │  |   ...
    |  └─ ...
    ├─ train_lq
    |  ├─ 001
    │  |  ├─ f001.png
    │  |   ...
    |  └─ ...
    ├─ valid_gt
    |  ├─ 001
    │  |  ├─ f001.png
    │  |   ...
    |  └─ ...
    ├─ valid_lq
    |  ├─ 001
    │  |  ├─ f001.png
    │  |   ...
    |  └─ ...
    ├─ data-test.xlsx
    ├─ data-train.xlsx
    ├─ data-valid.xlsx
    └─ run.sh
```

# [环境要求](#contents)

## GPU

- 硬件 (GPU)
  - 准备带有GPU处理器的硬件环境
- 框架
  - [MindSpore](https://www.mindspore.cn/install)
- 具体请参见以下资源:
  - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 其他python包:
  - 手动安装其他包，或者在模型目录下使用 `pip install -r requirements.txt` 命令。

## Ascend 910

- 硬件 (Ascend)
  - 准备 Ascend 910 硬件环境(cann_5.1.2, euler_2.8.3, py_3.7)
- 框架
  - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) or later

# [性能](#contents)

## [训练性能](#contents)

| 参数           | BasicVSR++ VSR (1xNPU)          | BasicVSR++ VDB (1xNPU)          | BasicVSR++ VDN (8xNPU)          | BasicVSR++ VE (8xNPU)           |
| -------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| 模型版本       | BasicVSR++Light                 | BasicVSR++Light                 | BasicVSR++Light                 | BasicVSR++Light                 |
| 资源           | 1x Ascend 910                   | 1x Ascend 910                   | 8x Ascend 910                   | 8x Ascend 910                   |
| 上传日期       | 03 / 21 / 2023 (month/day/year) | 06 / 01 / 2023 (month/day/year) | 04 / 20 / 2023 (month/day/year) | 05 / 30 / 2023 (month/day/year) |
| MindSpore 版本 | 2.0.0.20221118                  | 2.0.0-alpha                     | 2.0.0-alpha                     | 2.0.0-alpha                     |
| 数据集         | Vimeo90K                        | REDS                            | DAVIS                           | LDV_V2                          |
| 训练参数       | batch_size=8, 37 epochs         | batch_size=2, 7000              | batch_size=1, 55000 epochs      | batch_size=4, 10000 epochs      |
| 优化器         | Adam                            | Adam                            | Adam                            | Adam                            |
| 速度           | 960 ms/step                     | 2240 ms/step                    | 205 ms/step                     | 420 ms/step                     |
| 总时间         | 4d 2h 7m                        | 5d 0h 11m                       | 1d 14h 30m                      | 1d 11h 2m                       |
| 微调参数文件   | 185.4 MB (.ckpt file)           | 269.63 MB (.ckpt file)          | 201.76 MB (.ckpt file)          | 40.80 MB (.ckpt file)           |

## [评估性能](#contents)

| 参数           | BasicVSR++ VSR (1xNPU, CANN)    | BasicVSR++ VDB (1xNPU, CANN)     | BasicVSR++ VDN (1xNPU, CANN)     | BasicVSR++ VE (1xNPU, CANN)      |
| -------------- | ------------------------------- | -------------------------------- | -------------------------------- | -------------------------------- |
| 模型版本       | BasicVSR++Light                 | BasicVSR++Light                  | BasicVSR++Light                  | BasicVSR++Light                  |
| 资源           | 1x Ascend 910                   | 1x Ascend 910                    | 1x Ascend 910                    | 1x Ascend 910                    |
| 上传日期       | 03 / 21 / 2023 (month/day/year) | 06 / 01 / 2023 (month/day/year)  | 04 / 20 / 2023 (month/day/year)  | 05 / 30 / 2023 (month/day/year)  |
| MindSpore 版本 | 2.0.0.20221118                  | 2.0.0-alpha                      | 2.0.0-alpha                      | 2.0.0-alpha                      |
| 数据集         | Vimeo90K                        | REDS                             | Set8                             | LDV_V2                           |
| Batch_size     | 1                               | 1                                | 1                                | 1                                |
| 推理速度/s     | 0.773 (per 25 frames, 480x270)  | 1.064 (per 25 frames, 1920x1080) | 0.951 (per 25 frames, 1920x1080) | 0.780 (per 26 frames, 1920x1080) |
| PSNR 指标      | 37.61                           | 32.48                            | 30.01  (noise sigma 50)          | 32.46                            |
| SSIM 指标      | 0.9487                          | 0.9336                           | 0.851  (noise sigma 50)          | 0.8835                           |
