# 内容

- [内容](#内容)
- [BasicVSR 简介](#basicvsr-简介)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [模型简介](#模型简介)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [ModelZoo 首页](#modelzoo-首页)

# [BasicVSR 简介](#内容)

视频超分辨率(VSR)方法往往比图像方法有更多的组件，因为它们需要利用额外的时间维度。复杂的设计并不少见。在这项研究中，我们希望解决这些问题，并在四个基本功能(即传播、对齐、聚合和上采样)的指导下重新考虑VSR的一些最重要的组件。通过重用一些现有组件，并进行了最小程度的重新设计，我们展示了一个简洁的流程BasicVSR，与许多最先进的算法相比，它在速度和恢复质量方面取得了惊人的改进。我们进行了系统的分析，以解释如何获得这样的收益，并讨论陷阱。我们进一步展示了BasicVSR的可扩展性，通过提供信息补充机制和促进信息聚合的耦合传播方案。BasicVSR及其扩展IconVSR可以作为未来VSR方法的强大基线。

[论文](https://arxiv.org/abs/2012.02181): BasicVSR: 在视频超分辨率及以上的基本组件的研究。

[参考github存储库](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/README.md)

# [模型架构](#内容)

BasicVSR有以下主要的设计选择: 对于传播，BasicVSR选择了双向传播，强调长期和全局传播。对于对齐，BasicVSR在特征级采用了简单的基于流的对齐方法。
对于聚合和上采样，选择流行的特征拼接和像素重组就足够了。

# [数据集](#内容)

## 使用数据集

工作使用 [Vimeo90K](http://toflow.csail.mit.edu/) 数据集。
下载 [link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).

## 数据集预处理

官方数据集由高分辨率的真实样本组成。为了获得低分辨率的样本，应该使用' src/dataset/pre - process.py '脚本，该脚本生成注释文件，并通过双三次插值样本从真实值缩小4倍。

准备数据集:

1. 下载并解压缩数据集
2. 运行预处理脚本:

```commandline
python mindediting/dataset/src/vimeo_preprocess.py \
  --train-annotation ${DATASET_ROOT_DIR}/vimeo_septuplet/sep_trainlist.txt \
  --test-annotation ${DATASET_ROOT_DIR}vimeo_septuplet/sep_testlist.txt \
  --images-root ${DATASET_ROOT_DIR}vimeo_septuplet/sequences \
  --output-dir ${DATASET_ROOT_DIR}vimeo_septuplet/BIx4 \
  --generate-lq
```

欲知详情，请浏览:

```commandline
python mindediting/dataset/src/vimeo_preprocess.py --help
```

预处理脚本使用双三次插值的实现，比如在MatLab中，这对结果有很大的影响。

## 数据集组织方式

建议使用如下所示的数据集设置目录。然而，另一种文件结构是可能的，但需要修改配置文件。要设置默认的文件结构，请执行以下操作:

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

# [环境要求](#内容)

## GPU

- 硬件 (GPU)
    - 准备GPU处理器硬件环境
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 详细信息请参见以下资源：
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 其他python包：
    - 手动或在模型目录下 ` pip Install -r requirements.txt `命令安装其他包。

## Ascend 910

- 硬件 (Ascend)
    - 准备Ascend 910硬件环境 (cann_5.1.2, euler_2.8.3, py_3.7)
- 框架
    - [MindSpore Ascend 1.9.0](https://www.mindspore.cn/install) or 更新


# [模型介绍](#内容)

## [性能](#内容)

### [训练性能](#内容)

| 参数           | BasicVSR (2xGPU)                | BasicVSR (1xGPU)                |
| -------------- | ------------------------------- | ------------------------------- |
| 模型版本       | BasicVSR                        | BasicVSR                        |
| 资源           | 2x Nvidia A100                  | 1x Nvidia A100                  |
| 上传日期       | 12 / 14 / 2022 (month/day/year) | 12 / 14 / 2022 (month/day/year) |
| MindSpore 版本 | 1.9.0                           | 1.9.0                           |
| 数据集         | Vimeo90K                        | Vimeo90K                        |
| 训练参数       | batch_size=8, 37 epochs         | batch_size=8, 37 epochs         |
| 优化器         | Adam                            | Adam                            |
| 速度           | 2220 ms/step                    | 2807 ms/step                    |
| 总耗时         | 8d 8h 23m                       | 10d 6h 40m                      |
| 微调参数文件   | 99 MB (.ckpt file)              | 99 MB (.ckpt file)              |

| 参数           | BasicVSR (8xNPU)                |
| -------------- | ------------------------------- |
| 模型版本       | BasicVSR                        |
| 资源           | 8x Ascend 910                   |
| 上传日期       | 12 / 14 / 2022 (month/day/year) |
| MindSpore 版本 | 2.0.0.20221118                  |
| 数据集         | Vimeo90K                        |
| 训练参数       | batch_size=8, 37 epochs         |
| 优化器         | Adam                            |
| 速度           | 920 ms/step                     |
| 总耗时         | 3d 4h 23m                       |
| 微调参数文件   | 99 MB (.ckpt file)              |

### [评估性能](#内容)

| 参数           | BasicVSR (1xGPU)                | BasicVSR (2xGPU)                | BasicVSR (8xNPU)                |
| -------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| 模型版本       | BasicVSR                        | BasicVSR                        | BasicVSR                        |
| 资源           | 1x Nvidia A100                  | 2x Nvidia A100                  | 8x Ascend 910                   |
| 上传日期       | 12 / 14 / 2022 (month/day/year) | 12 / 14 / 2022 (month/day/year) | 12 / 14 / 2022 (month/day/year) |
| MindSpore 版本 | 1.9.0                           | 1.9.0                           | 2.0.0.20221118                  |
| 数据集         | Vimeo90K                        | Vimeo90K                        | Vimeo90K                        |
| 一次处理样本数 | 1                               | 1                               | 1                               |
| PSNR 指标      | 37.2                            | 37.21                           | 37.23                           |


# [ModelZoo 首页](#内容)

请访问官方网站 [首页](https://gitee.com/mindspore/models).
