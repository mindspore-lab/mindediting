# 内容

- [RVRT 简介](#rvrt-description)
- [模型架构](#model-architecture)
- [数据集](#dataset)
- [环境要求](#environmental-requirements)
- [性能](#performance)
  - [训练性能](#training-performance)
  - [评估性能](#evaluation-performance)

# [RVRT 简介](#contents)

视频恢复旨在从多个低质量帧中恢复多个高质量帧。现有的视频复原方法一般分为两个极端；在某些情况下，即并行恢复所有帧，或者以循环的方式逐帧恢复视频，会产生不同的优点和缺点。通常，前者具有时间信息融合的优点。然而，它的缺点是模型尺寸大，内存消耗大;后者相对较小的模型尺寸，因为它跨帧共享参数;然而，它缺乏远程依赖关系建模能力和并行性。在本文中，我们尝试结合这两种情况的优点，提出一种循环视频恢复变压器，即RVRT。RVRT处理本地邻居在全局循环框架内并行的框架可以在模型大小、有效性和效率之间实现良好的权衡。具体来说,RVRT将视频分成多个片段，并使用之前推断的片段特征来估计后续的片段特征。在每个剪辑中，不同的帧特征是隐式特征聚合联合更新。在不同的夹子中，导向的可变形注意力是为夹子到夹子的对齐而设计的，它可以预测多个从整个片段中推断出相关位置，并通过注意机制聚合其特征。在视频超分辨率，去模糊，和去噪表明，所提出的RVRT在平衡模型大小、测试内存和运行时间的基准数据集上达到了最先进的性能。

目前只支持视频超分辨率。

[论文](https://arxiv.org/pdf/2206.02146.pdf): Recurrent Video Restoration Transformer with Guided Deformable Attention.

[参考 github 仓库 (仅限评估)](https://github.com/JingyunLiang/RVRT)

[参考 github 仓库 (训练和评估)](https://github.com/cszn/KAIR)

# [模型架构](#contents)

该模型包括三个部分:浅层特征提取、循环特征细化和HQ帧重构。更具体地说，在浅特征提取，使用卷积层从LQ视频中提取特征。在此基础上，利用多个残频变压器块(Residual Swin Transformer block, rstb)提取浅层特征。然后，使用循环特征细化模块进行时间对应建模，并引导可变形注意进行视频对齐。最后，加入多rstb生成最终特征，并通过像素洗牌层重构HQ视频。

RVRT Light是由RVRT通过简单的注意力取代引导的可变形注意力并移除SpyNet而制成的。

# [数据集](#contents)

## 使用数据集

此任务使用 [Vimeo90K](http://toflow.csail.mit.edu/) 数据集.
下载 [链接](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).

## 数据集预处理

官方数据集由高分辨率的地面真值样本组成。要获得低分辨率的样本，应该使用 `src/dataset/preprocess.py` 生成注释文件的脚本，并通过双三次插值样本从地面真实值中缩放4倍。

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

有关详细信息，请运行:

```commandline
python mindediting/dataset/src/vimeo_preprocess.py --help
```

预处理脚本使用了双三次插值的实现，这对结果有很大的影响。

## 数据集组织方式

建议使用数据集设置目录，如下所示。然而，文件的另一种结构是可能的，但需要在配置文件中更改。要设置默认文件结构，请执行以下操作：

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

# [环境要求](#contents)

## GPU

- 硬件 (GPU)
  - 使用GPU处理器准备硬件环境
- 框架
  - [MindSpore](https://www.mindspore.cn/install)
- 有关详细信息，请参阅以下资源:
  - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 其他python包:
  - 手动安装其他包，或者在模型目录下使用 `pip install -r requirements.txt` 命令。

## Ascend 910

- 硬件 (Ascend)
  - 使用Ascend 910准备硬件环境 (cann_6.0.0, euler_2.8, py_3.7)
- 框架
  - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) 或 更新版本

# [性能](#contents)

## [训练性能](#contents)

| 参数           | 8xAscend      |
| -------------- | ------------- |
| 模型版本       | RVRT          |
| 资源           | 8x Ascend 910 |
| 上传日期       | N/A           |
| MindSpore 版本 | 1.9.0         |
| 数据集         | Vimeo90k      |
| 训练参数       | batch_size=8  |
| 优化器         | Adam          |
| 速度           | 1.42 s/step   |

| 参数           | 8xAscend      |
| -------------- | ------------- |
| 模型版本       | RVRT Light    |
| 资源           | 8x Ascend 910 |
| 上传日期       | N/A           |
| MindSpore 版本 | 1.9.0         |
| 数据集         | Vimeo90k      |
| 训练参数       | batch_size=8  |
| 优化器         | Adam          |
| 速度           | 1.25 s/step   |

## [评估性能](#contents)

| 参数        | 1xGPU (FP32)     | 1xAscend (MixedPrecison) |
| ----------- | ---------------- | ------------------------ |
| 模型版本    | RVRT             | RVRT                     |
| 资源        | 1x Nvidia 3090TI | 1x Ascend 910            |
| 后端        | MindSpore 2.0.0a | CANN 6.0.RC1.alpha005    |
| 数据集      | Vimeo90k         | Vimeo90k                 |
| Batch_size  | 1                | 1                        |
| 测试帧数    | 14               | 14                       |
| PSNR 指标   | 38.12            | 38.12                    |
| GPU内存消耗 | 11.7 GB          | N/A                      |
| 速度        | 2.32 s/call      | 0.74 s/call              |

| 参数        | 1xGPU (FP32)     | 1xAscend (MixedPrecison) |
| ----------- | ---------------- | ------------------------ |
| 模型版本    | RVRT Light       | RVRT light               |
| 资源        | 1x Nvidia 3090TI | 1x Ascend 910            |
| 后端        | MindSpore 2.0.0a | CANN 6.0.RC1.alpha005    |
| 数据集      | Vimeo90k         | Vimeo90k                 |
| Batch_size  | 1                | 1                        |
| 测试帧数    | 14               | 14                       |
| PSNR 指标   | 37.91            | 37.91                    |
| GPU内存消耗 | 6.3 GB           | N/A                      |
| 速度        | 1.9 s/call       | 0.4 s/call               |
