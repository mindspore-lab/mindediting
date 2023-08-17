# 内容

- [内容](#内容)
    - [CTSDG 简介](#ctsdg-简介)
    - [模型架构](#模型架构)
    - [数据集](#数据集)
    - [环境要求](#环境要求)
    - [快速启动](#快速启动)
    - [脚本介绍](#脚本介绍)
        - [脚本和示例代码](#脚本和示例代码)
        - [脚本参数](#脚本参数)
        - [训练过程](#训练过程)
        - [评估过程](#评估过程)
        - [导出 MINDIR](#导出mindir)
    - [模型介绍](#模型介绍)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
    - [随机情况描述](#随机情况描述)
    - [ModelZoo 首页](#modelzoo-首页)

## [CTSDG 简介](#内容)

近年来，通过引入结构先验，深度生成方法在图像修复方面取得了相当大的进展。然而，由于在结构重建过程中缺乏与图像纹理的适当交互，现有的解决方案在处理腐蚀较大的情况时能力不足，结果普遍失真。这是一种新型的用于图像修补的双流网络，它以耦合的方式对结构约束的纹理合成和纹理引导的结构重建进行建模，使它们更好地相互利用以获得更合理的生成。此外，为了提高全局一致性，设计了双向门控特征融合(Bi-GFF)模块来交换和组合结构和纹理信息，开发了上下文特征聚合(CFA)模块来通过区域亲和性学习和多尺度特征聚合来优化生成的内容。

> [论文](https://arxiv.org/pdf/2108.09760.pdf):  Image Inpainting via Conditional Texture and Structure Dual Generation
> Xiefan Guo, Hongyu Yang, Di Huang, 2021.
> [补充材料](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Guo_Image_Inpainting_via_ICCV_2021_supplemental.pdf)

## [模型架构](#内容)

CTSDG 遵循生成对抗网络(GAN)框架。

*生成器*。图像修复分为两个子任务，即结构约束纹理合成(左，蓝色)和纹理引导结构重建(右，红色)，两个并行耦合流相互借用编码的深层特征。Bi-GFF模块和CFA模块堆叠在生成器的末尾，以进一步细化结果。

*判别器*。纹理分支估计生成的纹理，而结构分支指导结构重建。

![ctsdg.png](image/ctsdg.png)

## [数据集](#内容)

数据集
使用: [CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

- 你需要从**CELEBA**下载  (section *Downloads -> Align&Cropped Images*):
    - `img_align_celeba.zip`
    - `list_eval_partitions.txt`
- 你需要从**NVIDIA Irregular Mask Dataset** 下载:
    - `irregular_mask.zip`
    - `test_mask.zip`
- 目录结构如下所示:

  ```text
    .
    ├── img_align_celeba            # images folder
    ├── irregular_mask              # masks for training
    │   └── disocclusion_img_mask
    ├── mask                        # masks for testing
    │   └── testing_mask_dataset
    └── list_eval_partition.txt     # train/val/test splits
  ```

## [环境要求](#内容)

- 硬件（GPU/Ascend）
    - 准备GPU or Ascend 910 处理器硬件环境.
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 欲了解更多信息，请查看下面的资源：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- 下载数据集

## [快速启动](#内容)

### [预训练 VGG16](#内容)

您需要将torch VGG16模型转换为感知损失模型以训练CTSDG模型。

1. [下载预训练 VGG16](https://download.pytorch.org/models/vgg16-397923af.pth)
2. 将torch版参数文件转换为mindspore版:

```shell
python converter.py --torch_pretrained_vgg=/path/to/torch_pretrained_vgg
```

转换后的mindspore参数文件将保存在与torch模型相同的目录中，名称为 `vgg16_feat_extr_ms.ckpt`.

在准备数据集并转换VGG16之后，您可以开始训练和评估，如下所示:

### [运行](#内容)

#### 训练

```shell
# standalone train
bash scripts/run_standalone_train_*.sh [DEVICE_ID] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]

# distribute train
bash scripts/run_distribute_train_*.sh [DEVICE_NUM] [CFG_PATH] [SAVE_PATH] [VGG_PRETRAIN] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

样例:

```shell
# standalone train
# DEVICE_ID - device number for training
# CFG_PATH - path to config
# SAVE_PATH - path to save logs and checkpoints
# VGG_PRETRAIN - path to pretrained VGG16
# IMAGES_PATH - path to CELEBA dataset
# MASKS_PATH - path to masks for training
# ANNO_PATH - path to file with train/val/test splits
bash scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
# or
bash scripts/run_standalone_train_ascend.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt

# distribute train (8p)
# DEVICE_NUM - number of devices for training
# other parameters as for standalone train
bash scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
# or
bash scripts/run_distribute_train_ascend.sh 8 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

#### 评估

```shell
# evaluate
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
# or
bash scripts/run_eval_ascend.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

样例：

```shell
# evaluate
# DEVICE_ID - device number for evaluating
# CFG_PATH - path to config
# CKPT_PATH - path to ckpt for evaluation
# IMAGES_PATH - path to img_align_celeba dataset
# MASKS_PATH - path to masks for testing
# ANNO_PATH - path to file with train/val/test splits
bash scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt
# or
bash scripts/run_eval_ascend.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt
```

## [脚本介绍](#内容)

### [脚本和示例代码](#内容)

```text
.
└── CTSDG
    ├── model_utils
    │   ├── __init__.py                     # init file
    │   └── config.py                       # parse arguments
    ├── scripts
    │   ├── run_distribute_train_ascend.sh  # launch distributed training(8p) on Ascend
    │   ├── run_distribute_train_gpu.sh     # launch distributed training(8p) on GPU
    │   ├── run_eval_ascend.sh              # launch evaluating on Ascend
    │   ├── run_eval_gpu.sh                 # launch evaluating on GPU
    │   ├── run_export_gpu.sh               # launch export mindspore model to mindir
    │   ├── run_standalone_train_ascend.sh  # launch standalone traininng(1p) on Ascend
    │   └── run_standalone_train_gpu.sh     # launch standalone traininng(1p) on GPU
    ├── src
    │   ├── discriminator
    │   │   ├── __init__.py                 # init file
    │   │   ├── discriminator.py            # discriminator
    │   │   └── spectral_conv.py            # conv2d with spectral normalization
    │   ├── generator
    │   │   ├── __init__.py                 # init file
    │   │   ├── bigff.py                    # bidirectional gated feature fusion
    │   │   ├── cfa.py                      # contextual feature aggregation
    │   │   ├── generator.py                # generator
    │   │   ├── pconv.py                    # partial convolution
    │   │   ├── projection.py               # feature to texture and texture to structure parts
    │   │   └── vgg16.py                    # VGG16 feature extractor
    │   ├── __init__.py                     # init file
    │   ├── callbacks.py                    # callbacks
    │   ├── dataset.py                      # celeba dataset
    │   ├── initializer.py                  # weight initializer
    │   ├── losses.py                       # model`s losses
    │   ├── trainer.py                      # trainer for ctsdg model
    │   └── utils.py                        # utils
    ├── __init__.py                         # init file
    ├── converter.py                        # convert VGG16 torch checkpoint to mindspore
    ├── default_config.yaml                 # config file
    ├── eval.py                             # evaluate mindspore model
    ├── export.py                           # export mindspore model to mindir format
    ├── README.md                           # readme file
    ├── requirements.txt                    # requirements
    └── train.py                            # train mindspore model
```

### [脚本参数](#内容)

训练参数可以在 `default_config.yaml`中配置。

```text
"gen_lr_train": 0.0002,                     # learning rate for generator training stage
"gen_lr_finetune": 0.00005,                 # learning rate for generator finetune stage
"dis_lr_multiplier": 0.1,                   # discriminator`s lr is generator`s lr multiply by this parameter
"batch_size": 6,                            # batch size
"train_iter": 350000,                       # number of training iterations
"finetune_iter": 150000                     # number of finetune iterations
"image_load_size": [256, 256]               # input image size
```

更多参数请参考 `default_config.yaml`的内容。

### [训练过程](#内容)

#### [在GPU上运行](#内容)

##### 单独训练 (1p)

```shell
# DEVICE_ID - device number for training (0)
# CFG_PATH - path to config (./default_config.yaml)
# SAVE_PATH - path to save logs and checkpoints (/path/to/output)
# VGG_PRETRAIN - path to pretrained VGG16 (/path/to/vgg16_feat_extr.ckpt)
# IMAGES_PATH - path to CELEBA dataset (/path/to/img_align_celeba)
# MASKS_PATH - path to masks for training (/path/to/training_mask)
# ANNO_PATH - path to file with train/val/test splits (/path/to/list_eval_partitions.txt)
bash scripts/run_standalone_train_gpu.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

日志将保存到 `/path/to/output/log.txt`

结果:

```text
...
DATE TIME iter: 250, loss_g: 18.492000579833984, loss_d: 2.052000045776367, step time: 585.55 ms
DATE TIME iter: 375, loss_g: 21.15999984741211, loss_d: 1.8960000276565552, step time: 585.97 ms
DATE TIME iter: 500, loss_g: 18.93600082397461, loss_d: 1.8509999513626099, step time: 586.43 ms
DATE TIME iter: 625, loss_g: 23.01099967956543, loss_d: 1.7970000505447388, step time: 587.83 ms
DATE TIME iter: 750, loss_g: 25.809999465942383, loss_d: 1.8359999656677246, step time: 587.7 ms
DATE TIME iter: 875, loss_g: 17.70800018310547, loss_d: 1.7239999771118164, step time: 587.28 ms
DATE TIME iter: 1000, loss_g: 21.058000564575195, loss_d: 1.6299999952316284, step time: 589.29 ms
...
```

##### 分布式训练 (8p)

```shell
# DEVICE_NUM - number of devices for training (8)
# other parameters as for standalone train
bash scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

日志将保存到 `/path/to/output/log.txt`

结果:

```text
...
DATE TIME iter: 250, loss_g: 26.28499984741211, loss_d: 1.680999994277954, step time: 757.67 ms
DATE TIME iter: 375, loss_g: 21.548999786376953, loss_d: 1.468000054359436, step time: 758.02 ms
DATE TIME iter: 500, loss_g: 17.89299964904785, loss_d: 1.2829999923706055, step time: 758.57 ms
DATE TIME iter: 625, loss_g: 18.750999450683594, loss_d: 1.2589999437332153, step time: 759.95 ms
DATE TIME iter: 750, loss_g: 21.542999267578125, loss_d: 1.1829999685287476, step time: 759.45 ms
DATE TIME iter: 875, loss_g: 27.972000122070312, loss_d: 1.1629999876022339, step time: 759.62 ms
DATE TIME iter: 1000, loss_g: 18.03499984741211, loss_d: 1.159000039100647, step time: 759.51 ms
...
```

#### [在 Ascend上运行](#内容)

##### 单独训练 (1p)

```shell
# all parameters match the ones for standalone GPU training command
bash scripts/run_standalone_train_ascend.sh 0 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

日志将保存到 `/path/to/output/log.txt`

结果:

```text
...
DATE TIME iter: 250, loss_g: 24.437000274658203, loss_d: 3.115000009536743, step time: 236.13 ms
DATE TIME iter: 375, loss_g: 18.98200035095215, loss_d: 3.0869998931884766, step time: 232.96 ms
DATE TIME iter: 500, loss_g: 22.756000518798828, loss_d: 3.0169999599456787, step time: 230.17 ms
DATE TIME iter: 625, loss_g: 25.884000778198242, loss_d: 2.9749999046325684, step time: 232.1 ms
DATE TIME iter: 750, loss_g: 20.797000885009766, loss_d: 2.9600000381469727, step time: 233.61 ms
DATE TIME iter: 875, loss_g: 22.4060001373291, loss_d: 2.8949999809265137, step time: 231.47 ms
DATE TIME iter: 1000, loss_g: 24.784000396728516, loss_d: 2.877000093460083, step time: 235.57 ms
...
```

##### 分布式训练 (8p)

```shell
# all parameters match the ones for distributed GPU training command
bash scripts/run_distribute_train_gpu.sh 8 ./default_config.yaml /path/to/output /path/to/vgg16_feat_extr.ckpt /path/to/img_align_celeba /path/to/training_mask /path/to/list_eval_partitions.txt
```

日志将保存到 `/path/to/output/log.txt`

结果:

```text
...
DATE TIME iter: 250, loss_g: 24.243000030517578, loss_d: 2.558000087738037, step time: 236.6 ms
DATE TIME iter: 375, loss_g: 22.350000381469727, loss_d: 2.365000009536743, step time: 236.47 ms
DATE TIME iter: 500, loss_g: 22.666000366210938, loss_d: 2.1989998817443848, step time: 236.79 ms
DATE TIME iter: 625, loss_g: 23.493999481201172, loss_d: 2.0880000591278076, step time: 236.7 ms
DATE TIME iter: 750, loss_g: 26.045000076293945, loss_d: 2.052999973297119, step time: 236.48 ms
DATE TIME iter: 875, loss_g: 20.440000534057617, loss_d: 2.075000047683716, step time: 236.54 ms
DATE TIME iter: 1000, loss_g: 21.094999313354492, loss_d: 2.0480000972747803, step time: 236.34 ms
...
```

### [评估过程](#内容)

#### [在GPU上运行](#内容)

```shell
bash scripts/run_eval_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

样例：

```shell
# DEVICE_ID - device number for evaluating (0)
# CFG_PATH - path to config (./default_config.yaml)
# CKPT_PATH - path to ckpt for evaluation (/path/to/ckpt)
# IMAGES_PATH - path to img_align_celeba dataset (/path/to/img_align_celeba)
# MASKS_PATH - path to masks for testing (/path/to/testing/mask)
# ANNO_PATH - path to file with train/val/test splits (/path/to/list_eval_partitions.txt)
bash scripts/run_eval_gpu.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt
```

日志将保存到 `./logs/eval_log.txt`.

结果:

```text
PSNR:
0-20%: 38.04
20-40%: 29.39
40-60%: 24.21
SSIM:
0-20%: 0.979
20-40%: 0.922
40-60%: 0.83
```

#### [在Ascend上运行](#内容)

```shell
bash scripts/run_eval_ascend.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH] [IMAGES_PATH] [MASKS_PATH] [ANNO_PATH]
```

样例：

```shell
# DEVICE_ID - device number for evaluating (0)
# CFG_PATH - path to config (./default_config.yaml)
# CKPT_PATH - path to ckpt for evaluation (/path/to/ckpt)
# IMAGES_PATH - path to img_align_celeba dataset (/path/to/img_align_celeba)
# MASKS_PATH - path to masks for testing (/path/to/testing/mask)
# ANNO_PATH - path to file with train/val/test splits (/path/to/list_eval_partitions.txt)
bash scripts/run_eval_ascend.sh 0 ./default_config.yaml /path/to/ckpt /path/to/img_align_celeba /path/to/testing_mask /path/to/list_eval_partitions.txt
```

日志将保存到  `./logs/eval_log.txt`.

结果:

```text
PSNR:
0-20%: 37.591
20-40%: 29.079
40-60%: 23.971
SSIM:
0-20%: 0.977
20-40%: 0.917
40-60%: 0.822
```

### [导出MINDIR](#内容)

如果您想要在Ascend 310上推断网络，您应该将模型转换为MINDIR。

#### GPU

```shell
bash scripts/run_export_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH]
```

样例：

```shell
# DEVICE_ID - device number (0)
# CFG_PATH - path to config (./default_config.yaml)
# CKPT_PATH - path to ckpt for evaluation (/path/to/ckpt)
bash scripts/run_export_gpu.sh 0 ./default_config.yaml /path/to/ckpt
```

日志将保存到  `./logs/export_log.txt`, converted model will have the same name as ckpt except extension.

## [模型介绍](#内容)

### [训练性能](#内容)

| 参数           | GPU                                                          | Ascend                                                       |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 资源           | 1x Nvidia V100                                               | 1x Ascend 910                                                |
| 上传日期       | 19.12.2022                                                   | 19.12.2022                                                   |
| Mindspore 版本 | 1.9.0                                                        | 2.0.0                                                        |
| 数据集         | CELEBA, NVIDIA Irregular Mask Dataset                        | CELEBA, NVIDIA Irregular Mask Dataset                        |
| 训练参数       | train_iter=350000, finetune_iter=150000, gen_lr_train=0.0002, gen_lr_finetune=0.00005, dis_lr_multiplier=0.1, batch_size=6 | train_iter=350000, finetune_iter=150000, gen_lr_train=0.0002, gen_lr_finetune=0.00005, dis_lr_multiplier=0.1, batch_size=6 |
| 优化器         | Adam                                                         | Adam                                                         |
| 损失函数       | Reconstruction Loss (L1), Perceptual Loss (L1), Style Loss(L1), Adversarial Loss (BCE), Intermediate Loss (L1 + BCE) | Reconstruction Loss (L1), Perceptual Loss (L1), Style Loss(L1), Adversarial Loss (BCE), Intermediate Loss (L1 + BCE) |
| 速度           | 590 ms / step                                                | 230 ms / step                                                |
| 指标*          | <table><tr><td></td><td>0-20%</td><td>20-40%</td><td>40-60%</td></tr><tr><td>PSNR</td><td>37.55</td><td>28.92</td><td>23.72</td></tr><tr><td>SSIM</td><td>0.978</td><td>0.919</td><td>0.823</td></tr></table> | <table><tr><td></td><td>0-20%</td><td>20-40%</td><td>40-60%</td></tr><tr><td>PSNR</td><td>37.59</td><td>29.08</td><td>23.97</td></tr><tr><td>SSIM</td><td>0.977</td><td>0.917</td><td>0.822</td></tr></table> |

\* 由于训练过程固有的随机性，当模型重新训练时，最终的指标值会受到随机抖动的影响。

### [推理性能](#内容)

| 设备       | 时间 (ms / image) @ batch 1 |
| ---------- | --------------------------- |
| GPU V100   | 76                          |
| Ascend 910 | 29                          |

## [随机情况描述](#内容)

`train.py `脚本使用mindspore.set_seed()设置全局随机种子，可以修改。

## [ModelZoo 首页](#内容)

请访问官方网站 [首页](https://gitee.com/mindspore/models).
