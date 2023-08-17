# 内容

- [内容](#内容)
- [MIMO-UNet 简介](#mimo-unet-简介)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速启动](#快速启动)
- [脚本介绍](#脚本介绍)
    - [脚本和示例代码](#脚本和示例代码)
    - [训练过程](#训练过程)
        - [独立训练](#独立训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [推理过程](#推理过程)
        - [导出 MindIR](#导出-mindir)
- [模型介绍](#模型介绍)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [评估性能](#评估性能)
- [随机情况描述](#内容)
- [ModelZoo 首页](#modelzoo-首页)

# [MIMO-UNet 简介](#内容)

由粗到细的策略已广泛应用于单幅图像去模糊网络的体系结构设计。传统的方法通常是在多尺度输入图像的子网络上进行堆叠，从底层子网络到顶层子网络逐步提高图像的锐度，不可避免地会产生较高的计算成本。为了实现快速准确的去模糊网络设计，我们重新审视了从粗到细的策略，并提出了一个多输入多输出U-net (MIMO-UNet)。首先，MIMO-UNet的单编码器采用多尺度输入图像，以减轻训练难度。其次，MIMO-UNet的单解码器输出多个不同尺度的去模糊图像，使用单个u型网络模拟多级联u型网络。最后，引入非对称特征融合，实现多尺度特征的高效融合。在GoPro和RealBlur数据集上的大量实验表明，所提出的网络在准确性和计算复杂性方面都优于最先进的方法。

[论文](https://arxiv.org/abs/2108.05054): Rethinking Coarse-to-Fine Approach in Single Image Deblurring.

[参考 github 仓库](https://github.com/chosj95/MIMO-UNet)

# [模型架构](#内容)

MIMO-UNet的架构是基于单一的U-Net，并对有效的多尺度去模糊进行了重大修改。MIMO-UNet的编码器和解码器由三个编码器块(EBs)和解码器块(db)组成，它们使用卷积层从不同阶段提取特征。

# [数据集](#内容)

## 使用数据集

经过处理的GOPRO数据集位于`dataset`文件夹中。

数据集链接 (Google
Drive): [GOPRO_Large](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)

提出了GOPRO_Large数据集动态场景去模糊。训练集和测试集是公开可用的。

- 数据集大小: ~6.2G
    - 训练: 3.9G, 2103 图像对
    - 测试: 2.3G, 1111 图像对
    - 数据格式: 图像
    - 注意: 数据将在src/data_augment.py和src/data_load.py中处理

## 数据集组织方式

```text
.
└─ GOPRO_Large
  ├─ train
  │  ├─ GOPR0xxx_xx_xx
  │  │  ├─ blur
  │  │  │  ├─ ***.png
  │  │  │  └─ ...
  │  │  ├─ blur_gamma
  │  │  │  ├─ ***.png
  │  │  │  └─ ...
  │  │  ├─ sharp
  │  │  │  ├─ ***.png
  │  │  │  └─ ...
  │  │  └─ frames X offset X.txt
  │  └─ ...
  └─ test
     ├─ GOPR0xxx_xx_xx
     │  ├─ blur
     │  │  ├─ ***.png
     │  │  └─ ...
     │  ├─ blur_gamma
     │  │  ├─ ***.png
     │  │  └─ ...
     │  └─ sharp
     │     ├─ ***.png
     │     └─ ...
     └─ ...
```

## 数据集预处理

下载数据集后，运行位于``src``文件夹中的 `preprocessing.py` 脚本。
下面是下载数据集的文件结构。

参数说明:

- `--root_src` - 原始数据集根目录的路径, 包含 `train/` and `test/` 文件夹。
- `--root_dst` - 目录的路径，预处理数据集将存储在其中。

```bash
python src/preprocessing.py --root_src /path/to/original/dataset/root --root_dst /path/to/preprocessed/dataset/root
```

### 数据集预处理后的组织方式

在上面的例子中，测试脚本执行后，预处理的图像将存储在/path/to/preprocessed/dataset/root路径下。下面是预处理数据集的文件结构。

```text
.
└─ GOPRO_preprocessed
  ├─ train
  │  ├─ blur
  │  │  ├─ 1.png
  │  │  ├─ ...
  │  │  └─ 2103.png
  │  └─ sharp
  │     ├─ 1.png
  │     ├─ ...
  │     └─ 2103.png
  └─ test
     ├─ blur
     │  ├─ 1.png
     │  ├─ ...
     │  └─ 1111.png
     └─ sharp
        ├─ 1.png
        ├─ ...
        └─ 1111.png
```

# [环境要求](#内容)

## GPU

- 硬件 (GPU)

    - 准备GPU处理器硬件环境
- 框架

    - [MindSpore](https://www.mindspore.cn/install)
- 具体请参见以下资源:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 其他python包:
    - Pillow
    - scikit-image
    - PyYAML

  手动或在模型目录下使用' pip Install -r requirements.txt '命令安装其他包。

## Ascend 910

- 硬件 (Ascend)
    - 准备 Ascend 910 硬件环境(cann_5.1.2, euler_2.8.3, py_3.7)
- 框架
    - [MindSpore Ascend 1.8.0](https://www.mindspore.cn/install)

## [快速启动](#内容)

通过官方网站和附加包安装MindSpore后，您可以开始如下的训练和评估:

- GPU运行

    ```bash
    # run the training example
    python ./train.py --dataset_root /path/to/dataset/root --ckpt_save_directory /save/checkpoint/directory
    # or
    bash scripts/run_standalone_train_gpu.sh /path/to/dataset/root /save/checkpoint/directory

    # run the distributed training example
    bash scripts/run_distribute_train_gpu.sh /path/to/dataset/root /save/checkpoint/directory

    # run the evaluation example
    python ./eval.py --dataset_root /path/to/dataset/root \
                      --ckpt_file /path/to/eval/checkpoint.ckpt \
                      --img_save_directory /path/to/result/images
    # or
    bash scripts/run_eval_gpu.sh /path/to/dataset/root /path/to/eval/checkpoint.ckpt /path/to/result/images
    ```

- Ascend 910运行

  建立模型的参数存储在` ./configs/ascend_config.yaml `中。命令行参数允许您更改config中的值。

  ```bash
    # run the evaluation example with pretrained weights on full dataset
    python ./eval.py \
     --dataset_root /path/to/dataset/root \
     --config_path ./configs/ascend_config.yaml \
     --ckpt_file ./weights_pretrained/MIMO-UNet_3000_1_fp16_o3_dls.ckpt
  ```

  ```bash
    # run the training example with 10 imgs and 4 epochs
    python ./train.py \
     --dataset_root /path/to/dataset/root \
     --config_path ./configs/ascend_config.yaml \
     --epochs_num 2 \
     --slice_dataset 10 \
     --val_acc_monitor_interval 1
  ```

# [脚本介绍](#内容)

## [脚本和示例代码](#内容)

```text
.
└─ cv
  └─ MIMO-UNet
    ├── configs
      ├── ascend_config.yaml               # Config for training on Ascend
      ├── gpu_config.yaml                  # Config for training on GPU
    ├── scripts
      ├── run_distribute_train_gpu.sh      # Distributed training on GPU shell script
      ├── run_standalone_train_gpu.sh      # Shell script for single GPU training
      ├── run_eval_gpu.sh                  # GPU evaluation script
    ├─ src
      ├─ callback.py                       # Callbacks for train and eval
      ├─ config.py                         # Config handler
      ├─ data_augment.py                   # Augmentation
      ├─ data_load.py                      # Dataloader
      ├─ init_weights.py                   # Weights initializers
      ├─ layers.py                         # Model layers
      ├─ loss.py                           # Loss function
      ├─ metric.py                         # Metrics
      ├─ mimo_unet.py                      # MIMO-UNet architecture
      ├─ preprocessing.py
    ├─ weights_pretrained                  # Contains pre-trained checkpoints
      ├─ MIMO-UNet_3000_1_fp16_o3_dls.ckpt # Example of checkpoint after training
      ├─ MIMO-UNet_pytorch_weights.ckpt    # Checkpoint from original PyTorch model
    ├─ eval.py                             # test script
    ├─ train.py                            # train script
    ├─ export.py                           # export script
    ├─ requirements.txt                    # requirements
    └─ README.md                           # MIMO-UNet file English description
```

## [训练过程](#内容)

### [独立训练](#内容)

- GPU运行

  参数说明:

    - `--dataset_root` - 数据集根目录的路径， 包含 `train/` and `test/`文件夹
    - `--ckpt_save_directory` - 输出目录，其中将存储来自训练过程的数据

    ```bash
    python ./train.py --dataset_root /path/to/dataset/root --ckpt_save_directory /save/checkpoint/directory
    # or
    bash scripts/run_standalone_train_gpu.sh [DATASET_PATH] [OUTPUT_CKPT_DIR]
    ```

    - DATASET_PATH - 数据集根目录的路径， 包含 `train/` and `test/`文件夹
    - OUTPUT_CKPT_DIR - 输出目录，其中将存储来自训练过程的数据

### [分布式训练](#内容)

- GPU运行

    ```bash
    bash scripts/run_distribute_train_gpu.sh [DATASET_PATH] [OUTPUT_CKPT_DIR]
    ```

    - DATASET_PATH - 数据集根目录的路径， 包含 `train/` and `test/`文件夹
    - OUTPUT_CKPT_DIR - 输出目录，其中将存储来自训练过程的数据

## [评估过程](#内容)

### [评估](#内容)

计算PSNR指标并保存去模糊的图像。

在计算时，选择最后生成的参数文件，并将其传递给验证脚本的相应参数。

- GPU运行

  参数说明:

    - `--dataset_root` - 数据集根目录的路径， 包含 `train/` and `test/`文件夹
    - `--ckpt_file` - 包含训练模型权重参数文件的路径。
    - `--img_save_directory` - 输出目录，验证过程中的图像将存储在其中。
      可选参数。如果未指定，验证图像将不会保存。

    ```bash
    python ./eval.py --dataset_root /path/to/dataset/root \
                     --ckpt_file /path/to/eval/checkpoint.ckpt \
                     --img_save_directory /path/to/result/images  # save validation images
    # or
    python ./eval.py --dataset_root /path/to/dataset/root \
                     --ckpt_file /path/to/eval/checkpoint.ckpt  # don't save validation images
    # or
    bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [SAVE_IMG_DIR]  # save validation images
    # or
    bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH]  # don't save validation images
    ```

    - DATASET_PATH - 数据集根目录的路径， 包含 `train/` and `test/`文件夹
    - CKPT_PATH - 包含训练模型权重参数文件的路径。
    - SAVE_IMG_DIR -  Output directory, where the images from the validation process will be stored. Optional parameter.If not specified, validation images will not be saved.


执行测试脚本后，去模糊的图像存储在``/path/to/result/img/``(如果指定了路径)中。

## [推理过程](#内容)

### [导出 MindIR](#内容)

```bash
python export.py --ckpt_file /path/to/mimounet/checkpoint.ckpt --export_device_target GPU --export_file_format MINDIR
```

该脚本将在当前目录中生成相应的MINDIR文件。

# [模型介绍](#内容)

## [性能](#内容)

### [训练性能](#内容)

| 参数           | MIMO-UNet (1xGPU)                                     | MIMO-UNet (8xGPU)                                     |
| -------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| 模型版本       | MIMO-UNet                                             | MIMO-UNet                                             |
| 资源           | 1x NV RTX3090-24G                                     | 8x NV RTX3090-24G                                     |
| 上传日期       | 04 / 12 / 2022 (month/day/year)                       | 04 / 12 / 2022 (month/day/year)                       |
| MindSpore 版本 | 1.6.1                                                 | 1.6.1                                                 |
| 数据集         | GOPRO_Large                                           | GOPRO_Large                                           |
| 训练参数       | batch_size=4, lr=0.0001 and bisected every 500 epochs | batch_size=4, lr=0.0005 and bisected every 500 epochs |
| 优化器         | Adam                                                  | Adam                                                  |
| 输出           | images                                                | images                                                |
| 速度           | 132 ms/step                                           | 167 ms/step                                           |
| 总时间         | 5d 6h 4m                                              | 9h 15m                                                |
| 微调参数文件   | 26MB(.ckpt file)                                      | 26MB(.ckpt file)                                      |

| 参数           | MIMO-UNet (1xNPU)                                      |
| -------------- | ------------------------------------------------------ |
| 模型版本       | MIMO-UNet                                              |
| 资源           | 1x Ascend 910                                          |
| 上传日期       | 15 / 11 / 2022 (month/day/year)                        |
| MindSpore 版本 | 1.8.0                                                  |
| 数据集         | GOPRO_Large                                            |
| 训练参数       | batch_size=4, lr=0.0001 and bisected every 500 epochs, |
|                | net fp16, O3, DynamicLossScaleManager                  |
| 优化器         | Adam                                                   |
| 输出           | images                                                 |
| 速度           | 63 ms/step                                             |
| 总时间         | 1d 4h 54m                                              |
| 微调参数文件   | 109MB(.ckpt file)                                      |

### [评估性能](#内容)

| 参数             | MIMO-UNet (1xGPU)               | MIMO-UNet train fp16 O3 DLS     | MIMO-UNet with original weights |
| ---------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| 模型版本         | MIMO-UNet                       | MIMO-UNet                       | MIMO-UNet                       |
| 资源             | 1x V100                         | 1x Ascend 910                   | 1x Ascend 910                   |
| 上传日期         | 12 / 21 / 2022 (month/day/year) | 12 / 21 / 2022 (month/day/year) | 12 / 21 / 2022 (month/day/year) |
| MindSpore 版本   | 1.9.0                           | 1.9.0                           | 1.9.0                           |
| 数据集           | GOPRO_Large                     | GOPRO_Large                     | GOPRO_Large                     |
| 一次处理数据数目 | 1                               | 1                               | 1                               |
| 速度             | 195 ms/step                     | 61 ms/step                      | 63 ms/step                      |
| 输出             | images                          | images                          | images                          |
| PSNR 指标        | 1p: 31.47                       | 1p: 31.511                      | 1p: 31.7                        |

# [随机情况描述](#内容)

在train.py中，我们在`train`函数中设置了种子。

在val.py中，我们在`val`函数中设置了种子。

# [ModelZoo 首页](#内容)

请查询官方网页 [首页](https://gitee.com/mindspore/models)
