# Contents

- [MIMO-UNet Description](#mimo-unet-description)
- [Model-architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [MIMO-UNet Description](#contents)

Coarse-to-fine strategies have been extensively used for the architecture design of single image deblurring networks.
Conventional methods typically stack sub-networks with multi-scale input images and gradually improve sharpness of
images from the bottom sub-network to the top sub-network, yielding inevitably high computational costs. Toward a fast
and accurate deblurring network design, we revisit the coarse-to-fine strategy and present a multi-input multi-output
U-net (MIMO-UNet). First, the single encoder of the MIMO-UNet takes multi-scale input images to ease the difficulty
of training. Second, the single decoder of the MIMO-UNet outputs multiple deblurred images with different scales to
mimic multi-cascaded U-nets using a single U-shaped network. Last, asymmetric feature fusion is introduced to merge
multi-scale features in an efficient manner. Extensive experiments on the GoPro and RealBlur datasets demonstrate that
the proposed network outperforms the state-of-the-art methods in terms of both accuracy and computational complexity.

[Paper](https://arxiv.org/abs/2108.05054): Rethinking Coarse-to-Fine Approach in Single Image Deblurring.

[Reference github repository](https://github.com/chosj95/MIMO-UNet)

# [Model architecture](#contents)

The architecture of MIMO-UNet is based on a single U-Net with significant modifications for efficient multi-scale
deblurring. The encoder and decoder of MIMO-UNet are composed of three encoder blocks (EBs) and decoder blocks (DBs)
that use convolutional layers to extract features from different stages.

# [Dataset](#contents)

## Dataset used

The processed GOPRO dataset is located in the "dataset" folder.

Dataset link (Google
Drive): [GOPRO_Large](https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing)

GOPRO_Large dataset is proposed for dynamic scene deblurring. Training and Test set are publicly available.

- Dataset size: ~6.2G
    - Train: 3.9G, 2103 image pairs
    - Test: 2.3G, 1111 image pairs
    - Data format: Images
    - Note: Data will be processed in src/data_augment.py and src/data_load.py

## Dataset organize way

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

## Dataset preprocessing

After downloading the dataset, run the `preprocessing.py` script located in the folder `src`.
Below is the file structure of the downloaded dataset.

Parameter description:

- `--root_src` - Path to the original dataset root, containing `train/` and `test/` folders.
- `--root_dst` - Path to the directory, where the pre-processed dataset will be stored.

```bash
python src/preprocessing.py --root_src /path/to/original/dataset/root --root_dst /path/to/preprocessed/dataset/root
```

### Dataset organize way after preprocessing

In the example above, after the test script is executed, the pre-processed images will be stored under
the /path/to/preprocessed/dataset/root path. Below is the file structure of the preprocessed dataset.

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

# [Environmental requirements](#contents)

## GPU

- Hardware (GPU)
    - Prepare hardware environment with GPU processor
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For details, see the following resources:
    - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- Additional python packages:
    - Pillow
    - scikit-image
    - PyYAML

  Install additional packages manually or using `pip install -r requirements.txt` command in the model directory.

## Ascend 910

- Hardware (Ascend)
    - Prepare hardware environment with Ascend 910 (cann_5.1.2, euler_2.8.3, py_3.7)
- Framework
    - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) or later

# [Performance](#contents)

## [Training Performance](#contents)

| Parameters                 | MIMO-UNet (1xGPU)                                     | MIMO-UNet (8xGPU)                                     |
|----------------------------|-------------------------------------------------------|-------------------------------------------------------|
| Model Version              | MIMO-UNet                                             | MIMO-UNet                                             |
| Resources                  | 1x NV RTX3090-24G                                     | 8x NV RTX3090-24G                                     |
| Uploaded Date              | 04 / 12 / 2022 (month/day/year)                       | 04 / 12 / 2022 (month/day/year)                       |
| MindSpore Version          | 1.6.1                                                 | 1.6.1                                                 |
| Dataset                    | GOPRO_Large                                           | GOPRO_Large                                           |
| Training Parameters        | batch_size=4, lr=0.0001 and bisected every 500 epochs | batch_size=4, lr=0.0005 and bisected every 500 epochs |
| Optimizer                  | Adam                                                  | Adam                                                  |
| Outputs                    | images                                                | images                                                |
| Speed                      | 132 ms/step                                           | 167 ms/step                                           |
| Total time                 | 5d 6h 4m                                              | 9h 15m                                                |
| Checkpoint for Fine tuning | 26MB(.ckpt file)                                      | 26MB(.ckpt file)                                      |

| Parameters                 | MIMO-UNet (1xNPU)                                      |
|----------------------------|--------------------------------------------------------|
| Model Version              | MIMO-UNet                                              |
| Resources                  | 1x Ascend 910                                          |
| Uploaded Date              | 15 / 11 / 2022 (month/day/year)                        |
| MindSpore Version          | 1.8.0                                                  |
| Dataset                    | GOPRO_Large                                            |
| Training Parameters        | batch_size=4, lr=0.0001 and bisected every 500 epochs, |
|                            | net fp16, O3, DynamicLossScaleManager                  |
| Optimizer                  | Adam                                                   |
| Outputs                    | images                                                 |
| Speed                      | 63 ms/step                                             |
| Total time                 | 1d 4h 54m                                              |
| Checkpoint for Fine tuning | 109MB(.ckpt file)                                      |

## [Evaluation Performance](#contents)

| Parameters        | MIMO-UNet (1xGPU)               | MIMO-UNet train fp16 O3 DLS     | MIMO-UNet with original weights |
|-------------------|---------------------------------|---------------------------------|---------------------------------|
| Model Version     | MIMO-UNet                       | MIMO-UNet                       | MIMO-UNet                       |
| Resources         | 1x V100                         | 1x Ascend 910                   | 1x Ascend 910                   |
| Uploaded Date     | 12 / 21 / 2022 (month/day/year) | 12 / 21 / 2022 (month/day/year) | 12 / 21 / 2022 (month/day/year) |
| MindSpore Version | 1.9.0                           | 1.9.0                           | 1.9.0                           |
| Datasets          | GOPRO_Large                     | GOPRO_Large                     | GOPRO_Large                     |
| Batch_size        | 1                               | 1                               | 1                               |
| Speed             | 195 ms/step                     | 61 ms/step                      | 63 ms/step                      |
| Outputs           | images                          | images                          | images                          |
| PSNR metric       | 1p: 31.47                       | 1p: 31.511                      | 1p: 31.7                        |
