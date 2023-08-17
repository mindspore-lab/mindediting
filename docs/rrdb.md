# Contents

- [RRDB Description](#rrdb-description)
- [Model-architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [RRDB Description](#contents)

Residual-in-Residual Dense Block (RRDB) combines multi-level residual network and dense
connections. Based on the observation that more layers and connections could always boost
performance, the proposed RRDB employs a deeper and more complex structure than the original
residual block. Specifically the proposed RRDB has a residual-in-residual structure,
where residual learning is used in different levels, so the network capacity becomes higher
benefiting from the dense connections. Generally, RRDB used as an encoder in generative models
but due to a fact that this model pretrained as a SISR model may be used as an independent one.

[Paper](https://arxiv.org/abs/1809.00219): ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks

[Reference github repository](https://github.com/LeiaLi/SRDiff)

# [Model architecture](#contents)

RRDB is quite simple and consists of sequence of residual-in-residual dense blocks.
Current implementation uses these blocks with 64 channels in intermediate layers and
differs from the original one in upsampling stage where is bilinear interpolation instead
nearest one.

# [Dataset](#contents)

## Dataset used

This work uses the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (data from NTIRE 2017)
and [Flickr2K](https://www.kaggle.com/datasets/hliang001/flickr2k) datasets.
`DIV2K` dataset includes train subset (800 images) and validation subset (100 images),
`Flick2K` consists of 2650 images which are used in training. Together this two datasets named
`DF2K`.

## Dataset preprocessing

Officially the datasets consists of high-resolution ground truth samples and low-resolution ones.
The model trained on small patches from images. To preprocess patches please use the
`src/dataset/div2k_preprocess.py` script that generates small crops from high- and
low-resolution samples.

To prepare the dataset:

1. Download and unzip the dataset
2. Run the preprocessing script:

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

For more details run:

```commandline
python div2k_preprocess.py --help
```

## Dataset organize way

It is recommended to set up directories with dataset as shown below. However, another structure of files
is possible but needed changes in config file. To set up default file structure do the following:

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
    - Install additional packages manually or using `pip install -r requirements.txt` command in the model directory.

## Ascend 910

- Hardware (Ascend)
    - Prepare hardware environment with Ascend 910 (cann_5.1.2, euler_2.8.3, py_3.7)
- Framework
    - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) or later

# [Performance](#contents)

## [Training Performance](#contents)

| Parameters                 | RRDB (1xNPU)                    |
|----------------------------|---------------------------------|
| Model Version              | RRDB                            |
| Resources                  | 1x Ascend 910                   |
| Uploaded Date              | 05 / 24 / 2023 (month/day/year) |
| MindSpore Version          | 2.0.0.20221118                  |
| Dataset                    | DIV2K                           |
| Training Parameters        | batch_size=128, 220 epochs      |
| Optimizer                  | Adam                            |
| Speed                      | 500 ms/step                     |
| Total time                 | 36h                             |
| Checkpoint for Fine tuning | 97 MB (.ckpt file)              |

## [Evaluation Performance](#contents)

| Parameters         | RRDB (1xNPU, CANN)              |
|--------------------|---------------------------------|
| Model Version      | RRDB                            |
| Resources          | 1x Ascend 910                   |
| Uploaded Date      | 05 / 24 / 2023 (month/day/year) |
| MindSpore Version  | 2.0.0.20221118                  |
| Datasets           | DIV2K                           |
| Batch_size         | 1                               |
| Inference speed, s | 1.3 (per 1 image, 1920x1080)    |
| PSNR metric        | 30.73                           |
| SSIM metric        | 0.845                           |
