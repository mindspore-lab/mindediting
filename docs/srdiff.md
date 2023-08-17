# Contents

- [SRDiff Description](#srdiff-description)
- [Model-architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [SRDiff Description](#contents)

Single image super-resolution (SISR) aims to reconstruct high-resolution (HR) images from
the given low-resolution (LR) ones, which is an ill-posed problem because one LR image
corresponds to multiple HR images. Recently, learning-based SISR methods have greatly
outperformed traditional ones, while suffering from over-smoothing, mode collapse or large
model footprint issues for PSNR-oriented, GAN-driven and flow-based methods respectively.
To solve these problems, we propose a novel single image super-resolution diffusion
probabilistic model (SRDiff), which is the first diffusion-based model for SISR. SRDiff is
optimized with a variant of the variational bound on the data likelihood and can provide
diverse and realistic SR predictions by gradually transforming the Gaussian noise into a
super-resolution (SR) image conditioned on an LR input through a Markov chain. In addition,
we introduce residual prediction to the whole framework to speed up convergence.
Our extensive experiments on facial and general benchmarks (CelebA and DIV2K datasets)
show that 1) SRDiff can generate diverse SR results in rich details with state-of-the-art
performance, given only one LR input; 2) SRDiff is easy to train with a small footprint;
and 3) SRDiff can perform flexible image manipulation including latent space interpolation
and content fusion.

[Paper](https://arxiv.org/abs/2104.14951): SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models.

[Reference github repository](https://github.com/LeiaLi/SRDiff)

# [Model architecture](#contents)

SRDiff consists of encoder RRDB and noise prediction net UNet, number of diffusion steps is 100.
As input the model takes low resolution image and upscaled this image by bicubic interpolation.

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

| Parameters                 | SRDiff (1xNPU)                  |
|----------------------------|---------------------------------|
| Model Version              | SRDiff                          |
| Resources                  | 1x Ascend 910                   |
| Uploaded Date              | 05 / 24 / 2023 (month/day/year) |
| MindSpore Version          | 2.0.0.20221118                  |
| Dataset                    | DIV2K                           |
| Training Parameters        | batch_size=64, 60 epochs        |
| Optimizer                  | Adam                            |
| Speed                      | 720 ms/step                     |
| Total time                 | 22h                             |
| Checkpoint for Fine tuning | 575 MB (.ckpt file)             |

## [Evaluation Performance](#contents)

| Parameters         | SRDiff (1xNPU, CANN)            |
|--------------------|---------------------------------|
| Model Version      | SRDiff                          |
| Resources          | 1x Ascend 910                   |
| Uploaded Date      | 05 / 24 / 2023 (month/day/year) |
| MindSpore Version  | 2.0.0.20221118                  |
| Datasets           | DIV2K                           |
| Batch_size         | 1                               |
| Inference speed, s | 0.96 (per 1 image, 320x270)     |
| PSNR metric        | 28.78                           |
| LPIPS metric       | 0.06                            |
