# Contents

- [BasicVSR++Light Description](#basicvsr-description)
- [Model-architecture](#model-architecture)
- [Tasks](#tasks)
- [Datasets](#datasets)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [BasicVSR++Light Description](#contents)

 A recurrent structure is a popular framework choice for the task of video super-resolution.
 The state-of-the-art method BasicVSR adopts bidirectional propagation with feature alignment
 to effectively exploit information from the entire input video. In this study, we redesign
 BasicVSR by proposing second-order grid propagation and flow-guided deformable alignment.
 We show that by empowering the recurrent framework with the enhanced propagation and alignment,
 one can exploit spatiotemporal information across misaligned video frames more effectively.
 The new components lead to an improved performance under a similar computational constraint.
 In particular, our model BasicVSR++ surpasses BasicVSR by 0.82 dB in PSNR with similar number
 of parameters. In addition to video super-resolution, BasicVSR++ generalizes well to other
 video restoration tasks such as compressed video enhancement.

[Paper](https://arxiv.org/abs/2104.13371): BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

[Reference github repository](https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/basicvsr_plusplus)

# [Model architecture](#contents)

This is a lightweight implementation of the BasicVSR++ model with the following optimizations
for MindSpore:
- SPyNet is excluded
- Deformable convolutions replaced by 5 residual blocks.

# [Tasks](#contents)

This model is used for the following tasks:
- Video Super-Resolution (VSR)
- Video Denoise (VDN)
- Video Deblur (VDB)
- Video Enhancement (VE)

# [Datasets](#contents)

## Video Super-Resolution

### Dataset used

This work uses the [Vimeo90K](http://toflow.csail.mit.edu/) dataset.
Download [link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).

### Dataset preprocessing

Oficially the datasets consists of high-resolution ground truth samples.
To obtain low-resolution samples should be used the `src/dataset/preprocess.py`
script that generates annotation files and downscaled by 4 times via bicubic
interpolation samples from ground truth ones.

To prepare the dataset:

1. Download and unzip the dataset
2. Run the preprocessing script:

```commandline
python vimeo_preprocess.py \
  --train-annotation ${DATASET_ROOT_DIR}/vimeo_septuplet/sep_trainlist.txt \
  --test-annotation ${DATASET_ROOT_DIR}vimeo_septuplet/sep_testlist.txt \
  --images-root ${DATASET_ROOT_DIR}vimeo_septuplet/sequences \
  --output-dir ${DATASET_ROOT_DIR}vimeo_septuplet/BIx4 \
  --generate-lq
```

For more details run:

```commandline
python vimeo_preprocess.py --help
```

The preprocessing script uses implementation of bicubic interpolation like in MatLab that has
a big impact on the results.

### Dataset organize way

It is recommended to set up directories with dataset as shown below. However, another structure of files
is possible but needed changes in config file. To set up default file structure do the following:

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

## Video Denoise

### Dataset used
This work uses the [DAVIS 2017](https://davischallenge.org/) dataset with resolution 480p.
Download [link](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip)

### Dataset organize way
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

## Video Deblur

### Dataset used
This work uses the [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset.
Download [link](https://seungjunnah.github.io/Datasets/reds.html)

### Dataset organize way

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

## Video Enhancement

### Dataset used
This work uses the [LDV 2.0](https://github.com/RenYang-home/LDV_dataset) dataset.
Download [link](https://github.com/RenYang-home/LDV_dataset#ldv-20-ldv-10--95-videos--335-videos)

### Dataset preprocessing

To prepare the dataset:

1. Download and unzip the dataset
2. Run the preprocessing script `run.sh`

### Dataset organize way

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

| Parameters                 | BasicVSR++ VSR (1xNPU)          | BasicVSR++ VDB (1xNPU)          | BasicVSR++ VDN (8xNPU)          | BasicVSR++ VE (8xNPU)           |
|----------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| Model Version              | BasicVSR++Light                 | BasicVSR++Light                 | BasicVSR++Light                 | BasicVSR++Light                 |
| Resources                  | 1x Ascend 910                   | 1x Ascend 910                   | 8x Ascend 910                   | 8x Ascend 910                   |
| Uploaded Date              | 03 / 21 / 2023 (month/day/year) | 06 / 01 / 2023 (month/day/year) | 04 / 20 / 2023 (month/day/year) | 05 / 30 / 2023 (month/day/year) |
| MindSpore Version          | 2.0.0.20221118                  | 2.0.0-alpha                     | 2.0.0-alpha                     | 2.0.0-alpha                     |
| Dataset                    | Vimeo90K                        | REDS                            | DAVIS                           | LDV_V2                          |
| Training Parameters        | batch_size=8, 37 epochs         | batch_size=2, 7000              | batch_size=1, 55000 epochs      | batch_size=4, 10000 epochs      |
| Optimizer                  | Adam                            | Adam                            | Adam                            | Adam                            |
| Speed                      | 960 ms/step                     | 2240 ms/step                    | 205 ms/step                     | 420 ms/step                     |
| Total time                 | 4d 2h 7m                        | 5d 0h 11m                       | 1d 14h 30m                      | 1d 11h 2m                       |
| Checkpoint for Fine tuning | 185.4 MB (.ckpt file)           | 269.63 MB (.ckpt file)          | 201.76 MB (.ckpt file)          | 40.80 MB (.ckpt file)           |

## [Evaluation Performance](#contents)

| Parameters         | BasicVSR++ VSR (1xNPU, CANN)    | BasicVSR++ VDB (1xNPU, CANN)    | BasicVSR++ VDN (1xNPU, CANN)    | BasicVSR++ VE (1xNPU, CANN)     |
|--------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| Model Version      | BasicVSR++Light                 | BasicVSR++Light                 | BasicVSR++Light                 | BasicVSR++Light                 |
| Resources          | 1x Ascend 910                   | 1x Ascend 910                   | 1x Ascend 910                   | 1x Ascend 910                   |
| Uploaded Date      | 03 / 21 / 2023 (month/day/year) | 06 / 01 / 2023 (month/day/year) | 04 / 20 / 2023 (month/day/year) | 05 / 30 / 2023 (month/day/year) |
| MindSpore Version  | 2.0.0.20221118                  | 2.0.0-alpha                     | 2.0.0-alpha                     | 2.0.0-alpha                     |
| Datasets           | Vimeo90K                        | REDS                            | Set8                            | LDV_V2                          |
| Batch_size         | 1                               | 1                               | 1                               | 1                               |
| Inference speed, s | 0.773 (per 25 frames, 480x270)  | 1.064 (per 25 frames, 1920x1080)| 0.951 (per 25 frames, 1920x1080)| 0.780 (per 26 frames, 1920x1080)|
| PSNR metric        | 37.61                           | 32.48                           | 30.01  (noise sigma 50)         | 32.46                           |
| SSIM metric        | 0.9487                          | 0.9336                          | 0.851  (noise sigma 50)         | 0.8835                          |
