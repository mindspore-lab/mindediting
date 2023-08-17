# Contents

- [RVRT Description](#rvrt-description)
- [Model-architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [RVRT Description](#contents)

Video restoration aims at restoring multiple high-quality frames from multiple lowquality frames. Existing video restoration methods generally fall into two extreme
cases, i.e., they either restore all frames in parallel or restore the video frame by frame in a recurrent way, which would result in different merits and drawbacks.
Typically, the former has the advantage of temporal information fusion. However, it suffers from large model size and intensive memory consumption; the latter has
a relatively small model size as it shares parameters across frames; however, it lacks long-range dependency modeling ability and parallelizability. In this paper,
we attempt to integrate the advantages of the two cases by proposing a recurrent video restoration transformer, namely RVRT. RVRT processes local neighboring
frames in parallel within a globally recurrent framework which can achieve a good trade-off between model size, effectiveness, and efficiency. Specifically, RVRT
divides the video into multiple clips and uses the previously inferred clip feature to estimate the subsequent clip feature. Within each clip, different frame features are
jointly updated with implicit feature aggregation. Across different clips, the guided deformable attention is designed for clip-to-clip alignment, which predicts multiple
relevant locations from the whole inferred clip and aggregates their features by the attention mechanism. Extensive experiments on video super-resolution, deblurring,
and denoising show that the proposed RVRT achieves state-of-the-art performance on benchmark datasets with balanced model size, testing memory and runtime.

Currently Video Super Resolution is supported only.

[Paper](https://arxiv.org/pdf/2206.02146.pdf): Recurrent Video Restoration Transformer with Guided Deformable Attention.

[Reference github repository (evaluation only)](https://github.com/JingyunLiang/RVRT)

[Reference github repository (both training and evaluation)](https://github.com/cszn/KAIR)

# [Model architecture](#contents)

The model consists of three parts: shallow feature extraction, recurrent feature refinement and HQ frame reconstruction. More specifically, in
shallow feature extraction, a convolution layer is used to extract features from the LQ video. After that, several Residual Swin Transformer Blocks (RSTBs) are used to extract the shallow feature. Then, recurrent feature refinement modules are used for temporal correspondence modeling and guided deformable attention for video alignment. Lastly, several
RSTBs are added to generate the final feature and reconstruct the HQ video by pixel shuffle layer.

RVRT Light is made from RVRT by replacing the guided deformable attention by simple attention and removing SpyNet.

# [Dataset](#contents)

## Dataset used

This work uses the [Vimeo90K](http://toflow.csail.mit.edu/) dataset.
Download [link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).

## Dataset preprocessing

Officially the datasets consists of high-resolution ground truth samples.
To obtain low-resolution samples should be used the `src/dataset/preprocess.py`
script that generates annotation files and downscaled by 4 times via bicubic
interpolation samples from ground truth ones.

To prepare the dataset:

1. Download and unzip the dataset
2. Run the preprocessing script:

```commandline
python mindediting/dataset/src/vimeo_preprocess.py \
  --train-annotation ${DATASET_ROOT_DIR}/vimeo_septuplet/sep_trainlist.txt \
  --test-annotation ${DATASET_ROOT_DIR}vimeo_septuplet/sep_testlist.txt \
  --images-root ${DATASET_ROOT_DIR}vimeo_septuplet/sequences \
  --output-dir ${DATASET_ROOT_DIR}vimeo_septuplet/BIx4 \
  --generate-lq
```

For more details run:

```commandline
python mindediting/dataset/src/vimeo_preprocess.py --help
```

The preprocessing script uses implementation of bicubic interpolation like in MatLab that has
a big impact on the results.

## Dataset organize way

It is recommended to setup directories with dataset as shown below. However another structure of files
is possible but needed changes in config file. To setup default file structure do the following:

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
    - Prepare hardware environment with Ascend 910 (cann_6.0.0, euler_2.8, py_3.7)
- Framework
    - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) or later

# [Performance](#contents)

## [Training Performance](#contents)

| Parameters                  |  8xAscend      |
|-----------------------------|----------------|
| Model Version               | RVRT           |
| Resources                   | 8x Ascend 910  |
| Uploaded Date               | N/A            |
| MindSpore Version           | 1.9.0          |
| Dataset                     | Vimeo90k       |
| Training Parameters         | batch_size=8   |
| Optimizer                   | Adam           |
| Speed                       | 1.42 s/step    |

| Parameters                  |  8xAscend      |
|-----------------------------|----------------|
| Model Version               | RVRT Light     |
| Resources                   | 8x Ascend 910  |
| Uploaded Date               | N/A            |
| MindSpore Version           | 1.9.0          |
| Dataset                     | Vimeo90k       |
| Training Parameters         | batch_size=8   |
| Optimizer                   | Adam           |
| Speed                       | 1.25 s/step    |

## [Evaluation Performance](#contents)

| Parameters                  | 1xGPU (FP32)                  | 1xAscend (MixedPrecison)    |
|-----------------------------|-------------------------------|-----------------------------|
| Model Version               | RVRT                          | RVRT                        |
| Resources                   | 1x Nvidia 3090TI              | 1x Ascend 910               |
| Backend                     | MindSpore 2.0.0a              | CANN 6.0.RC1.alpha005       |
| Datasets                    | Vimeo90k                      | Vimeo90k                    |
| Batch_size                  | 1                             | 1                           |
| num_frame_testing           | 14                            | 14                          |
| PSNR metric                 | 38.12                         | 38.12                       |
| GPU memory consumption      | 11.7 GB                       | N/A                         |
| Speed                       | 2.32 s/call                   | 0.74 s/call                 |

| Parameters                  | 1xGPU (FP32)                  | 1xAscend (MixedPrecison)    |
|-----------------------------|-------------------------------|-----------------------------|
| Model Version               | RVRT Light                    | RVRT light                  |
| Resources                   | 1x Nvidia 3090TI              | 1x Ascend 910               |
| Backend                     | MindSpore 2.0.0a              | CANN 6.0.RC1.alpha005       |
| Datasets                    | Vimeo90k                      | Vimeo90k                    |
| Batch_size                  | 1                             | 1                           |
| num_frame_testing           | 14                            | 14                          |
| PSNR metric                 | 37.91                         | 37.91                       |
| GPU memory consumption      | 6.3 GB                        | N/A                         |
| Speed                       | 1.9 s/call                    | 0.4 s/call                  |
