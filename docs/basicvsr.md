# Contents

- [BasicVSR Description](#basicvsr-description)
- [Model-architecture](#model-architecture)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [BasicVSR Description](#contents)

Video super-resolution (VSR) approaches tend to have more components than the image counterparts
as they need to exploit the additional temporal dimension. Complex designs are not uncommon.
In this study, we wish to untangle the knots and reconsider some most essential components for
VSR guided by four basic functionalities, i.e., Propagation, Alignment, Aggregation, and Upsampling.
By reusing some existing components added with minimal redesigns, we show a succinct pipeline,
BasicVSR, that achieves appealing improvements in terms of speed and restoration quality in
comparison to many state-of-the-art algorithms. We conduct systematic analysis to explain how such
gain can be obtained and discuss the pitfalls. We further show the extensibility of BasicVSR by
presenting an information-refill mechanism and a coupled propagation scheme to facilitate information
aggregation. The BasicVSR and its extension, IconVSR, can serve as strong baselines for future VSR approaches.

[Paper](https://arxiv.org/abs/2012.02181): BasicVSR: The Search for Essential Components in Video Super-Resolution and
Beyond.

[Reference github repository](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/README.md)

# [Model architecture](#contents)

BasicVSR has the following main design choices: for propagation, BasicVSR has chosen bidirectional
propagation with emphasis on long-term and global propagation. For alignment,
BasicVSR adopts a simple flow-based alignment but taking place at feature level.
For aggregation and upsampling, popular choices on feature concatenation and pixelshuffle suffice.

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
    - Prepare hardware environment with Ascend 910 (cann_5.1.2, euler_2.8.3, py_3.7)
- Framework
    - [MindSpore 2.0.0-alpha](https://www.mindspore.cn/install) or later

# [Performance](#contents)

## [Training Performance](#contents)

| Parameters                 | BasicVSR (2xGPU)                | BasicVSR (1xGPU)                |
|----------------------------|---------------------------------|---------------------------------|
| Model Version              | BasicVSR                        | BasicVSR                        |
| Resources                  | 2x Nvidia A100                  | 1x Nvidia A100                  |
| Uploaded Date              | 12 / 14 / 2022 (month/day/year) | 12 / 14 / 2022 (month/day/year) |
| MindSpore Version          | 1.9.0                           | 1.9.0                           |
| Dataset                    | Vimeo90K                        | Vimeo90K                        |
| Training Parameters        | batch_size=8, 37 epochs         | batch_size=8, 37 epochs         |
| Optimizer                  | Adam                            | Adam                            |
| Speed                      | 2220 ms/step                    | 2807 ms/step                    |
| Total time                 | 8d 8h 23m                       | 10d 6h 40m                      |
| Checkpoint for Fine tuning | 99 MB (.ckpt file)              | 99 MB (.ckpt file)              |

| Parameters                 | BasicVSR (8xNPU)                |
|----------------------------|---------------------------------|
| Model Version              | BasicVSR                        |
| Resources                  | 8x Ascend 910                   |
| Uploaded Date              | 12 / 14 / 2022 (month/day/year) |
| MindSpore Version          | 2.0.0.20221118                  |
| Dataset                    | Vimeo90K                        |
| Training Parameters        | batch_size=8, 37 epochs         |
| Optimizer                  | Adam                            |
| Speed                      | 920 ms/step                     |
| Total time                 | 3d 4h 23m                       |
| Checkpoint for Fine tuning | 99 MB (.ckpt file)              |

## [Evaluation Performance](#contents)

| Parameters        | BasicVSR (1xGPU)                | BasicVSR (2xGPU)                | BasicVSR (8xNPU)                |
|-------------------|---------------------------------|---------------------------------|---------------------------------|
| Model Version     | BasicVSR                        | BasicVSR                        | BasicVSR                        |
| Resources         | 1x Nvidia A100                  | 2x Nvidia A100                  | 8x Ascend 910                   |
| Uploaded Date     | 12 / 14 / 2022 (month/day/year) | 12 / 14 / 2022 (month/day/year) | 12 / 14 / 2022 (month/day/year) |
| MindSpore Version | 1.9.0                           | 1.9.0                           | 2.0.0.20221118                  |
| Datasets          | Vimeo90K                        | Vimeo90K                        | Vimeo90K                        |
| Batch_size        | 1                               | 1                               | 1                               |
| PSNR metric       | 37.2                            | 37.21                           | 37.23                           |
