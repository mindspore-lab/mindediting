# Contents

- [TTVSR Description](#ttvsr-description)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [TTVSR Description](#contents)

Video super-resolution (VSR) aims to restore a sequence of high-resolution (HR) frames from their low-resolution (LR)
counterparts. Although some progress has been made, there are grand challenges to effectively utilize temporal
dependency in entire video sequences. Existing approaches usually align and aggregate video frames from limited adjacent
frames (e.g., 5 or 7 frames), which prevents these approaches from satisfactory results. In this paper, we take one step
further to enable effective spatio-temporal learning in videos. We propose a novel Trajectory-aware Transformer for
Video Super-Resolution (TTVSR). In particular, we formulate video frames into several pre-aligned trajectories which
consist of continuous visual tokens. For a query token, self-attention is only learned on relevant visual tokens along
spatio-temporal trajectories. Compared with vanilla vision Transformers, such a design significantly reduces the
computational cost and enables Transformers to model long-range features. We further propose a cross-scale feature
tokenization module to overcome scale-changing problems that often occur in long-range videos. Experimental results
demonstrate the superiority of the proposed TTVSR over state-of-the-art models, by extensive quantitative and
qualitative evaluations in four widely-used video super-resolution benchmarks.

[Paper](https://arxiv.org/abs/2204.04216): Learning Trajectory-Aware Transformer for Video Super-Resolution

[Reference github repository (PyTorch)](https://github.com/researchmm/TTVSR)

## Optimization

After the export to MindSpore the computation graph has been optimized for the best performance on both training and
inference phases.

# [Dataset](#contents)

1. Training set
    * [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset. We regroup the training and validation dataset
      into one folder. The original training dataset has 240 clips from 000 to 239. The original validation dataset were
      renamed from 240 to 269.
        - Make REDS structure be:
        ```
            ├────REDS
                    ├────trainval_sharp
                        ├────000
                        ├────...
                        ├────269
                    ├────trainval_sharp_bicubic
                        ├────X4
                            ├────000
                            ├────...
                            ├────269
        ```

2. Testing set
    * [REDS4](https://seungjunnah.github.io/Datasets/reds.html) dataset. The 000, 011, 015, 020 clips from the original
      training dataset of REDS.

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

### Training dataset info

|                          |        |
|--------------------------|--------|
| Dataset name             | REDS   |
| Batch size               | 2      |
| Sequence length (frames) | 50     |
| Input patch size         | 64x64  |
| Upscale                  | x4     |
| Iterations num           | 400000 |
| Warm up steps            | 100    |

### Training results

| Model           | Device type | Device      | PSNR (dB) | SSIM     | Train time (seconds per iter) | Train time (expected)\** |
|-----------------|-------------|-------------|-----------|----------|-------------------------------|--------------------------|
| TTVSR-PyTorch   | GPU         | 8 x A100    | 32.12     | 0.9021   | 4.2                           | 20 days                  |
| TTVSR-MS        | GPU         | 8 x A100    | 31.97\*   | 0.9016\* | 3.8                           | 18 days                  |
| TTVSR-MS-Ascend | Ascend      | Ascend-910A | 31.97\*   | 0.9016\* | 15.6                          | 72 days                  |

\* Metrics are measured with the exported from PyTorch original weights.
\** Training does not finished yet due to long training time but it's expected to reproduce the original result.

## [Inference performance](#contents)

### Inference dataset info

|                          |          |
|--------------------------|----------|
| Dataset name             | REDS     |
| Batch size               | 1        |
| Sequence length (frames) | 100      |
| Input resolution (HxW)   | 320x180  |
| Output resolution (HxW)  | 1280x720 |
| Upscale                  | x4       |

### Timings

| Model           | Device type | Device      | Inference time (seconds per sequence) |
|-----------------|-------------|-------------|---------------------------------------|
| TTVSR-Torch     | GPU         | V100        | 11                                    |
| TTVSR-MS        | GPU         | V100        | 7.5                                   |
| TTVSR-MS-Ascend | Ascend      | Ascend-910A | 13                                    |
