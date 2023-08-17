# Contents

- [EMVD Description](#emvd-description)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [EMVD Description](#contents)

Efficient Multi-Stage Video Denoising With Recurrent Spatio-Temporal Fusion.

EMVD is an efficient video denoising method which recursively exploit the spatio temporal correlation inherently present
in natural videos through multiple cascading processing stages applied in a recurrent fashion, namely temporal fusion,
spatial denoising, and spatio-temporal refinement.

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.pdf): Accelerating the Super-Resolution Convolutional Neural Network

[Reference github repository (PyTorch)](https://github.com/Baymax-chen/EMVD)

# [Dataset](#contents)

[CRVD Dataset](https://github.com/cao-cong/RViDeNet)

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
    - [MindSpore Ascend 1.9.0](https://www.mindspore.cn/install) or later

# [Performance](#contents)

## [Training Performance](#contents)
## [Evaluation Performance](#contents)

| Model               | Device type | Device      | PSNR (dB) | Train time (secs per epoch)      |
|---------------------|--------|-------------|-----------|------------------|
| EMVD-Torch  | GPU    | V100           |     42.09 | 29                |
| EMVD-MS-GPU           | GPU    | V100        |     42.12 | 28              |
| EMVD-MS-Ascend    | Ascend | Ascend-910A |     40.77 | 19              |
