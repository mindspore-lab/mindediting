# Contents

- [FSRCNN Description](#fsrcnn-description)
- [Dataset](#dataset)
- [Environmental requirements](#environmental-requirements)
- [Performance](#performance)
    - [Training performance](#training-performance)
    - [Evaluation performance](#evaluation-performance)

# [FSRCNN Description](#contents)

As a successful deep model applied in image super-resolution (SR), the Super-Resolution Convolutional Neural Network (
SRCNN) [1,2] has demonstrated superior performance to the previous hand-crafted models either in speed and restoration
quality. However, the high computational cost still hinders it from practical usage that demands real-time performance (
24 fps). In this paper, we aim at accelerating the current SRCNN, and propose a compact hourglass-shape CNN structure
for faster and better SR. We re-design the SRCNN structure mainly in three aspects. First, we introduce a deconvolution
layer at the end of the network, then the mapping is learned directly from the original low-resolution image (without
interpolation) to the high-resolution one. Second, we reformulate the mapping layer by shrinking the input feature
dimension before mapping and expanding back afterwards. Third, we adopt smaller filter sizes but more mapping layers.
The proposed model achieves a speed up of more than 40 times with even superior restoration quality. Further, we present
the parameter settings that can achieve real-time performance on a generic CPU while still maintaining good performance.
A corresponding transfer strategy is also proposed for fast training and testing across different upscaling factors.

[Paper](https://arxiv.org/pdf/1608.00367.pdf): Accelerating the Super-Resolution Convolutional Neural Network

[Reference github repository (PyTorch)](https://github.com/yjn870/FSRCNN-pytorch)

# [Dataset](#contents)

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset  | Scale | Type  | Link                                                                      |
|----------|-------|-------|---------------------------------------------------------------------------|
| 91-image | 2     | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
| 91-image | 3     | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
| 91-image | 4     | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
| Set5     | 2     | Eval  | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0)     |
| Set5     | 3     | Eval  | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0)     |
| Set5     | 4     | Eval  | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0)     |

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

To compare results we use scale = 4 here and everywhere in the report.

| Model               | Device type | Device      | PSNR (dB) | Train time (secs per epoch) |
|---------------------|-------------|-------------|-----------|-----------------------------|
| FSRCNN-Torch-paper  | GPU         | ?           | 30.55     | ?                           |
| FSRCNN-Torch-github | GPU         | ?           | 30.50     | ?                           |
| FSRCNN-Torch        | GPU         | P100        | 30.44     | 210                         |
| FSRCNN-MS           | GPU         | P100        | 30.15     | 320                         |
| FSRCNN-MS-Ascend    | Ascend      | Ascend-910A | 30.18     | 360                         |

PSNR differences between the original (paper and github) and our implementations are small. So the result are
reproduced.

## [Evaluation Performance](#contents)

### Inference dataset info

|                       |      |
|-----------------------|------|
| Dataset name          | Set5 |
| batch size            | 8    |
| Scale                 | x4   |
| Experiments num (its) | 1000 |
| Warm up (its)         | 100  |

### Timings

| Model            | Device type | Device      | Inference time (ms, median, per it) |
|------------------|-------------|-------------|-------------------------------------|
| FSRCNN-Torch     | GPU         | P100        | 6                                   |
| FSRCNN-MS        | GPU         | P100        | 13                                  |
| FSRCNN-MS-Ascend | Ascend      | Ascend-910A | 20                                  |
