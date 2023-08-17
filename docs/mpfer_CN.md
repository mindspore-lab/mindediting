# 内容

- [MPFER 简介](#mpfer-description)
- [模型架构](#model-architecture)
- [数据集](#dataset)
- [环境要求](#environmental-requirements)
- [性能](#performance)
  - [训练性能](#training-performance)
  - [评估性能](#evaluation-performance)

# [MPFER 简介](#contents)

我们介绍了第一个基于3d的多帧去噪方法，该方法在计算需求较低的情况下显著优于基于2d的去噪方法。我们的方法通过引入可学习的编码器-渲染器对来操作特征空间中的多平面表示，扩展了用于新型视图合成的多平面图像(MPI)框架。编码器跨视图融合信息并以深度方式操作，而呈现器跨深度融合信息并以视图方式操作。这两个模块端到端进行训练，并学习以无监督的方式分离深度，从而产生多平面特征(MPF)表示。在空间和真实前面向数据集以及原始突发数据上的实验验证了我们的视图合成、多帧去噪和噪声条件下的视图合成方法。

[论文](https://arxiv.org/pdf/2303.18139.pdf): Efficient View Synthesis and 3D-based Multi-Frame Denoising with Multiplane Feature Representations.

# [模型架构](#contents)

![1689649341993](image/mpfer.png)

# [数据集](#contents)

Spaces数据集由100个室内和室外场景组成，每个场景使用放置在稍微不同位置的16台摄像机拍摄5到10次。90个场景用于训练，10个场景用于评估。图片的分辨率为480×800。

数据集格式如下：

```
|--data
| |--scene_000
| | |--cam_00
| | | |--image_000.JPG
| | | |--...
| | |--cam_01
| | | |--image_000.JPG
| | | |--...
| | |--...
| | |--models.json
| | |--multi_model.pb.bin
| |--scene_009
| | |--cam_00
| | | |--image_000.JPG
| | | |--...
| | |--cam_01
| | | |--image_000.JPG
| | | |--...
| | |--...
| | |--models.json
| | |--multi_model.pb.bin
| |--...
```

# [环境要求](#contents)

## GPU

- 硬件 (GPU)
  - 准备带有GPU处理器的硬件环境
- 框架
  - [MindSpore](https://www.mindspore.cn/install)
- 详细信息请参见以下参考资料:
  - [MindSpore Tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 其他python包:
  - 手动安装其他包，或者在模型目录下使用 `pip install -r requirements.txt` 命令。

## Ascend 910

- 硬件 (Ascend)
  - 使用Ascend 910准备硬件环境 (cann_5.1.2, euler_2.8.3, py_3.7)
- 框架
  - [MindSpore 2.0.0](https://www.mindspore.cn/install) 或 更新版本

# [性能](#contents)

## [训练性能](#contents)

暂不支持训练。

## [评估性能](#contents)

| 模型    | PSNR 指标 | SSIM 指标 |
| ------- | --------- | --------- |
| MPFER16 | 32.44     | 0.91      |
