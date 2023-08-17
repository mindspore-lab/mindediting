# [模型](#内容)

## EMVD 简介

基于循环时空融合的高效多阶段视频去噪。

EMVD是一种有效的视频去噪方法，它通过多次重复的级联处理阶段，即时间融合、空间去噪和时空细化，递归地利用自然视频中固有的时空相关性。

[论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.pdf): Accelerating the Super-Resolution Convolutional Neural Network

# [数据集](#内容)

[CRVD 数据集](https://github.com/cao-cong/RViDeNet)

# 模型实现

[参考github仓库 (PyTorch)]https://github.com/Baymax-chen/EMVD)

# 训练性能

| 模型              | 设备类型 | 规格    | PSNR (dB) | 训练时间 (secs per epoch)  |
|---------------------|--------|-------------|-----------|------------------|
| EMVD-Torch  | GPU    | V100           |     42.09 | 29                |
| EMVD-MS-GPU           | GPU    | V100        |     42.12 | 28              |
| EMVD-MS-Ascend    | Ascend | Ascend-910A |     40.77 | 19              |
