<div align="center">

# MindEditing

[English](README.md) | 中文

[简介](#简介) |
[依赖](#依赖) |
[开始](#开始) |
[新闻](#新闻) |
[教程](#教程) |
[模型列表](#模型列表)

</div>

## 简介

MindEditing是基于MindSpore的开源工具包，包含开源或华为技术有限公司最先进的图像和视频任务模型 ，例如 IPT, FSRCNN, BasicVSR 等模型。这些模型主要用于底层的视觉任务，如超分辨率，去噪，去雨，修复。MindEditing还支持多种平台，包括CPU/GPU/Ascend。当然，在Ascend设备上你会得到更优的体验。

<div align="center">
  <img src="./docs/image/display_tasks.png"/>
</div>

一些演示:

- **视频超分演示**

<video src="./docs/video/Video_SR_Demo.mp4">
</video>

- **视频插帧演示**

<video src="./docs/video/Video_frame_Interpolation_Demo_IFRNet.mp4">
</video>

<details open>
<summary>主要特性</summary>

- **易于使用**

  我们采用统一的入口，您只需指定支持的模型名称并在参数yaml文件中配置参数即可启动任务。
- **支持多种任务**

  MindEditing支持多种当前流行的底层视觉任务如 ：*去模糊*, *去噪音*, *超分辨率*, 和*修复*.
- **SOTA**

  MindEditing 在 *去模糊*, *去噪音*, *超分辨率*, 和 *修复* 任务中提供最先进的算法。

</details>

## 多任务

有这么多的任务，是否有这样一个模型可以处理多个任务？当然，预训练模型，即图像处理transformer([IPT](docs/ipt_CN.md))。IPT模型是一种新的预训练模型，它是在多头多尾图像上进行训练的。 此外，还引入了对比学习，以适应不同的图像处理任务。 因此，经过优化后的预训练模型可以有效地用于预期的任务。由于只有一个预训练的模型，IPT在各种底层基准测试中优于当前最先进的方法。

<details open>
<summary>优秀的性能</summary>

- **与目前最先进的图像处理模型相比，IPT模型在不同任务下的表现更好**

<div align="center">
  <img src="./docs/image/performance_on_different_tasks.png"/>
</div>

- **刷榜多个底层视觉任务**

  *与目前最先进的方法相比，IPT模型取得了最好的性能。*

<div align="center">
  <img src="./docs/image/quantitative_results.png"/>
</div>

- **泛化能力**

  *IPT模型对不同噪声水平彩色图像去噪的生成能力(table 4)。*
- **CNN和IPT模型使用不同百分比数据的性能**

  *在预训练数据有限的情况下，CNN模型可以获得更好的性能。随着数据量的增加，基于Transformer模块的IPT模型获得了显著的性能提升，从曲线(table 5)趋势也可以看出IPT模型很有潜力。*

<div align="center">
  <img src="./docs/image/other_performance.png"/>
</div>

- **惊人的实际图像推理结果**

  - 图像超分辨率任务

  *下图显示了来自Urban100数据集的双三次下采样(×4)的超分辨率结果。提出的IPT模型恢复了更多的细节。*

  <div align="center">
      <img src="./docs/image/IPT_SR_Task.png"/>
    </div>

  - 图像去噪任务

  必须指出的是，**IPT获得了CVPR2023 NTIRE图像去噪赛道冠军。**

  *下图为噪声级σ = 50时彩色图像去噪结果。*

  <div align="center">
      <img src="./docs/image/IPT_Denoising_task.png"/>
    </div>

  - 图像去雨任务

  *下图显示了Rain100L数据集上的图像去雨结果。*

  <div align="center">
      <img src="./docs/image/IPT_Deraining_task.png"/>
    </div>

</details>

## 依赖

- mindspore >=1.9
- numpy =1.19.5
- scikit-image =0.19.3
- pyyaml =5.1
- pillow =9.3.0
- lmdb =1.3.0
- h5py =3.7.0
- imageio =2.25.1
- munch =2.5.0

Python可以通过Conda安装。

安装 Miniconda:

```shell
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

创建一个虚拟环境，以Python 3.7.5为例:

```shell
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

检查Python版本。

```shell
python --version
```

要安装依赖项，请运行：

```shell
pip install -r requirements.txt
```

MindSpore(>=1.9)可以通过遵循官方[说明](https://www.mindspore.cn/install)轻松安装，在那里您可以选择您最适合的硬件平台。要在分布式模式下运行，需要安装[openmpi](https://www.open-mpi.org/software/ompi/v4.0/)

## 开始

我们提供了训练和验证的启动文件，选择不同的模型配置启动。关于MindEditing的更多基本使用方法请看 [文档](tutorials/document.md) 。

```shell
python3 train.py --config_path ./configs/basicvsr/train.yaml
# or
python3 val.py --config_path ./configs/basicvsr/val.yaml
```

- Graph Mode and Pynative Mode

  图模式使用编译静态图来优化效率和并行计算。相比之下，pynative模式的优势在于灵活性和易于开发。你可以改变模型配置文件中的参数system.context_mode切换到纯pynative模式进行开发。

## 新闻

MindEditing目前分支是0.x，后续将继续推出1.x分支。将发现更多特性在1.x分支中，请继续关注。

---

- 2023年4月6日

基于多平面特征表示的高效视图合成和3d多帧去噪模型(MPFER)即将推出，敬请期待。

- 2023年3月15日

[Tunable Conv](docs/ tunable_convv .md)的推理代码和演示已经作为测试用例加入，您可以在./tests/中找到它们。此外，训练代码很快就会出来。Tunable Conv有4个模型用于演示，NAFNet用于调节图像去噪，SwinIR用于调节图像去噪和感知超分辨率，EDSR用于调节联合图像去噪和去模糊，StyleNet用于调节风格转移。

## 并行性能

增加并行工作数可以加快训练速度，下面是示例模型在CPU 16核 GPU 2xP100上的实验：

```text
num_parallel_workers: 8
epoch 1/100 step 1/133, loss = 0.045729052, duration_time = 00:01:07, step_time_avg = 0.00 secs, eta = 00:00:00
epoch 1/100 step 2/133, loss = 0.027709303, duration_time = 00:01:20, step_time_avg = 6.66 secs, eta = 1 day(s) 00:36:02
epoch 1/100 step 3/133, loss = 0.027135072, duration_time = 00:01:33, step_time_avg = 8.74 secs, eta = 1 day(s) 08:17:56

num_parallel_workers: 16
epoch 1/100 step 1/133, loss = 0.04535071, duration_time = 00:00:47, step_time_avg = 0.00 secs, eta = 00:00:00
epoch 1/100 step 2/133, loss = 0.032363698, duration_time = 00:01:00, step_time_avg = 6.74 secs, eta = 1 day(s) 00:54:38
epoch 1/100 step 3/133, loss = 0.02718924, duration_time = 00:01:13, step_time_avg = 8.83 secs, eta = 1 day(s) 08:36:07
```

## 教程

以下教程将帮助用户学习使用MindEditing。

- [文档](tutorials/document_CN.md)

## 模型列表

| 模型名                                              | 任务                       | 会议                                                                                                                                                                     | 支持平台   |
| --------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| [IPT](docs/ipt_CN.md)                                  | 多任务                     | [CVPR 2021](https://arxiv.org/abs/2012.00364)                                                                                                                               | Ascend/GPU |
| [BasicVSR](docs/basicvsr_CN.md)                        | 视频超分辨率               | [CVPR 2021](https://arxiv.org/abs/2012.02181)                                                                                                                               | Ascend/GPU |
| [BasicVSR++Light](docs/basicvsr_plus_plus_light_CN.md) | 视频超分辨率               | [CVPR 2022](https://arxiv.org/abs/2104.13371)                                                                                                                               | Ascend/GPU |
| [NOAHTCV](docs/noahtcv_CN.md)                          | 图像去噪                   | [CVPR 2021(MAI Challenge)](https://arxiv.org/pdf/2105.08629.pdf)                                                                                                            | Ascend/GPU |
| [RRDB](docs/rrdb_CN.md)                                | 图像超分辨率               |                                                                                                                                                                          | Ascend/GPU |
| [FSRCNN](docs/fsrcnn_CN.md)                            | 图像超分辨率               | [ECCV 2016](https://arxiv.org/pdf/1608.00367.pdf)                                                                                                                           | Ascend/GPU |
| [SRDiff](docs/srdiff_CN.md)                            | 图像超分辨率               | [Neurocomputing 2022](https://arxiv.org/abs/2104.14951)                                                                                                                     | Ascend/GPU |
| [VRT](docs/vrt_CN.md)                                  | 多任务                     | [arXiv(2022.01)](https://arxiv.org/abs/2201.12288)                                                                                                                          | Ascend/GPU |
| [RVRT](docs/rvrt_CN.md)                                | 多任务                     | [arXiv(2022.06)](https://arxiv.org/abs/2206.02146)                                                                                                                          | Ascend/GPU |
| [TTVSR](docs/ttvsr_CN.md)                              | 视频超分辨率               | [CVPR 2022](https://arxiv.org/abs/2204.04216)                                                                                                                               | Ascend/GPU |
| [MIMO-Unet](docs/mimo_unet_CN.md)                      | 图像去模糊                 | [ICCV 2021](https://arxiv.org/abs/2108.05054)                                                                                                                               | Ascend/GPU |
| [NAFNet](docs/nafnet_CN.md)                            | 图像去模糊                 | [arXiv(2022.04)](https://arxiv.org/abs/2204.04676)                                                                                                                          | Ascend/GPU |
| [CTSDG](docs/ctsdg_CN.md)                              | 图像修复                   | [ICCV 2021](https://arxiv.org/pdf/2108.09760.pdf)                                                                                                                           | Ascend/GPU |
| [EMVD](docs/emvd_CN.md)                                | 视频去噪                   | [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Maggioni_Efficient_Multi-Stage_Video_Denoising_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.pdf) | Ascend/GPU |
| [Tunable_Conv](docs/tunable_conv_CN.md)                | 可调任务(图像处理)         | [arXiv(2023.04)](https://arxiv.org/abs/2304.00898v1)                                                                                                                        | Ascend/GPU |
| [IFR+](docs/ifr_plus_CN.md)                            | 视频帧插值                 | [CVPR 2022](https://arxiv.org/abs/2205.14620)                                                                                                                               | Ascend/GPU |
| [MPFER](docs/mpfer_CN.md)                              | 基于3d的多帧去噪(即将推出) | [arXiv(2023.04)](https://arxiv.org/pdf/2303.18139.pdf)                                                                                                                      | GPU        |

关于模型的更多信息请看 [ModelZoo Homepage](https://gitee.com/mindspore/models) 或者 `docs`文件夹下的模型文档。

## 许可证

本项目遵循 [Apache License 2.0](LICENSE.md) 开源许可证。

## 反馈与联系

动态版本仍在开发中，如果您发现任何问题或对新功能有任何想法，请通过 [issue](https://github.com/mindspore-lab/mindediting/issues)与我们联系。

## 感谢

MindSpore 是一个开源项目，欢迎任何贡献和反馈。我们希望这个工具箱和基准能够通过提供一个灵活且标准化的工具包来重新实现现有的方法，并开发他们自己的新的计算机视觉方法，从而为不断增长的研究社区服务。

 如果你发现 *MindEditing* 对你的研究有用，请考虑引用以下相关论文:

```
@misc{MindEditing 2022,
    title={{MindEditing}:MindEditing for low-level vision task},
    author={MindEditing},
    howpublished = {\url{https://github.com/mindspore-lab/mindediting}},
    year={2022}
}

```

## MindSpore-Lab 的其他项目

- [MindCV](https://github.com/mindspore-lab/mindcv): 一个基于MindSpore的视觉模型和算法工具箱。
- [MindNLP](https://github.com/mindspore-lab/mindnlp): 一个基于MindSpore的开源NLP库。
- [MindDiffusion](https://github.com/mindspore-lab/minddiffusion):基于MindSpore的扩散模型集合。
- [MindFace](https://github.com/mindspore-lab/mindface): MindFace是一个基于MindSpore的开源工具包，包含了最先进的人脸识别和检测模型，如ArcFace、RetinaFace等模型。
- [MindAudio](https://github.com/mindspore-lab/mindaudio): 一个基于MindSpore的语音领域的开源一体化工具包。
- [MindOCR](https://github.com/mindspore-lab/mindocr): 一个基于MindSpore的OCR模型、算法和管道工具箱。
- [MindRL](https://github.com/mindspore-lab/mindrl): 高性能、可扩展的MindSpore强化学习框架。
- [MindREC](https://github.com/mindspore-lab/mindrec): MindSpore大规模推荐系统库。
- [MindPose](https://github.com/mindspore-lab/mindpose): 一个基于MindSpore的姿势估计开源工具箱。

---
