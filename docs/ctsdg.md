# Contents <!-- omit in toc -->

- [CTSDG description](#ctsdg-description)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment requirements](#environment-requirements)
- [Performance](#performance)
  - [Training Performance](#training-performance)
  - [Evaluation Performance](#evaluation-performance)

# [CTSDG description](#contents)

Deep generative approaches have recently made considerable progress in image inpainting by introducing
structure priors. Due to the lack of proper interaction with image texture during structure reconstruction, however,
current solutions are incompetent in handling the cases with large corruptions, and they generally suffer from distorted
results. This is a novel two-stream network for image inpainting, which models the structure constrained texture
synthesis and texture-guided structure reconstruction in a coupled manner so that they better leverage each other
for more plausible generation. Furthermore, to enhance the global consistency, a Bi-directional Gated Feature Fusion (
Bi-GFF)
module is designed to exchange and combine the structure and texture information and a Contextual Feature Aggregation (
CFA)
module is developed to refine the generated contents by region affinity learning and multiscale feature aggregation.

> [Paper](https://arxiv.org/pdf/2108.09760.pdf):  Image Inpainting via Conditional Texture and Structure Dual Generation
> Xiefan Guo, Hongyu Yang, Di Huang, 2021.
> [Supplementary materials](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Guo_Image_Inpainting_via_ICCV_2021_supplemental.pdf)

# [Model architecture](#contents)

CTSDG follows the Generative Adversarial Network (GAN) framework.

*Generator*. Image inpainting is cast into two subtasks, i.e. structure-constrained texture synthesis (left, blue) and
texture-guided structure reconstruction (right, red), and the two parallel-coupled streams borrow encoded deep features
from each other. The Bi-GFF module and CFA module are stacked at the end of the generator to further refine the results.

*Discriminator*. The texture branch estimates the generated texture, while the structure branch guides structure
reconstruction.

![ctsdg.png](image/ctsdg.png)

# [Dataset](#contents)

Dataset
used: [CELEBA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting)

- From **CELEBA** you need to download (section *Downloads -> Align&Cropped Images*):
    - `img_align_celeba.zip`
    - `list_eval_partitions.txt`
- From **NVIDIA Irregular Mask Dataset** you need to download:
    - `irregular_mask.zip`
    - `test_mask.zip`
- The directory structure is as follows:

  ```text
    .
    ├── img_align_celeba            # images folder
    ├── irregular_mask              # masks for training
    │   └── disocclusion_img_mask
    ├── mask                        # masks for testing
    │   └── testing_mask_dataset
    └── list_eval_partition.txt     # train/val/test splits
  ```

# [Environment requirements](#contents)

- Hardware（GPU/Ascend）
    - Prepare hardware environment with GPU or Ascend 910 processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# [Performance](#contents)

## [Training Performance](#contents)

| Parameters                 | CTSDG (1xGPU)                                          | CTSDG (1xNPU)                                         |
|----------------------------|--------------------------------------------------------|-------------------------------------------------------|
| Model Version              | CTSDG                                                  | CTSDG                                                 |
| Resources                  | 1x Nvidia V100                                         | 1x Ascend 910A                                        |
| Uploaded Date              | 12 / 19 / 2022 (month/day/year)                        | 12 / 19 / 2022 (month/day/year)                       |
| MindSpore Version          | 2.0.0-alpha                                            | 2.0.0-alpha                                           |
| Dataset                    | CELEBA, NVIDIA Irregular Mask Dataset                  | CELEBA, NVIDIA Irregular Mask Dataset                 |
| Training Parameters        | batch_size=6, train_iter=350000, finetune_iter=150000  | batch_size=6, train_iter=350000, finetune_iter=150000 |
| Optimizer                  | Adam                                                   | Adam                                                  |
| Speed                      | 590 ms/step                                            | 230 ms/step                                           |
| Total time                 | 3d 10h 0m                                              | 1d 8h 0m                                              |
| Checkpoint for Fine tuning | 200 MB (.ckpt file)                                    | 200 MB (.ckpt file)                                   |


## [Evaluation Performance](#contents)

| Parameters               | CTSDG (1xNPU)                         |
|--------------------------|---------------------------------------|
| Model Version            | CTSDG                                 |
| Resources                | 1x Ascend 910A                        |
| Uploaded Date            | 12 / 19 / 2022 (month/day/year)       |
| MindSpore Version        | 2.0.0-alpha                           |
| Datasets                 | CELEBA, NVIDIA Irregular Mask Dataset |
| Batch_size               | 1                                     |
| Inference speed, s       | 0.029                                 |
| PSNR ( 0-20% corruption) | 37.59                                 |
| PSNR (20-40% corruption) | 29.08                                 |
| PSNR (40-60% corruption) | 23.97                                 |
| PSNR (overall)           | 31.47                                 |
| SSIM ( 0-20% corruption) | 0.977                                 |
| SSIM (20-40% corruption) | 0.917                                 |
| SSIM (40-60% corruption) | 0.822                                 |
| SSIM (overall)           | 0.923                                 |
