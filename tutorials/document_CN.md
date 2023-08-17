## 开始

### 自动安装

下面以Ubuntu 18.04为例，介绍在有GPU环境的Linux系统下，通过pip安装MindSpore的方法。如果您想手动安装或了解更多信息，请参阅官方 [文档](https://www.mindspore.cn/install) 。

在使用自动安装脚本之前，需要确保系统正确安装了NVIDIA GPU驱动。CUDA 10.1要求最低显卡驱动版本为418.39；CUDA 11.1要求最低显卡驱动版本为450.80.02。执行以下指令检查驱动版本。

```text
nvidia-smi
```

如果未安装GPU驱动，执行如下命令安装。

```text
sudo apt-get update
sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

安装完成后，请重新启动系统。

自动安装脚本需要更改软件源配置以及通过APT安装依赖，所以需要申请root权限。使用以下命令获取自动安装脚本并执行。自动安装脚本仅支持安装MindSpore>=1.6.0。

```text
wget https://gitee.com/mindspore/mindspore/raw/r2.0.0-alpha/scripts/install/ubuntu-gpu-pip.sh
# install MindSpore 2.0.0-alpha, Python 3.7 and CUDA 11.1
MINDSPORE_VERSION=2.0.0a0 bash -i ./ubuntu-gpu-pip.sh
# to specify Python and MindSpore version, taking Python 3.9 and MindSpore 1.6.0 as examples, use the following manners
# PYTHON_VERSION=3.9 CUDA_VERSION=10.1 MINDSPORE_VERSION=1.6.0 bash -i ./ubuntu-gpu-pip.sh
```

该脚本会执行以下操作：

- 更改软件源配置为华为云源。

- 安装MindSpore所需的依赖，如GCC。

- 通过APT安装Python3和pip3，并设为默认。

- 下载CUDA和cuDNN并安装。

- 通过pip安装MindSpore GPU版本。

- 如果OPENMPI设置为on，则安装Open MPI。

自动安装脚本执行完成后，需要重新打开终端窗口以使环境变量生效。





## 基本用法

### 使用预训练模型进行评估

目前，我们提供`val.py`脚本来评估完整数据集上的预训练模型。您可以使用以下命令来评估预训练的模型。

```shell
python val.py [--config_path ${CONFIG_FILE_PATH}] [--save_metrics ${SAVE_METRICS_RESULTS_PATH}]
```

例如,

```shell
python val.py --config_path ./configs/basicvsr/val.yaml --save_metrics ./output/quality_metrics.json
```

> 注意:训练的参数文件`ckpt`路径需要在相应的模型配置文件中设置，参数设置可参考 [learn about Configs](#learning-about-model-configuration). 如果你想查看推理输出图像，你可以在yaml文件`callback `中设置`img_save_directory `来保存输出图像的路径。

### 训练模型

一旦在yaml配置文件中设置了参数，就可以按照以下命令启动培训任务。有关参数设置的详细信息，请参考 [learn about Configs](#learning-about-model-configuration)。

- 单GPU训练

```shell
python train.py [--config_path ${CONFIG_FILE_PATH}]
```

- 例如,

```shell
python train.py --config_path ./configs/basicvsr/train.yaml
```

- 分布式训练

```shell
mpirun [-n ${DEVICE_NUMBER}] python train.py [--config_path ${CONFIG_FILE_PATH}]
```

- 例如,
```shell
mpirun -n 2 python train.py --config_path ./configs/basicvsr/train.yaml
```

- 在ModelArts平台上进行训练

在 [ModelArts](https://www.huaweicloud.com/intl/en-us/product/modelarts.html) 云平台上运行训练:

```text
1.登录到ModelArts管理控制台。
2.在导航区中选择“训练管理>训练作业”。进入训练任务列表。
3.单击“创建训练任务”。然后配置参数
```





## 了解模型配置

`mindediting` 提供模型的yaml文件来配置参数。下面通过配置实例说明如何配置相应的参数。

### 设置环境

1. 参数说明

- context_mode: 使用静态图模式或动态图模式。

2. yaml文件样例

```text
system:
  context_mode: "graph_mode"
  ...
```

3. Parse参数设置

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. 相应的代码示例

```python
...
path_args, _ = parser.parse_known_args()
default, helper, choices = parse_yaml(path_args.config_path)
...
...
def train(cfg):
    init_env(cfg)
    ...
def init_env(cfg):
    ...
    mode = cfg.system.context_mode.lower()
    context_mode = None
    if mode == "pynative_mode":
        context_mode = context.PYNATIVE_MODE
    elif mode == "graph_mode":
        context_mode = context.GRAPH_MODE
    context.set_context(mode=context_mode,
                        device_target=cfg.system.device_target)
    ...
```

### 数据集

1. 参数说明

- dataset_name: 数据集名称。

- input_path: 数据集文件路径。

- train_annotation: 训练样本。

- test_annotation: 测试样本。

- gt_subdir: 超分辨率图片。

- lr_subdir: 低分辨率图片。

- batch_size: 每个批次包含的图像数目。

- epoch_size: 训练总的轮次。

- num_frames: 图像帧数。

- num_parallel_workers: 读取数据的工作线程数。


2. yaml文件样例

```text
dataset:
  dataset_name: "vimeo_super_resolution"
  input_path: "./vimeo_super_resolution"
  train_annotation: 'sep_trainlist.txt'
  test_annotation: 'sep_testlist.txt'
  gt_subdir: 'sequences'
  lr_subdir: 'BIx4'
  batch_size: 8
  epoch_size: 37
  num_frames: 7
  num_parallel_workers: 8
  ...
```

3. Parse参数设置

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. 相应的代码示例

```python
def train(cfg):
    ...
    rank_id = get_device_id()
    group_size = get_device_num()
    dataset_train = create_dataset(
        name=cfg.dataset.dataset_name, root=cfg.dataset.input_path, split='train',
        shuffle=True, nframes=cfg.dataset.num_frames, num_shards=group_size, shard_id=rank_id, **cfg.dataset.cfg_dict)

    train_operations, train_input_columns, train_output_columns = create_transform(model_name=cfg.model.name,
                                                                                   split='train',
                                                                                   pipeline=cfg.train_pipeline,
                                                                                   **cfg.dataset.cfg_dict)
    loader_train = create_loader(
        dataset=dataset_train,
        batch_size=cfg.dataset.batch_size,
        operations=train_operations,
        input_columns=train_input_columns,
        output_columns=train_output_columns,
        split='train',
    )
    ...
```

### 模型

1. 参数说明

- name: 模型名。

- load_path: 模型参数文件所在路径。

- need_val: 训练时是否验证评估。

- spynet_load_path: spynet参数文件所在路径。


2. yaml文件样例

```text
model:
  name: "basicvsr"
  load_path: "./basicvsr.ckpt"
  spynet_load_path: "./spynet_20210409-c6c1bd09.ckpt"
train_params:
  need_val: True
```

3. Parse参数设置

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. 相应的代码示例

```python
def train(cfg):
    ...
    # create model
    cfg.mode = 'train'
    net, eval_network= create_model_by_name(model_name=cfg.model.name, cfg=cfg)
    ...
```

### 损失函数

1. 参数说明

- name: 损失函数名简称。

- reduction: 对输出应用特定的计算方法。均值或求和。

- weight:  权重。应用于每个批处理数据损失的缩放权重。

2. yaml文件样例

```text
loss:
  name: "CharbonnierLoss"
  reduction:'mean'
  weight: 1.0
  ...
```

3. Parse参数设置

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. 相应的代码示例

```python
def train(cfg):
    ...
    loss = create_loss(loss_name=cfg.loss.name, **cfg.loss.cfg_dict)
    ...
```

### 优化器

1. 参数说明

- name: 优化器名称。

- learning_rate: 浮点数或学习策略。支持固定学习率和动态学习率。

- weight_decay:  权重衰减因子。

2. yaml文件样例

```text
optimizer:
  name: "Adam"
  learning_rate: 1.0e-3
  weight_decay: 0.0
  ...
```

3. Parse参数设置

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. 相应的代码示例

```python
def train(cfg):
    ...
    loss_scale = 1.0 if cfg.loss.amp_level == "O0" else cfg.loss.loss_scale
    optimizer = create_optimizer(params=optimizer_params, lr=learning_rate, 						                                                              opt=cfg.optimizer.name, loss_scale=loss_scale,
                                 **{'beta1': cfg.optimizer.beta1,
                                    'beta2': cfg.optimizer.beta2})
    ...
```

### 学习率策略

1. 参数说明

- name:  学习率策略名称。

- base_lr: 学习率值。

- min_lr:  cosine_decay 策略的学习率最小值。

- warmup_epochs: 如果学习策略支持，epochs 预热学习率。

2. yaml文件样例

```text
scheduler:
  name: "cosine_annealing"
  base_lr: 2.0e-4
  min_lr: 1.0e-7
  warmup_epochs: 0
  ...
```

3. Parse参数设置

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. 相应的代码示例

```python
def train(cfg):
    ...
    optimizer_params, learning_rate = create_scheduler_by_name(model_name=cfg.model.name, cfg=cfg)
    ...
```

### 评估指标
1. 参数说明

- PSNR: 峰值信噪比

   - reduction: 计算方法。

   - crop_border:  在图像的每个边缘裁剪像素。这些像素不涉及PSNR计算。

   - input_order:  输入顺序是“HWC”还是“CHW”。

   - convert_to:  是否将图像转换为其他颜色模型。如果为None，则不改变图像。

- SSIM: 结构相似性

   - reduction:  计算方法。

   - crop_border:  在图像的每个边缘裁剪像素。这些像素不涉及PSNR计算。

   - input_order:  输入顺序是“HWC”还是“CHW”。

   - convert_to:  是否将图像转换为其他颜色模型。如果为None，则不改变图像。

2. yaml文件样例
```text
metric:
  PSNR:
    reduction: 'avg'
    crop_border: 0
    input_order: 'CHW'
    convert_to: 'y'
    ...
  SSIM:
    reduction: 'avg'
    crop_border: 0
    input_order: 'CHW'
    convert_to: 'y'
    ...
```

3. Parse参数设置

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. 相应的代码示例

```python
def train(cfg):
    ...
    metrics = create_metrics(cfg.metric)
    ...
```

### 回调
1. 参数说明

- keep_checkpoint_max:  可保存的参数文件最大数量。

- save_epoch_frq:  间隔设置的轮次保存输出参数文件。

- eval_frequency:  间隔设置的轮次验证指标。

- workers_num:  线程工作数。

- print_frequency:  输出打印频率。

- ckpt_save_dir:  训练输出参数文件保存路径。

2. yaml文件样例
```text
callback:
  keep_checkpoint_max: 60
  save_epoch_frq: 50
  eval_frequency: 10
  workers_num: 2
  print_frequency: 10
  ckpt_save_dir: "./ckpt/basicvsr"
```

3. Parse参数设置

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. 相应的代码示例

```python
def train(cfg):
    ...
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.train_params.save_epoch_frq,
                                 keep_checkpoint_max=cfg.train_params.keep_checkpoint_max)
    ...
    if cfg.train_params.need_val:
        callbacks.append(EvalAsInValPyCallBack(cfg, eval_network or net))
    ...
```
