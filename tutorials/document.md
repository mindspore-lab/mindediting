## Get Started

### Automatic Installation

The following takes Ubuntu 18.04 as an example to describe how to install MindSpore by pip in a Linux system with a GPU environment. See the official [documentation](https://www.mindspore.cn/install) if you want to install it manually or for more information.

Before using the automatic installation script, you need to make sure that the NVIDIA GPU driver is correctly installed on the system. The minimum required GPU driver version of CUDA 10.1 is 418.39. The minimum required GPU driver version of CUDA 11.1 is 450.80.02. The minimum required GPU driver version of CUDA 11.1 is 450.80.02. Execute the following command to check the driver version.

```text
nvidia-smi
```

If the GPU driver is not installed, run the following command to install it.

```text
sudo apt-get update
sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

After the installation is complete, please reboot your system.

The root permission is required because the automatic installation script needs to change the software source configuration and install dependencies via APT. Run the following command to obtain and run the automatic installation script. The automatic installation script only supports the installation of MindSpore>=1.6.0.

```text
wget https://gitee.com/mindspore/mindspore/raw/r2.0.0-alpha/scripts/install/ubuntu-gpu-pip.sh
# install MindSpore 2.0.0-alpha, Python 3.7 and CUDA 11.1
MINDSPORE_VERSION=2.0.0a0 bash -i ./ubuntu-gpu-pip.sh
# to specify Python and MindSpore version, taking Python 3.9 and MindSpore 1.6.0 as examples, use the following manners
# PYTHON_VERSION=3.9 CUDA_VERSION=10.1 MINDSPORE_VERSION=1.6.0 bash -i ./ubuntu-gpu-pip.sh
```

This script performs the following operations:

- Change the software source configuration to a HUAWEI CLOUD source.

- Install the dependencies required by MindSpore, such as GCC.

- Install Python3 and pip3 via APT and set them as default.

- Download and install CUDA and cuDNN.

- Install MindSpore GPU by pip.

- Install Open MPI if OPENMPI is set to on.

After the automatic installation script is executed, you need to reopen the terminal window to make the environment variables take effect.





## Basic Usage

### Evaluation with Pre-Trained Models

Currently, we provide the `val.py` script to evaluate the pre-training model on a complete data set. you can use the following commands to evaluate a pre-trained model.

```shell
python val.py [--config_path ${CONFIG_FILE_PATH}] [--save_metrics ${SAVE_METRICS_RESULTS_PATH}]
```

for example,

```shell
python val.py --config_path ./configs/basicvsr/val.yaml --save_metrics ./output/quality_metrics.json
```

> Note：The trained parameter file`.ckpt `path needs to be set in the corresponding model configuration file, and the parameter setting can be referred to [learn about Configs](#learning-about-model-configuration). If you want to see the inference output image, you can set `img_save_directory` in the yaml file `callback` to save the output image path.

### Train a model

Once you have set the parameters in the yaml configuration file, you can start the training task by following the commands below. For details about parameter Settings, see [learn about Configs](#learning-about-model-configuration).

- Train with single GPU

```shell
python train.py [--config_path ${CONFIG_FILE_PATH}]
```

- for example,

```shell
python train.py --config_path ./configs/basicvsr/train.yaml
```

- Distributed Training

```shell
mpirun [-n ${DEVICE_NUMBER}] python train.py [--config_path ${CONFIG_FILE_PATH}]
```

- for example，
```shell
mpirun -n 2 python train.py --config_path ./configs/basicvsr/train.yaml
```

- Train on ModelArts Platform

To run training on the [ModelArts](https://www.huaweicloud.com/intl/en-us/product/modelarts.html) cloud platform:

```text
1.Log in to the ModelArts management console.
2.In the navigation pane, choose Training Management > Training Jobs. The training job list is displayed.
3.Click Create Training Job. Then, configure parameters.
```





## Learning about Model Configuration

`mindediting` provide the yaml file of the model to configure parameters. Here is an example to explain how to configure the corresponding parameters.

### Set Context

1. Parameter description

- context_mode: Use graph mode or pynative mode.

2. Sample yaml file

```text
system:
  context_mode: "graph_mode"
  ...
```

3. Parse parameter setting

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. Corresponding code example

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

### Dataset

1. Parameter description

- dataset_name: dataset name.

- input_path: Path of dataset file.

- train_annotation: training sample.

- test_annotation: testing sample.

- gt_subdir: super resolution image.

- lr_subdir: low resolution image.

- batch_size: The number of images each batch.

- epoch_size: train epoch size.

- num_frames: frame number.

- num_parallel_workers: Number of workers(threads) to process the dataset in parallel.


2. Sample yaml file

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

3. Parse parameter setting

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. Corresponding code example

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

### Model

1. Parameter description

- name: model name.

- load_path: initialize master model weights from this checkpoint.

- need_val: whether to validate during training.

- spynet_load_path: initialize spynet weights from this checkpoint.


2. Sample yaml file

```text
model:
  name: "basicvsr"
  load_path: "./basicvsr.ckpt"
  spynet_load_path: "./spynet_20210409-c6c1bd09.ckpt"
train_params:
  need_val: True
```

3. Parse parameter setting

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. Corresponding code example

```python
def train(cfg):
    ...
    # create model
    cfg.mode = 'train'
    net, eval_network= create_model_by_name(model_name=cfg.model.name, cfg=cfg)
    ...
```

### Loss Function

1. Parameter description

- name: name of loss function, bce (BinaryCrossEntropy) or ce (CrossEntropy).

- reduction: apply specific reduction method to the output: 'mean' or 'sum'. Default: 'mean'.

- weight: class weight. Shape [C]. A rescaling weight applied to the loss of each batch element.

2. Sample yaml file

```text
loss:
  name: "CharbonnierLoss"
  reduction:'mean'
  weight: 1.0
  ...
```

3. Parse parameter setting

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. Corresponding code example

```python
def train(cfg):
    ...
    loss = create_loss(loss_name=cfg.loss.name, **cfg.loss.cfg_dict)
    ...
```

### optimizer

1. Parameter description

- name: optimizer name.

- learning_rate: float or lr scheduler. Fixed and dynamic learning rate are supported.

- weight_decay:  weight decay factor.

2. Sample yaml file

```text
optimizer:
  name: "Adam"
  learning_rate: 1.0e-3
  weight_decay: 0.0
  ...
```

3. Parse parameter setting

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. Corresponding code example

```python
def train(cfg):
    ...
    loss_scale = 1.0 if cfg.loss.amp_level == "O0" else cfg.loss.loss_scale
    optimizer = create_optimizer(params=optimizer_params, lr=learning_rate,
                                 opt=cfg.optimizer.name, loss_scale=loss_scale,
                                 **{'beta1': cfg.optimizer.beta1,
                                    'beta2': cfg.optimizer.beta2})
    ...
```

### Learning Rate Scheduler

1. Parameter description

- name: name of scheduler.

- base_lr: learning rate value.

- min_lr: lower lr bound for 'cosine_decay' schedulers.

- warmup_epochs: epochs to warmup LR, if scheduler supports.

2. Sample yaml file

```text
scheduler:
  name: "cosine_annealing"
  base_lr: 2.0e-4
  min_lr: 1.0e-7
  warmup_epochs: 0
  ...
```

3. Parse parameter setting

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. Corresponding code example

```python
def train(cfg):
    ...
    optimizer_params, learning_rate = create_scheduler_by_name(model_name=cfg.model.name, cfg=cfg)
    ...
```

### metric
1. Parameter description

- PSNR: Peak Signal-to-Noise Ratio

   - reduction: Calculation method.

   - crop_border:  Cropped pixels in each edges of an image. These pixels are not involved in the PSNR calculation.

   - input_order:  Whether the input order is 'HWC' or 'CHW'.

   - convert_to:  Whether to convert the images to other color models. If None, the images are not altered.

- SSIM: Structural similarity

   - reduction: Calculation method.

   - crop_border:  Cropped pixels in each edges of an image. These pixels are not involved in the PSNR calculation.

   - input_order:  Whether the input order is 'HWC' or 'CHW'.

   - convert_to:  Whether to convert the images to other color models. If None, the images are not altered.

2. Sample yaml file
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

3. Parse parameter setting

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. Corresponding code example

```python
def train(cfg):
    ...
    metrics = create_metrics(cfg.metric)
    ...
```

### callback
1. Parameter description

- keep_checkpoint_max: Maximum number of checkpoint files can be saved.

- save_epoch_frq: Epochs to save checkpoint.

- eval_frequency: Steps to evaluate checkpoint.

- workers_num: worker number.

- print_frequency: print frequency.

- ckpt_save_dir: Path to save checkpoint.

2. Sample yaml file
```text
callback:
  keep_checkpoint_max: 60
  save_epoch_frq: 50
  eval_frequency: 10
  workers_num: 2
  print_frequency: 10
  ckpt_save_dir: "./ckpt/basicvsr"
```

3. Parse parameter setting

```text
python train.py --config_path ./configs/basicvsr/train.yaml
```

4. Corresponding code example

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
