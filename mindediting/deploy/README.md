# Deployment tool

You can run inference of models using different backends. Currently, only Ascend backend is supported.
To proceed, follow next steps:

## 1. Create "om" file from the Mindspore checkpoint. For it, ASCEND_TOOLKIT_HOME environment variable should be set:
```
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/<ascend_toolkit_version>
```
By default, it's set to /usr/local/Ascend/ascend-toolkit/latest.

Then run the script:
```
python export.py --config_path <path_to_models_config_file> --target_format om --shape <N, T, C, H, W> --output_name <name_of_file> --output_type <UINT8/FP16/FP32> --fusion_switch_file mindediting/deploy/utils/fusion_switch.cfg
```
**Important:** Do not forget to set fusion file, it highly impacts the performance.

Finally, you'll get a file <name_of_file>.om

## 2. Prepare the configuration file for deployment:
You can run different tasks in a pipeline, for that create own section in the configuration file.

task_name: <name of task>
task_type: <image/video>
input_file: <input file / folder>
output_file: <output file>
windows_size: <portion of frames to read, better to set to 1>
fixed_width: <width to interpolate the original resolution before the inference, better to use original width>
fixed_height: <height to interpolate the original resolution before the inference, better to use original height>
up_scale: <scale factor>
color_space: <rgb/bgr>
tasks:
  - task_name: <task name>
    task_type: super_resolution
    data_io: <basic/ipt_sr/srdiff_sr/rgb_to_bgr/bgr_to_rgb>
    backend: ascend
    dtype: <float16/float32>
    model_file: "<path_to_model>.om"
    once_process_frames: <number of frames to run inference on, may be all video size>
    frame_overlap: <number of frames to be overlapped>
    patch_overlap: <spatial overlap>
    up_scale: <scale factor>
    device_id: <device id>
    tiling: <default/simple_image/None>
  - task_name: ...
    ...

The examples of task configuraton files are placed at `LLVT/configs` directory.

## 3. Add path to mindediting package to the PYTHONPATH environment variable:
```
export PYTHONPATH={path_to_LLVT}:$PYTHONPATH
```

And run the script using the created configuration file:
```
python -m mindediting.deploy.deploy -pf <config.yaml>
```

## 4. You can also validate the "om" model and get metrics using `val.py` script.
For that, set the following lines to the model configuration file:
```
model:
  name: "om"
  load_path: <path to "om" model>
validator:
  name: "tiling" or "om_default"
  temporal_overlap: <>
  spatial_overlap: <>
  temporal_size: <>
  spatial_size: <>
  scale: <>
  input_tensor_type: numpy
  dtype: <>
```
Then run:
```
python val.py --config_path <configuration file>
```
