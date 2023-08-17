## NAFNet for Modulated Image Denoising

- Dataset

  Need to prepare [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) Srgb validation dataset by running.

- Inference Test Case

  Then run inference as

    ```bash
    python test_tunable_nafnet.py
    ```

- Tunable Parameters

  The tunable parameters should always sum to one. In this testcase, we control the denoising strength. For
  example `1.0 0.0` will provide maximum denoising, `0.5 0.5` will provide medium denoising, `0.0 1.0` will provide
  minimum denoising.

  For visualization, you can set flag to save image.
  The inference images will be saved in `output/nafnet`, and each png will contain (from left to right): noisy input,
  predicted output, ground-truth.

## SwinIR for Modulated Image Denoising and Perceptual Super-Resolution

- Dataset

  Need to prepare [Kodak](https://r0k.us/graphics/kodak/) dataset.

- Inference Test Case

  Main interface is very similar as before, just run

    ```bash
    python test_tunable_nafnet.py
    ```
  and by default the denoising experiment with standard deviation 25 will start.

- Tunable Parameters

  As before, you can change tunable parameters, add flag to save image, and also add reference PyTorch implementation.
  In
  addition to that you can also select which experiment to run.

  Where `noise_stddev` will run the denoising experiment (Gaussian noise with standard deviation 15/25/50,
  and `sr_factor`
  the super-resolution experiment (super-resolution factor 4).

  In both experiments, we have two tunable parameters. In the denoising experiment, the parameters control the denoising
  strength. For example `1.0 0.0` will provide maximum denoising, and `0.0 1.0` will provide minimum denoising. In the
  super-resolution experiment the parameters control the perception-distortion tradeoff. For example `1.0 0.0` will
  provide maximum accuracy, and `0.0 1.0` will provide maximum perceptual quality.

## EDSR for Modulated Joint Image Denoising and Deblurring

- Dataset

  Need to prepare [Kodak](https://r0k.us/graphics/kodak/) dataset.

- Inference Test Case

  Main interface is very similar as before, just run

    ```bash
    python test_tunable_edsr.py
    ```

- Tunable Parameters

  As before, you can change tunable parameters, add flag to save image, and also add reference PyTorch implementation.
  In
  addition to that you can also select the noise standard deviation as well as the blur size via the following
  arguments:

  Valid values of `noise_stddev` are in [5, 30], and for `blur_stddev` are in [0,4]. Low `noise_stddev` values
  correspond
  to low amount of noise in the input image, and low `blur_stddev` values correspond to less blur in the input image.

  Note that in this case we again have two tunable parameters: the first parameter control the denoising strength, and
  the
  second the deblurring strength. For example `1.0 0.0` will provide maximum denoising and minimum deblurring,
  and `1.0 1.0` will provide maximum denoising and deblurring.

## StyleNet for Modulated Style Transfer

- Dataset

  Need to prepare [Kodak](https://r0k.us/graphics/kodak/) dataset.

- Inference Test Case

  Main interface is very similar as before, just run

    ```bash
    python test_tunable_edsr.py
    ```

- Tunable Parameters

  As before, you can change tunable parameters using `--params`, add flag to save image `--save_images`. Note that in
  this case we again three tunable parameters controlling the
  influence of three different styles, namely, Mosaic (`1.0 0.0 0.0`), Edtaonisl (`0.0 1.0 0.0`), and
  Kandinsky (`0.0 0.0 1.0`). You can try any combination of parameters, as long as the sum of all parameters is `1.0`;
  for instance `0.0 0.5 0.5` is correct, but `1.0 0.5 0.0` is not.
