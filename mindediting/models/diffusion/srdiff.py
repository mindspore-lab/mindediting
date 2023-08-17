# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore as ms
import mindspore.nn as nn
from mindspore import get_seed, set_seed

from mindediting.models.common.unet import UNet
from mindediting.models.image_super_resolution.rrdb import RRDBNet
from mindediting.utils.init_weights import init_weights


def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(t, axis=-1)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, repeat=False):
    if repeat:
        c, h, w = shape[1:]
        return ms.numpy.randn((1, c, h, w)).repeat(shape[0], axis=0)
    else:
        return ms.numpy.randn(shape)


def get_beta_schedule(num_diffusion_timesteps, beta_schedule="linear", beta_start=0.0001, beta_end=0.02):
    if beta_schedule == "quad":
        betas = ms.numpy.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=ms.float32) ** 2
    elif beta_schedule == "linear":
        betas = ms.numpy.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=ms.float32)
    elif beta_schedule == "const":
        betas = beta_end * ms.numpy.ones(num_diffusion_timesteps, dtype=ms.float32)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / ms.numpy.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=ms.float32)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = ms.numpy.linspace(0, steps, steps)
    alphas_cumprod = ms.numpy.cos(((x / steps) + s) / (1 + s) * ms.numpy.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clip(0, 0.999)


class SRDiff(nn.Cell):
    def __init__(
        self,
        encoder="rrdb",
        encoder_in_ch=3,
        encoder_out_ch=3,
        encoder_mid_ch=32,
        encoder_num_blocks=8,
        encoder_int_ch=16,
        hidden_size=64,
        dim_mults=(1, 2, 4, 8),
        scale=4,
        bias=True,
        timesteps=100,
        loss_type="l1",
        beta_schedule="linear",
        beta_s=0.008,
        beta_end=0.02,
        res=True,
        res_rescale=2.0,
        up_input=False,
        encoder_load_path="",
        input_shape=None,
    ):
        super().__init__()
        self.scale = scale
        self.res = res
        self.res_rescale = res_rescale
        self.denoise_fn = UNet(
            hidden_size,
            out_dim=3,
            dim_mults=dim_mults,
            cond_dim=encoder_mid_ch,
            num_block=encoder_num_blocks,
            scale=scale,
            bias=bias,
            res=res,
            up_input=up_input,
        )
        init_weights(self.denoise_fn, init_type="he")
        if encoder == "rrdb":
            self.encoder = RRDBNet(
                encoder_in_ch,
                encoder_out_ch,
                encoder_mid_ch,
                encoder_num_blocks,
                scale,
                internal_ch=encoder_int_ch,
                bias=bias,
                get_feat=True,
                scale_to_0_1=True,
            )
        else:
            raise NotImplementedError(f"Supported only RRDB encoder, but got {encoder}")
        if encoder_load_path:
            loaded_param_dict = ms.load_checkpoint(encoder_load_path)
            loaded_param_dict_ = {}
            for name, param in loaded_param_dict.items():
                loaded_param_dict_["encoder." + name] = param
            ms.load_param_into_net(self.encoder, loaded_param_dict_, strict_load=True)

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps, s=beta_s)
        elif beta_schedule in ["linear", "constant", "jsd", "quad"]:
            betas = get_beta_schedule(timesteps, beta_schedule, beta_end=beta_end)
            if res:
                betas[-1] = 0.999
        else:
            raise NotImplementedError(
                f"'beta_schedule' must be one of [cosine, linear, constant, jsd, quad], " f"but got {beta_schedule}"
            )

        self.alphas = 1.0 - betas
        self.alphas_cumprod = ms.numpy.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = ms.ops.concat([ms.ops.ones((1,), type=ms.float32), self.alphas_cumprod[:-1]])

        timesteps = betas.shape
        self.num_timesteps = int(timesteps[0])
        if loss_type.lower().startswith("l1"):
            self.loss_func = nn.L1Loss()
        elif loss_type.lower().startswith("l2"):
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError(f"Supported only l1 and l2 loss functions, but got {self.loss_type}")

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = ms.numpy.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = ms.numpy.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = ms.numpy.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = ms.numpy.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = ms.numpy.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = ms.numpy.log(ms.numpy.maximum(self.posterior_variance, 1e-20))
        self.posterior_mean_coef1 = betas * ms.numpy.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * ms.numpy.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        if input_shape is not None:
            self.predefined_noise = True
            b, c, h, w = input_shape
            h1, w1 = h * scale, w * scale
            self.initial_noise = ms.numpy.randn((b, c, h1, w1))
            self.step_noise = []
            for _ in range(self.num_timesteps):
                current_seed = get_seed()
                if current_seed is not None:
                    set_seed(current_seed + 1)
                self.step_noise.append(noise_like((b, c, h1, w1), repeat=False))
        else:
            self.predefined_noise = False

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, noise_pred, clip_denoised):
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clip(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def construct(self, *inputs):
        if len(inputs) == 4:
            img_lr, img_hr, img_lr_up, t = inputs
        elif len(inputs) == 3:
            img_lr, img_hr, img_lr_up = inputs
            t = None
        elif len(inputs) == 2:
            return self.sample(*inputs)
        else:
            raise ValueError(f"Number of model's inputs must be 2, 3 or 4 but got {len(inputs)}")
        b = img_hr.shape[0]
        t = ms.numpy.randint(0, self.num_timesteps, (b,)) if t is None else ms.Tensor([t], dtype=ms.int32).repeat(b)
        cond = self.encoder(img_lr)

        x = self.img2res(img_hr, img_lr_up)
        loss = self.p_losses(x, t, cond, img_lr_up)
        return loss

    def p_losses(self, x_start, t, cond, img_lr_up, noise=None):
        noise = noise if noise is not None else ms.numpy.randn(x_start.shape)
        x_tp1_gt = self.q_sample(x_start, t, noise)
        noise_pred = self.denoise_fn(x_tp1_gt, t, cond, img_lr_up)
        loss = self.loss_func(noise, noise_pred)
        return loss

    def q_sample(self, x_start, t, noise=None):
        noise = noise if noise is not None else ms.numpy.randn(x_start.shape)
        t_cond = (t[:, None, None, None] >= 0).float()
        t = t.clip(0, None)
        res = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return res * t_cond + x_start * (1 - t_cond)

    def p_sample(self, x, t, cond, img_lr_up, timestep, noise_pred=None, clip_denoised=True, repeat_noise=False):
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond, img_lr_up=img_lr_up)
        b = x.shape[0]
        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
            x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, repeat_noise) if not self.predefined_noise else self.step_noise[timestep]
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred

    def sample(self, img_lr, img_lr_up):
        b = img_lr.shape[0]
        if not self.res:
            t = ms.numpy.full((b,), self.num_timesteps - 1)
            img = self.q_sample(img_lr_up, t)
        else:
            img = ms.numpy.randn(img_lr_up.shape) if not self.predefined_noise else self.initial_noise
        cond = self.encoder(img_lr)
        for i in range(self.num_timesteps - 1, -1, -1):
            img, x_recon = self.p_sample(img, ms.numpy.full((b,), i), cond, img_lr_up, i)
        img = self.res2img(img, img_lr_up)
        return (img + 1) / 2

    def res2img(self, img_, img_lr_up, clip_input=True):
        if self.res:
            if clip_input:
                img_ = img_.clip(-1, 1)
            img_ = img_ / self.res_rescale + img_lr_up
        return img_

    def img2res(self, x, img_lr_up, clip_input=True):
        if self.res:
            x = (x - img_lr_up) * self.res_rescale
            if clip_input:
                x = x.clip(-1, 1)
        return x
