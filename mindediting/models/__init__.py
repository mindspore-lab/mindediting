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

import json
import sys
from collections import defaultdict
from pathlib import Path

import mindspore
import mindspore.nn as nn
import numpy as np

from mindediting.loss import create_loss
from mindediting.models.common.vgg16 import VGG16FeatureExtractor
from mindediting.models.common.with_eval_cell_out_fix import (
    WithEvalCellNoLoss,
    WithEvalCellOutFix,
    WithEvalCellOutFix_Ctsdg,
)
from mindediting.models.diffusion.srdiff import SRDiff
from mindediting.models.image_deblur.mimo_unet import MIMOUNet
from mindediting.models.image_deblur.nafnet import NAFNet
from mindediting.models.image_denoise.noahtcv import NoahTCVNet
from mindediting.models.image_inpainting.ctsdg.discriminator.discriminator import Discriminator
from mindediting.models.image_inpainting.ctsdg.generator.generator import Generator
from mindediting.models.image_super_resolution.fsrcnn import FSRCNN
from mindediting.models.image_super_resolution.rrdb import RRDBNet
from mindediting.models.mpfer.mpfer import MPFER
from mindediting.models.mutil_task.ipt import IPT, IPT_post
from mindediting.models.mutil_task.rvrt import RVRT
from mindediting.models.mutil_task.rvrt_light import RVRT_LIGHT
from mindediting.models.mutil_task.vrt import VRT
from mindediting.models.tunable_image_denoise.tunable_nafnet import TunableNAFNet, TunableNAFNet_post
from mindediting.models.tunable_image_denoise_deblur.tunable_edsr import TunableEDSR, TunableEDSR_post
from mindediting.models.tunable_image_style.tunable_stylenet import TunableStyleNet, TunableStyleNet_post
from mindediting.models.tunable_mutil_task.tunable_swinir import TunableSwinIR, TunableSwinIR_post
from mindediting.models.video_denoise.emvd import EMVD_post, EMVDNet
from mindediting.models.video_frame_interpolation.ifr_plus import IFRPlus
from mindediting.models.video_super_resolution.basicvsr import BasicVSRNet
from mindediting.models.video_super_resolution.basicvsr_plus_plus_light import BasicVSRPlusPlusLightNet
from mindediting.models.video_super_resolution.ttvsr import TTVSRNet
from mindediting.utils.checkpoint import load_vrt
from mindediting.utils.download import DownLoad
from mindediting.utils.init_weights import init_weights
from mindediting.utils.utils import cast_module, change_dict_name, check_if_mirrored, is_ascend
from mindediting.deploy.utils.config import Config, parse_yaml

_module_to_models = defaultdict(set)
_model_to_module = {}
_model_entrypoints = {}
_model_has_pretrained = set()


def register_model(fn):
    # lookup containing module
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split(".")
    module_name = module_name_split[-1] if len(module_name_split) else ""

    # add model to __all__ in module
    model_name = fn.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]

    # add entries to registry dict/sets
    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False
    if hasattr(mod, "default_cfgs") and model_name in mod.default_cfgs:
        cfg = mod.default_cfgs[model_name]
        has_pretrained = "url" in cfg and cfg["url"]
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn


@register_model
def basicvsr(cfg):
    mirrored_train = check_if_mirrored(cfg.cfg_dict["train_pipeline"]["pipeline"])
    mirrored_test = check_if_mirrored(cfg.cfg_dict["test_pipeline"]["pipeline"])
    net = BasicVSRNet(
        spynet_pretrained=None if cfg.mode == "val" else cfg.model.spynet_load_path,
        is_mirror_extended_train=mirrored_train,
        is_mirror_extended_test=mirrored_test,
        precompute_grid=cfg.optimization.precompute_grid,
        base_resolution=cfg.optimization.basicvsr.base_resolution,
        levels=cfg.optimization.basicvsr.levels,
        sp_base_resolution=cfg.optimization.spynet.base_resolution,
        sp_levels=cfg.optimization.spynet.levels,
        eliminate_gradient_for_gather=cfg.optimization.eliminate_gradient_for_gather,
    )
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)
    return net, WithEvalCellNoLoss(net)


@register_model
def ipt(cfg):
    net = IPT(cfg)
    if not cfg.model.is_training_finetune:
        init_weights(net, init_type="he", init_gain=1.0)
    if cfg.model.load_path:
        param_dict = mindspore.load_checkpoint(cfg.model.load_path)
        new_param_dict = change_dict_name(param_dict)
        mindspore.load_param_into_net(net, new_param_dict, strict_load=True)

    return net, WithEvalCellNoLoss(IPT_post(net, cfg))


@register_model
def ttvsr(cfg):
    net = TTVSRNet(
        mid_channels=cfg.model.mid_channels,
        num_blocks=cfg.model.num_blocks,
        stride=cfg.model.stride,
        spynet_pretrained=cfg.model.spynet_load_path,
    )
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)

    return net, WithEvalCellNoLoss(net)


@register_model
def mimo_unet(cfg):
    net = MIMOUNet()
    load_path = cfg.model.load_path
    if load_path and Path(load_path).exists():
        param_dict = mindspore.load_checkpoint(load_path)
        mindspore.load_param_into_net(net, param_dict, strict_load=True)

    # eval_network
    is_add_cast_fp32 = cfg.loss.amp_level in ["O2", "O3", "auto"]
    loss = create_loss(loss_name=cfg.loss.name, **cfg.loss.cfg_dict)
    eval_network = WithEvalCellOutFix(net, loss, is_add_cast_fp32)
    return net, eval_network


@register_model
def fsrcnn(cfg):
    net = FSRCNN(scale_factor=cfg.model.scale, num_channels=3 if cfg.model.rgb else 1)
    load_path = cfg.model.load_path
    if cfg.mode == "val" and load_path and Path(load_path).exists():
        param_dict = mindspore.load_checkpoint(load_path)
        mindspore.load_param_into_net(net, param_dict, strict_load=True)
        net.set_train(False)
    return net, WithEvalCellNoLoss(net)


@register_model
def mpfer(cfg):
    def load_params(params_filename):
        with open(params_filename) as file:
            params = json.load(file)
        return params

    params = load_params(f"{cfg.model.load_path}/{cfg.model.version}_ms/params.json")

    net = MPFER(params)
    load_path = f"{cfg.model.load_path}/{cfg.model.version}_ms/{cfg.model.version}_ms.ckpt"
    if cfg.mode == "val" and load_path and Path(load_path).exists():
        param_dict = mindspore.load_checkpoint(load_path)
        mindspore.load_param_into_net(net, param_dict, strict_load=True)
        net.set_train(False)
    return net, WithEvalCellNoLoss(net)


@register_model
def ctsdg(cfg):
    generator = Generator(
        image_in_channels=cfg.model.image_in_channels,
        edge_in_channels=cfg.model.edge_in_channels,
        out_channels=cfg.model.out_channels,
    )
    load_path = getattr(cfg.model, "load_path", None)
    if load_path and Path(load_path).exists():
        print("Loading generator weights...")
        param_dict = mindspore.load_checkpoint(load_path)
        mindspore.load_param_into_net(generator, param_dict, strict_load=True)

    discriminator = Discriminator(
        image_in_channels=cfg.model.image_in_channels, edge_in_channels=cfg.model.edge_in_channels
    )
    load_path = getattr(cfg.model, "load_discriminator_path", None)
    if load_path and Path(load_path).exists():
        print("Loading discriminator weights...")
        param_dict = mindspore.load_checkpoint(load_path)
        mindspore.load_param_into_net(discriminator, param_dict, strict_load=True)

    vgg16_feature_extractor = VGG16FeatureExtractor()
    load_path = getattr(cfg.model, "pretrained_vgg", None)
    if load_path and Path(load_path).exists():
        print("Loading VGG16 weights...")
        mindspore.load_param_into_net(vgg16_feature_extractor, mindspore.load_checkpoint(load_path), strict_load=True)

    class CSTG(nn.Cell):
        def __init__(self, generator, discriminator, vgg16_feature_extractor):
            super().__init__()
            self.generator = generator
            self.discriminator = discriminator
            self.vgg16_feature_extractor = vgg16_feature_extractor

    net = CSTG(generator, discriminator, vgg16_feature_extractor)

    # eval_network
    is_add_cast_fp32 = cfg.loss.amp_level in ["O2", "O3", "auto"]
    loss = create_loss(loss_name=cfg.loss.name, **cfg.loss.cfg_dict)
    eval_network = WithEvalCellOutFix_Ctsdg(generator, loss, is_add_cast_fp32)

    return net, eval_network


@register_model
def noahtcv(cfg):
    net = NoahTCVNet(use_bn=False)
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)
    return net, WithEvalCellNoLoss(net)


@register_model
def emvd(cfg):
    net = EMVDNet()
    eval_network = EMVD_post(backbone=net, frame_num=cfg.dataset.num_frames)
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)
    return net, eval_network


@register_model
def vrt(cfg):
    net = VRT(
        upscale=cfg.dataset.scale,
        img_size=cfg.model.img_size,
        window_size=cfg.model.window_size,
        depths=cfg.model.depths,
        indep_reconsts=cfg.model.indep_reconsts,
        embed_dims=cfg.model.embed_dims,
        num_heads=cfg.model.num_heads,
        spynet_path=cfg.model.spynet_load_path,
        pa_frames=cfg.model.pa_frames,
        deformable_groups=cfg.model.deformable_groups,
    )
    if is_ascend():
        cast_module("float16", net)
    if cfg.model.load_path:
        load_vrt(net, cfg.model.load_path)

    return net, WithEvalCellNoLoss(net)


@register_model
def basicvsr_plus_plus_light(cfg):
    params = cfg.model.params.cfg_dict
    net = BasicVSRPlusPlusLightNet(**params)
    load_path = cfg.model.load_path
    if load_path:
        mindspore.load_checkpoint(load_path, net)
    return net, WithEvalCellNoLoss(net)


@register_model
def nafnet(cfg):
    net = NAFNet()
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net)
    return net, WithEvalCellNoLoss(net)


@register_model
def rvrt(cfg):
    params = cfg.model.cfg_dict
    load_path = params.pop("load_path")
    params.pop("name")
    if cfg.mode == "val":
        params["spynet_path"] = None
    net = RVRT(**params)
    if load_path:
        load_vrt(net, load_path)

    return net, WithEvalCellNoLoss(net)


@register_model
def rvrt_light(cfg):
    params = cfg.model.cfg_dict
    load_path = params.pop("load_path")
    params.pop("name")
    net = RVRT_LIGHT(**params)
    if load_path:
        load_vrt(net, load_path)

    return net, WithEvalCellNoLoss(net)


@register_model
def ifr_plus(cfg):
    net = IFRPlus(
        input_channels=cfg.model.input_channels,
        decoder_channels=cfg.model.decoder_channels,
        side_channels=cfg.model.side_channels,
        refiner_channels=cfg.model.refiner_channels,
        encoder_pretrained=cfg.model.encoder_load_path,
        to_float16=cfg.model.to_float16,
        flow_scale_factor=cfg.model.flow_scale_factor,
        refiner_scale_factor=cfg.model.refiner_scale_factor,
    )

    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)

    return net, WithEvalCellNoLoss(net)


@register_model
def rrdb(cfg):
    net = RRDBNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        mid_channels=cfg.model.mid_channels,
        num_blocks=cfg.model.num_blocks,
        scale=cfg.dataset.scale,
        internal_ch=cfg.model.internal_ch,
        bias=cfg.model.bias,
    )
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net)
    return net, WithEvalCellNoLoss(net)


@register_model
def srdiff(cfg):
    class SRDiffEvalCell(nn.WithLossCell):
        def __init__(self, net):
            super().__init__(net, None)

        def construct(self, *data):
            assert len(data) == 3
            img_lr, img_hr, img_lr_up = data
            return self._backbone(img_lr, img_lr_up), img_hr

    net = SRDiff(
        encoder=cfg.model.encoder,
        encoder_in_ch=cfg.model.encoder_in_ch,
        encoder_out_ch=cfg.model.encoder_out_ch,
        encoder_mid_ch=cfg.model.encoder_mid_ch,
        encoder_num_blocks=cfg.model.encoder_num_blocks,
        encoder_int_ch=cfg.model.encoder_int_ch,
        hidden_size=cfg.model.hidden_size,
        dim_mults=cfg.model.dim_mults,
        scale=cfg.model.scale,
        bias=cfg.model.bias,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.loss.name,
        beta_schedule=cfg.model.beta_schedule,
        beta_s=cfg.model.beta_s,
        beta_end=cfg.model.beta_end,
        res=cfg.model.res,
        res_rescale=cfg.model.res_rescale,
        up_input=cfg.model.up_input,
        encoder_load_path=cfg.model.encoder_load_path,
        input_shape=getattr(cfg.model, "input_shape", None),
    )
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net)
    return net, SRDiffEvalCell(net)


@register_model
def tunable_edsr(cfg):
    net = TunableEDSR(
        img_channels=cfg.model.img_channels,
        scale=cfg.dataset.scale,
        num_params=cfg.model.num_params,
        mode=cfg.model.mode,
    )
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)
    net.set_train(False)
    eval_network = TunableEDSR_post(net, cfg.model.params)
    return net, eval_network


@register_model
def tunable_nafnet(cfg):
    net = TunableNAFNet(img_channels=cfg.model.img_channels, num_params=cfg.model.num_params, mode=cfg.model.mode)
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)
    net.set_train(False)
    eval_network = TunableNAFNet_post(net, cfg.model.params)
    return net, eval_network


@register_model
def tunable_stylenet(cfg):
    net = TunableStyleNet(img_channels=cfg.model.img_channels, num_params=cfg.model.num_params, mode=cfg.model.mode)
    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)
    net.set_train(False)
    eval_network = TunableStyleNet_post(net, cfg.model.params)
    return net, eval_network


@register_model
def tunable_swinir(cfg):
    net = TunableSwinIR(
        img_channels=cfg.model.img_channels,
        window_size=cfg.model.window_size,
        depths=cfg.model.depths,
        num_heads=cfg.model.num_heads,
        embed_dim=cfg.model.embed_dim,
        mlp_ratio=cfg.model.mlp_ratio,
        resi_connection=cfg.model.resi_connection,
        num_params=cfg.model.num_params,
        mode=cfg.model.mode,
        upsampler=cfg.model.upsampler,
        upscale=cfg.model.upscale,
    )

    if cfg.model.load_path:
        mindspore.load_checkpoint(cfg.model.load_path, net, strict_load=True)
    net.set_train(False)
    eval_network = TunableSwinIR_post(net, cfg.model.params)
    return net, eval_network


@register_model
def om(cfg):
    load_path = cfg.model.load_path
    profile = getattr(cfg.model, "profile", False)

    from mindediting.deploy.backend.ascend.ascend_backend import AscendBackend

    class OmNet:
        def __init__(self, model_file):
            self.backend = AscendBackend(model_file, profiler=profile)

        def __call__(self, input_tensor):
            if not isinstance(input_tensor, (np.ndarray, list)):
                raise RuntimeError()
            output_data = self.backend.run(input_list=input_tensor)
            return output_data

        def set_train(self, mode):
            pass

    net = OmNet(load_path)

    return None, net


def download_ckpt(cfg):
    download = DownLoad()
    for k, v in cfg.model.cfg_dict.items():
        if isinstance(v, str) and v.startswith("http") and v.endswith(".ckpt"):
            path = f"{cfg.callback.ckpt_save_dir}/download_ckpt"
            download.download_url(url=v, path=path)
            setattr(cfg.model, k, path)

MODEL_NAME_TO_DEFAULT_INFO = {
    'IPT_DENOISE': ['ipt', 'configs/ipt/denoise/val_denoise_50.yaml'],
    'IPT_DERAIN': ['ipt', 'configs/ipt/derain/val_derain.yaml'],
    'IPT_SR_X2': ['ipt', 'configs/ipt/super_resolution/val_sr_x2.yaml'],
    'IPT_SR_X3': ['ipt', 'configs/ipt/super_resolution/val_sr_x3.yaml'],
    'IPT_SR_X4': ['ipt', 'configs/ipt/super_resolution/val_sr_x4.yaml'],
    'FSRCNN_SR_X4': ['fsrcnn', 'configs/fsrcnn/val_sr_x4_Set5_gpu.yaml'],
    'CTSDG': ['ctsdg', 'configs/ctsdg/val_inpainting_celeba_gpu.yaml'],
    'MIMOUNET': ['mimo_unet', 'configs/mimo_unet/val_deblur_gopro_gpu.yaml'],
    'NOAHTCV': ['noahtcv', 'configs/noahtcv/val.yaml'],

}

def create_model_by_name(model_name, cfg=None):
    cfg, network_creator = get_model_info(model_name, cfg)
    download_ckpt(cfg)
    if network_creator not in _model_entrypoints:
        raise Exception
    net, eval_network = _model_entrypoints[network_creator](cfg)
    if eval_network:
        eval_network.set_train(mode=False)
    return net, eval_network

def get_model_info(model_name, cfg=None):
    if cfg is None:
        default_cfg_path = MODEL_NAME_TO_DEFAULT_INFO[model_name][1]
        cfg, _, _ = parse_yaml(default_cfg_path)
        cfg = Config(default_cfg)
        cfg.model.load_path = None
    model_creator_name = MODEL_NAME_TO_DEFAULT_INFO[model_name][0]
    return cfg, model_creator_name



if __name__ == "__main__":
    model_name_test = "VRT"
    model_test = create_model_by_name(model_name_test)
    print("model", model_test)
