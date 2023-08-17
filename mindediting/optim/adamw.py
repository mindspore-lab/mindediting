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

"""Gradient clipping wrapper for optimizers."""

import mindspore as ms
import numpy as np
from mindspore import ops
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.optim import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

try:
    # MindSpore version <= 2.0.0-alpha
    from mindspore._checkparam import Rel, Validator

    check_value_type = Validator.check_value_type
    check_float_range = Validator.check_float_range
    check_positive_float = Validator.check_positive_float
    INC_NEITHER = Rel.INC_NEITHER
except:
    from mindspore._checkparam import INC_NEITHER, check_float_range, check_positive_float, check_value_type


_grad_scale = ops.MultitypeFuncGraph("grad_scale")
map_ = ops.Map()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale_with_tensor(scale, grad):
    return ops.mul(grad, ops.cast(scale, grad.dtype))


@_grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    if scale == 1.0:
        return grad
    return ops.mul(grad, ops.cast(scale, grad.dtype))


def scale_grad(gradients, reciprocal_scale):
    gradients = map_(ops.partial(_grad_scale, reciprocal_scale), gradients)
    return gradients


_adam_opt = ops.MultitypeFuncGraph("adam_opt")
_scaler_one = Tensor(1, ms.int32)


@_adam_opt.register(
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Bool",
    "Bool",
)
def _update_run_op(
    beta1_power, beta2_power, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flag, optim_filter
):
    """
    Update parameters.
    Args:
        beta1 [Tensor]: The exponential decay rate for the 1st moment estimations. Should be in range (0.0, 1.0).
        beta2 [Tensor]: The exponential decay rate for the 2nd moment estimations. Should be in range (0.0, 1.0).
        eps [Tensor]: Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr [Tensor]: Learning rate.
        weight_decay [Tensor]: Weight decay. Should be equal to or greater than 0.
        param [Tensor]: Parameters.
        m [Tensor]: m value of parameters.
        v [Tensor]: v value of parameters.
        gradient [Tensor]: Gradient of parameters.
        decay_flag [bool]: Applies weight decay or not.
        optim_filter [bool]: Applies parameter update or not.
    Returns:
        Tensor, the new value of v after updating.
    """
    if optim_filter:
        param_fp32 = ops.cast(param, ms.float32)
        m_fp32 = ops.cast(m, ms.float32)
        v_fp32 = ops.cast(v, ms.float32)
        gradient_fp32 = ops.cast(gradient, ms.float32)

        next_m = ops.mul(beta1, m_fp32) + ops.mul(
            ops.cast(ops.tuple_to_array((1.0,)), ms.float32) - beta1, gradient_fp32
        )

        next_v = ops.mul(beta2, v_fp32) + ops.mul(
            ops.cast(ops.tuple_to_array((1.0,)), ms.float32) - beta2, ops.square(gradient_fp32)
        )

        regulate_m = next_m / (_scaler_one - beta1_power)
        regulate_v = next_v / (_scaler_one - beta2_power)

        update = regulate_m / (eps + ops.sqrt(regulate_v))
        if decay_flag:
            update = ops.mul(weight_decay, param_fp32) + update

        update_with_lr = ops.mul(lr, update)
        next_param = param_fp32 - ops.reshape(update_with_lr, ops.shape(param_fp32))

        next_param = ops.depend(next_param, ops.assign(param, ops.cast(next_param, param.dtype)))
        next_param = ops.depend(next_param, ops.assign(m, ops.cast(next_m, m.dtype)))
        next_param = ops.depend(next_param, ops.assign(v, ops.cast(next_v, v.dtype)))

        return ops.cast(next_param, param.dtype)
    return gradient


class AdamW(Optimizer):
    """
    Implements the gradient clipping by norm for a AdamWeightDecay optimizer.
    """

    @opt_init_args_register
    def __init__(
        self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, loss_scale=1.0, clip=False
    ):
        super().__init__(learning_rate, params, weight_decay)
        check_value_type("eps", eps, [float], self.cls_name)
        check_value_type("beta1", beta1, [float], self.cls_name)
        check_value_type("beta2", beta2, [float], self.cls_name)
        check_float_range(beta1, 0.0, 1.0, INC_NEITHER, "beta1", self.cls_name)
        check_float_range(beta2, 0.0, 1.0, INC_NEITHER, "beta2", self.cls_name)
        check_positive_float(eps, "eps", self.cls_name)

        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.hyper_map = ops.HyperMap()
        self.beta1_power = Parameter(initializer(1, [1], ms.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], ms.float32), name="beta2_power")
        self.moments1 = self.parameters.clone(prefix="adam_m", init="zeros")
        self.moments2 = self.parameters.clone(prefix="adam_v", init="zeros")

        self.reciprocal_scale = Tensor(1.0 / loss_scale, ms.float32)
        self.clip = clip

    def construct(self, gradients):
        gradients = scale_grad(gradients, self.reciprocal_scale)

        beta1_power = self.beta1_power * self.beta1
        self.beta1_power = beta1_power
        beta2_power = self.beta2_power * self.beta2
        self.beta2_power = beta2_power
        if self.clip:
            gradients = ops.clip_by_global_norm(gradients, 5.0, None)

        lr = self.get_lr()
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps),
                    lr,
                    self.weight_decay,
                    self.parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(_adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps, lr),
                    self.weight_decay,
                    self.parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(
                    _adam_opt, beta1_power, beta2_power, self.beta1, self.beta2, self.eps, lr, self.weight_decay
                ),
                self.parameters,
                self.moments1,
                self.moments2,
                gradients,
                self.decay_flags,
                self.optim_filter,
            )
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result
