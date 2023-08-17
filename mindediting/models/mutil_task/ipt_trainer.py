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

"""define network with loss function and train by one step"""
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Parameter, Tensor, ops
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = C.MultitypeFuncGraph("clip_grad")
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    if clip_type not in (0, 1):
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(
            grad, ops.cast(ops.tuple_to_array((-clip_value,)), dt), ops.cast(ops.tuple_to_array((clip_value,)), dt)
        )
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad


class ClipGradients(nn.Cell):
    """
    Clip gradients.
    Returns:
        [List], a list of clipped_grad tuples.
    """

    def __init__(self):
        super(ClipGradients, self).__init__()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.clip_by_norm = nn.ClipByNorm()

    def construct(self, grads, clip_type, clip_value):
        if clip_type in (0, 1):
            new_grads = ()
            for grad in grads:
                dt = self.dtype(grad)
                if clip_type == 0:
                    t = ops.clip_by_value(
                        grad,
                        self.cast(ops.tuple_to_array((-clip_value,)), dt),
                        self.cast(ops.tuple_to_array((clip_value,)), dt),
                    )
                else:
                    t = self.clip_by_norm(grad, self.cast(ops.tuple_to_array((clip_value,)), dt))
                t = self.cast(t, dt)
                new_grads = new_grads + (t,)
                return new_grads
        else:
            return grads


class IPTTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of bert network training.
     Append an optimizer to the training network after that the construct
     function can be called to create the backward graph.

     Args:
         network [Cell]: The training network. Note that loss function should have been added.
         optimizer [Optimizer]: Optimizer for updating the weights.
         scale_update_cell [Cell]: Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(IPTTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = ops.Cast()
        self.degree = 1
        self.loss_scaling_manager = scale_update_cell
        self.loss_scale = None
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ms.float32))
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, lr, hr, idx, filename, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        idx = Tensor(np.ones(idx[0]), ms.int32)
        lr = lr.astype(ms.float32)
        loss = self.network(lr, hr, idx)
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens = self.cast(scaling_sens, ms.float32)
        grad = self.grad(self.network, weights)(lr, hr, idx, scaling_sens)
        grad = self.grad_reducer(grad)
        grad = self.hyper_map(F.partial(grad_scale, scaling_sens), grad)
        grad = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grad)
        conds = self.get_overflow_status(status, grad)
        overflow = conds
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, conds)
        if not overflow:
            self.optimizer(grad)
        return loss
