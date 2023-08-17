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

"""trainer"""

from mindspore import Parameter, Tensor, context
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context

from mindediting.utils.callbacks_ctsdg import get_callbacks
from mindediting.utils.utils import is_ascend


class TrainOneStepCell(nn.Cell):
    """TrainOneStepCell"""

    def __init__(self, network, optimizer, initial_scale_sense=1.0):
        super(TrainOneStepCell, self).__init__()
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.scale_sense = Parameter(Tensor(initial_scale_sense, dtype=mstype.float32), name="scale_sense")
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)


class GTrainOneStepCell(TrainOneStepCell):
    """Generator TrainOneStepCell"""

    def __init__(self, network, optimizer, initial_scale_sense=1.0):
        super(GTrainOneStepCell, self).__init__(network, optimizer, initial_scale_sense)
        self.network.vgg_feat_extractor.set_grad(False)
        self.network.discriminator.set_grad(False)

    def set_train(self, mode=True):
        super().set_train(mode)
        self.network.vgg_feat_extractor.set_train(False)
        self.network.discriminator.set_train(False)
        return self

    def construct(self, *inputs):
        network_fwd_bwd = ops.value_and_grad(self.network, grad_position=None, weights=self.weights, has_aux=True)
        (loss, pred), grads = network_fwd_bwd(*inputs)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        opt_res = self.optimizer(grads)
        return ops.depend((loss, pred), opt_res)


class DTrainOneStepCell(TrainOneStepCell):
    """Discriminator TrainOneStepCell"""

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs, self.scale_sense * 1.0)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


class CTSDGTrainer:
    """CTSDGTrainer"""

    def __init__(self, train_one_step_g, train_one_step_d, cfg, finetune=False):
        super(CTSDGTrainer, self).__init__()
        self.train_one_step_g = train_one_step_g
        self.train_one_step_d = train_one_step_d
        self.finetune = finetune
        self.cfg = cfg

    def run(self, *inputs):
        ground_truth, _, edge, gray_image = inputs
        self.train_one_step_g.set_train(not self.finetune)
        loss_g, output = self.train_one_step_g(*inputs)
        self.train_one_step_d.set_train()
        loss_d = self.train_one_step_d(ground_truth, gray_image, edge, output)
        return loss_g, loss_d

    def train(self, total_steps, dataset, callbacks, save_ckpt_logs=True, **kwargs):
        total_steps = self.cfg.train.total_steps
        callbacks = get_callbacks(
            self.cfg,
            self.train_one_step_g.network.generator,
            self.train_one_step_g.network.discriminator,
            self.finetune,
        )
        print(f"Start training for {total_steps} iterations")
        dataset_size = dataset.get_dataset_size()
        repeats_num = (total_steps + dataset_size - 1) // dataset_size
        dataset = dataset.repeat(repeats_num)
        dataloader = dataset.create_dict_iterator()
        for num_batch, sample in enumerate(dataloader):
            if num_batch >= total_steps:
                print("Reached the target number of iterations")
                break
            ground_truth = sample["image"]
            mask = sample["mask"]
            edge = sample["edge"]
            gray_image = sample["gray_image"]
            loss_g, loss_d = self.run(ground_truth, mask, edge, gray_image)
            if save_ckpt_logs:
                callbacks([loss_g.asnumpy().mean(), loss_d.asnumpy().mean()])

        print("Training completed")


def gram_matrix(feat):
    """gram matrix"""
    b, ch, h, w = feat.shape
    feat = feat.view(b, ch, h * w)
    gram = ops.BatchMatMul(False, True)(feat, feat) / (ch * h * w)
    return gram


class GramMat(nn.Cell):
    def construct(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h * w) / ops.sqrt(Tensor(c * h * w))
        gram = ops.BatchMatMul(False, True)(x, x)
        return gram


class GWithLossCell(nn.Cell):
    """Generator with loss cell"""

    def __init__(self, generator, discriminator, vgg_feat_extractor, cfg):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg_feat_extractor = vgg_feat_extractor

        if is_ascend():
            self.gram_matrix = GramMat().to_float(mstype.float16)
        else:
            self.gram_matrix = gram_matrix

        self.l1 = nn.L1Loss()
        self.criterion = nn.BCELoss(reduction="mean")
        self.real_target = Tensor(1.0, mstype.float32)

        self.hole_loss_w = cfg.loss.hole_loss_w
        self.valid_loss_w = cfg.loss.valid_loss_w
        self.perceptual_loss_w = cfg.loss.perceptual_loss_w
        self.style_loss_w = cfg.loss.style_loss_w
        self.adversarial_loss_w = cfg.loss.adversarial_loss_w
        self.intermediate_loss_w = cfg.loss.intermediate_loss_w

    def construct(self, ground_truth, mask, edge, gray_image):
        input_image = ground_truth * mask
        input_edge = edge * mask
        input_gray_image = gray_image * mask
        output, projected_image, projected_edge = self.generator(
            input_image, ops.Concat(axis=1)((input_edge, input_gray_image)), mask
        )

        output_pred, output_edge = self.discriminator(output, gray_image, edge, is_real=False)

        loss_hole = self.l1((1 - mask) * output, (1 - mask) * ground_truth)

        loss_valid = self.l1(mask * output, mask * ground_truth)

        comp = ground_truth * mask + output * (1 - mask)
        vgg_comp = self.vgg_feat_extractor(comp)
        vgg_output = self.vgg_feat_extractor(output)
        vgg_ground_truth = self.vgg_feat_extractor(ground_truth)

        loss_perceptual = 0.0
        for i in range(3):
            loss_perceptual += self.l1(vgg_output[i], vgg_ground_truth[i])
            loss_perceptual += self.l1(vgg_comp[i], vgg_ground_truth[i])

        loss_style = 0.0
        for i in range(3):
            mats = ops.Concat(axis=0)((vgg_ground_truth[i], vgg_output[i], vgg_comp[i]))
            gram = self.gram_matrix(mats)
            gram_gt, gram_out, gram_comp = ops.Split(axis=0, output_num=3)(gram)
            loss_style += self.l1(gram_out, gram_gt)
            loss_style += self.l1(gram_comp, gram_gt)

        real_target = self.real_target.expand_as(output_pred)
        loss_adversarial = self.criterion(output_pred, real_target) + self.criterion(output_edge, edge)

        loss_intermediate = self.criterion(projected_edge, edge) + self.l1(projected_image, ground_truth)

        loss_g = (
            loss_hole.mean() * self.hole_loss_w
            + loss_valid.mean() * self.valid_loss_w
            + loss_perceptual.mean() * self.perceptual_loss_w
            + loss_style.mean() * self.style_loss_w
            + loss_adversarial.mean() * self.adversarial_loss_w
            + loss_intermediate.mean() * self.intermediate_loss_w
        )

        result = ops.stop_gradient(output)
        return loss_g, result


class DWithLossCell(nn.Cell):
    """Discriminator with loss cell"""

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator
        self.criterion = nn.BCELoss(reduction="mean")
        self.real_target = Tensor(1.0, mstype.float32)
        self.fake_target = Tensor(0.0, mstype.float32)

    def construct(self, ground_truth, gray_image, edge, output):
        real_pred, real_pred_edge = self.discriminator(ground_truth, gray_image, edge, is_real=True)
        fake_pred, fake_pred_edge = self.discriminator(output, gray_image, edge, is_real=False)

        real_target = self.real_target.expand_as(real_pred)
        fake_target = self.fake_target.expand_as(fake_pred)

        loss_adversarial = (
            self.criterion(real_pred, real_target)
            + self.criterion(fake_pred, fake_target)
            + self.criterion(real_pred_edge, edge)
            + self.criterion(fake_pred_edge, edge)
        )
        return loss_adversarial
