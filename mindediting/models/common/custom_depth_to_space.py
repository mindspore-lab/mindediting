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

import mindspore.nn as nn
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops.op_info_register import DataType, TBERegOp, op_info_register
from mindspore.ops.primitive import PrimitiveWithInfer, prim_attr_register

try:
    from te import tik
    from topi.cce import util
except ImportError:
    pass


custom_depth_to_space_op_info = (
    TBERegOp("CustomDepthToSpace")
    .fusion_type("OPAQUE")
    .async_flag(False)
    .binfile_name("custom_depth_to_space.so")
    .compute_cost(10)
    .kernel_name("custom_depth_to_space")
    .partial_flag(True)
    .input(0, "x", False, "required", "all")
    .output(0, "y", False, "required", "all")
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD)
    .get_op_info()
)


def _get_tik():
    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
    return tik_instance


@op_info_register(custom_depth_to_space_op_info)
def custom_depth_to_space(input_x, output, kernel_name="custom_depth_to_space"):
    input_shape = input_x.get("shape")
    output_shape = output.get("shape")
    input_shape = tuple(input_shape)

    tik_instance = _get_tik()

    input_x = tik_instance.Tensor("float16", input_shape, name="input_x", scope=tik.scope_gm)
    res = tik_instance.Tensor("float16", output_shape, name="res", scope=tik.scope_gm)

    dtype = "float16"
    tik_instance, res = depth_to_space_core(tik_instance, input_x, res, dtype)

    tik_instance.BuildCCE(kernel_name, inputs=[input_x], outputs=[res])
    return tik_instance


def get_steps(shape):
    steps = [
        1,
    ]
    for dim in reversed(shape):
        steps.append(steps[-1] * dim)
    steps = steps[::-1][1:]
    return steps


def elements_to_blocks(n):
    element_size_byte = 16 // 8  # 16 bit for fp16 // 8 bits in byte
    block_size_byte = 32
    return n * element_size_byte // block_size_byte


def depth_to_space_core(tik_instance, input_x, res, dtype):
    input_shape = tuple(input_x.shape)
    n, c1, h, w, c0 = input_shape
    s = 2
    c = c1 // s // s

    ai_core_size = 32
    while (c * h) % ai_core_size != 0:
        ai_core_size //= 2
    assert (c * h) % ai_core_size == 0, f"{c} {h} {ai_core_size}"

    input_shape = (n, s, s, ai_core_size, (c * h) // ai_core_size, w, c0)
    input_steps = get_steps(input_shape)

    def input_offset(n=0, j=0, i=0, ai_core=0, ch=0, w=0, c0=0):
        return sum(idx * step for idx, step in zip([n, j, i, ai_core, ch, w, c0], input_steps))

    output_shape = (n, ai_core_size, (c * h) // ai_core_size, s, w, s, c0)
    output_steps = get_steps(output_shape)

    def output_offset(n=0, ai_core=0, ch=0, j=0, w=0, i=0, c0=0):
        return sum(idx * step for idx, step in zip([n, ai_core, ch, j, w, i, c0], output_steps))

    with tik_instance.for_range(0, n) as n_idx, tik_instance.for_range(
        0, ai_core_size, block_num=ai_core_size
    ) as ai_core_idx, tik_instance.for_range(0, input_shape[4]) as ch_idx:
        buffer_size = output_offset(ch=1)
        multi_row_buffer = tik_instance.Tensor("float16", [buffer_size], name="multi_row_buffer", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, s) as j, tik_instance.for_range(0, s, thread_num=s) as i:
            tik_instance.data_move(
                multi_row_buffer[output_offset(j=j, i=i)],
                input_x[input_offset(n_idx, j, i, ai_core_idx, ch_idx)],
                0,
                w,
                elements_to_blocks(c0),
                0,
                elements_to_blocks(c0),
            )
        tik_instance.data_move(
            res[output_offset(n_idx, ai_core_idx, ch_idx)],
            multi_row_buffer,
            0,
            1,
            elements_to_blocks(buffer_size),
            0,
            0,
        )
    return tik_instance, res


class CustomDepthToSpace(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=["x"], outputs=["y"])

    def infer_shape(self, data1_shape):
        n, c, h, w = data1_shape
        s = 2
        shape = (n, c // s // s, h * s, w * s)
        return shape

    def infer_dtype(self, data1_dtype):
        return data1_dtype


custom_space_to_depth_op_info = (
    TBERegOp("CustomSpaceToDepth")
    .fusion_type("OPAQUE")
    .async_flag(False)
    .binfile_name("custom_space_to_depth.so")
    .compute_cost(10)
    .kernel_name("custom_space_to_depth")
    .partial_flag(True)
    .input(0, "x", False, "required", "all")
    .output(0, "y", False, "required", "all")
    .dtype_format(DataType.F16_5HD, DataType.F16_5HD)
    .get_op_info()
)


@op_info_register(custom_space_to_depth_op_info)
def custom_space_to_depth(input_x, output, kernel_name="custom_space_to_depth"):
    input_shape = input_x.get("shape")
    output_shape = output.get("shape")
    input_shape = tuple(input_shape)

    tik_instance = _get_tik()

    input_x = tik_instance.Tensor("float16", input_shape, name="input_x", scope=tik.scope_gm)
    res = tik_instance.Tensor("float16", output_shape, name="res", scope=tik.scope_gm)

    dtype = "float16"
    tik_instance, res = space_to_depth_core(tik_instance, input_x, res, dtype)

    tik_instance.BuildCCE(kernel_name, inputs=[input_x], outputs=[res])
    return tik_instance


def space_to_depth_core(tik_instance, input_x, res, dtype):
    input_shape = tuple(input_x.shape)
    n, c, h, w, c0 = input_shape
    s = 2
    h = h // s
    w = w // s

    ai_core_size = 32
    while (c * h) % ai_core_size != 0:
        ai_core_size //= 2
    assert (c * h) % ai_core_size == 0, f"{c} {h} {ai_core_size}"
    ch_group_size = (c * h) // ai_core_size

    input_shape = (n, ai_core_size, ch_group_size, s, w, s, c0)
    input_steps = get_steps(input_shape)

    def input_offset(n=0, ai_core=0, ch=0, j=0, w=0, i=0, c0=0):
        return sum(idx * step for idx, step in zip([n, ai_core, ch, j, w, i, c0], input_steps))

    output_shape = (n, s, s, ai_core_size, ch_group_size, w, c0)
    output_steps = get_steps(output_shape)

    def output_offset(n=0, j=0, i=0, ai_core=0, ch=0, w=0, c0=0):
        return sum(idx * step for idx, step in zip([n, j, i, ai_core, ch, w, c0], output_steps))

    with tik_instance.for_range(0, n) as n_idx, tik_instance.for_range(0, s) as j, tik_instance.for_range(
        0, s
    ) as i, tik_instance.for_range(0, ai_core_size, block_num=ai_core_size) as ai_core_idx:
        buffer_size = output_offset(ai_core=1)
        multi_row_buffer = tik_instance.Tensor("float16", [buffer_size], name="multi_row_buffer", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, ch_group_size) as ch_idx:
            tik_instance.data_move(
                multi_row_buffer[output_offset(ch=ch_idx)],
                input_x[input_offset(n_idx, ai_core_idx, ch_idx, j, i=i)],
                0,
                w,
                elements_to_blocks(c0),
                elements_to_blocks(c0),
                0,
            )
        tik_instance.data_move(
            res[output_offset(n_idx, j, i, ai_core_idx)], multi_row_buffer, 0, 1, elements_to_blocks(buffer_size), 0, 0
        )
    return tik_instance, res


class CustomSpaceToDepth(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=["x"], outputs=["y"])

    def infer_shape(self, data1_shape):
        n, c, h, w = data1_shape
        s = 2
        h //= s
        w //= s
        shape = (n, s * s * c, h, w)
        return shape

    def infer_dtype(self, data1_dtype):
        return data1_dtype


@bprop_getters.register(CustomDepthToSpace)
def get_bprop_depth_to_space(self):
    """Generate bprop for CustomDepthToSpace"""
    op = CustomSpaceToDepth()

    def bprop(x, out, dout):
        return (op(dout),)

    return bprop


@bprop_getters.register(CustomSpaceToDepth)
def get_bprop_depth_to_space(self):
    """Generate bprop for CustomSpaceToDepth"""
    op = CustomDepthToSpace()

    def bprop(x, out, dout):
        return (op(dout),)

    return bprop


class CusDepthToSpace(nn.Cell):
    def __init__(self):
        super().__init__()
        self.core = CustomDepthToSpace()

    def construct(self, x):
        return self.core(x)


class CusSpaceToDepth(nn.Cell):
    def __init__(self):
        super().__init__()
        self.core = CustomSpaceToDepth()

    def construct(self, x):
        return self.core(x)
