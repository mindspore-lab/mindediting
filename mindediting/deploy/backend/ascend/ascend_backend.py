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

import os

import acl
import numpy as np
from deploy.backend.backend_metaclass import Backend
from deploy.utils import constant as const
from deploy.utils.wrapper import func_time, singleton

buffer_method = {"in": acl.mdl.get_input_size_by_index, "out": acl.mdl.get_output_size_by_index}


def check_ret(message: str, ret_int: int):
    """check the error code."""
    if ret_int != 0:
        # skip ACL_ERROR_REPEAT_INITIALIZE and ACL_ERROR_REPEAT_FINALIZE errors
        if ret_int == 100002 or ret_int == 100037:
            print("Warning: Repeated initializetion of ACL")
        else:
            raise Exception("{} failed ret_int={}".format(message, ret_int))


@singleton
class AclInit(object):
    def __init__(self):
        ret = acl.init()
        check_ret("acl.init", ret)

    def __del__(self):
        ret = acl.finalize()
        check_ret("acl.finalize", ret)


class AscendBackend(Backend):
    """Ascend  class for inference.

    Args:
        model (str): Path of the model file.
    """

    def __init__(self, model_path, profiler=False):
        # self.prof_config = None
        super().__init__(model_path)

        self.device_id = int(os.environ.get("DEVICE_ID", "0"))
        self.model_path = model_path  # string
        self.model_id = self.device_id  # pointer
        self.context = None  # pointer
        self.stream = None
        self.prof_config = None

        self.input_data = []
        self.output_data = []
        self.model_desc = None  # pointer when using
        self.load_input_dataset = None
        self.load_output_dataset = None
        self.profiler = profiler

        self.init_resource()

    def __del__(self):
        print("Releasing resources stage:")
        ret = acl.mdl.unload(self.model_id)
        check_ret("acl.mdl.unload", ret)
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)
            self.model_desc = None

        while self.input_data:
            item = self.input_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        while self.output_data:
            item = self.output_data.pop()
            ret = acl.rt.free(item["buffer"])
            check_ret("acl.rt.free", ret)

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)
        # ret = acl.finalize()
        # check_ret("acl.finalize", ret)

        if self.profiler:
            ret = acl.prof.stop(self.prof_config)
            check_ret("acl.prof.stop", ret)
            ret = acl.prof.destroy_config(self.prof_config)
            check_ret("acl.prof.destroy_config", ret)
            ret = acl.prof.finalize()
            check_ret("acl.prof.finalize", ret)
            ret = acl.rt.destroy_stream(self.stream)
        print("Resources released successfully.")

    def init_resource(self):
        print("init resource stage:")
        AclInit()
        # check_ret("acl.init", ret)

        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.mdl.create_stream", ret)

        # load_model
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)
        print("model_id:{}".format(self.model_id))

        self.model_desc = acl.mdl.create_desc()
        self._get_model_info()
        print("init resource success")

        self.output_tensor_list = self._gen_output_tensor()

        if self.profiler:
            # Profiling init
            PROF_INIT_PATH = "./"
            ret = acl.prof.init(PROF_INIT_PATH)
            check_ret("acl.prof.init", ret)

            device_list = [0]
            ACL_PROF_ACL_API = 0x0001
            ACL_PROF_TASK_TIME = 0x0002
            ACL_PROF_AICORE_METRICS = 0x0004
            # Create an address of the configuration pointer
            self.prof_config = acl.prof.create_config(
                device_list, 0, 0, ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS
            )
            ret = acl.prof.start(self.prof_config)
            check_ret("acl.prof.start", ret)

    def get_input_shape(self, index=0):
        input_info, ret = acl.mdl.get_input_dims(self.model_desc, index)
        check_ret("acl.mdl.get_input_dims", ret)
        return input_info["dims"]

    def get_output_shape(self, index=0):
        input_info, ret = acl.mdl.get_output_dims(self.model_desc, index)
        check_ret("acl.mdl.get_input_dims", ret)
        return input_info["dims"]

    def _get_model_info(
        self,
    ):
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)
        input_size = acl.mdl.get_num_inputs(self.model_desc)
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        self._gen_data_buffer(input_size, des="in")
        self._gen_data_buffer(output_size, des="out")

    def _gen_data_buffer(self, size, des):
        func = buffer_method[des]
        for i in range(size):
            # check temp_buffer dtype
            temp_buffer_size = func(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, const.ACL_MEM_MALLOC_HUGE_FIRST)
            check_ret("acl.rt.malloc", ret)

            if des == "in":
                self.input_data.append({"buffer": temp_buffer, "size": temp_buffer_size})
            elif des == "out":
                self.output_data.append({"buffer": temp_buffer, "size": temp_buffer_size})

    def _gen_output_tensor(self):
        output_tensor_list = []
        output_size = acl.mdl.get_num_outputs(self.model_desc)
        for i in range(output_size):
            dims = acl.mdl.get_output_dims(self.model_desc, i)
            shape = tuple(dims[0]["dims"])
            datatype = acl.mdl.get_output_data_type(self.model_desc, i)
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)

            if datatype == const.ACL_FLOAT:
                np_type = np.float32
                output_tensor = np.zeros(size // 4, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_DOUBLE:
                np_type = np.float64
                output_tensor = np.zeros(size // 8, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_INT64:
                np_type = np.int64
                output_tensor = np.zeros(size // 8, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_UINT64:
                np_type = np.uint64
                output_tensor = np.zeros(size // 8, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_INT32:
                np_type = np.int32
                output_tensor = np.zeros(size // 4, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_UINT32:
                np_type = np.uint32
                output_tensor = np.zeros(size // 4, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_FLOAT16:
                np_type = np.float16
                output_tensor = np.zeros(size // 2, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_INT16:
                np_type = np.int16
                output_tensor = np.zeros(size // 2, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_UINT16:
                np_type = np.uint16
                output_tensor = np.zeros(size // 2, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_INT8:
                np_type = np.int8
                output_tensor = np.zeros(size, dtype=np_type).reshape(shape)
            elif datatype == const.ACL_BOOL or datatype == const.ACL_UINT8:
                np_type = np.uint8
                output_tensor = np.zeros(size, dtype=np_type).reshape(shape)
            else:
                print("Unspport model output datatype ", datatype)
                return None

            if not output_tensor.flags["C_CONTIGUOUS"]:
                output_tensor = np.ascontiguousarray(output_tensor)

            if "bytes_to_ptr" in dir(acl.util):
                bytes_data = output_tensor.tobytes()
                tensor_ptr = acl.util.bytes_to_ptr(bytes_data)
                output_tensor_list.append(
                    {
                        "ptr": tensor_ptr,
                        "tensor": bytes_data,
                        "shape": output_tensor.shape,
                        "dtype": output_tensor.dtype,
                    },
                )
            else:
                tensor_ptr = acl.util.numpy_to_ptr(output_tensor)
                output_tensor_list.append({"ptr": tensor_ptr, "tensor": output_tensor})

        return output_tensor_list

    def _data_interaction(self, dataset, policy=const.ACL_MEMCPY_HOST_TO_DEVICE):
        temp_data_buffer = self.input_data if policy == const.ACL_MEMCPY_HOST_TO_DEVICE else self.output_data
        if len(dataset) == 0 and policy == const.ACL_MEMCPY_DEVICE_TO_HOST:
            for item in self.output_data:
                temp, ret = acl.rt.malloc_host(item["size"])
                if ret != 0:
                    raise Exception("can't malloc_host ret={}".format(ret))
                dataset.append({"size": item["size"], "buffer": temp})

        for i, item in enumerate(temp_data_buffer):
            if policy == const.ACL_MEMCPY_HOST_TO_DEVICE:
                bytes_in = dataset[i].tobytes()
                ptr = acl.util.bytes_to_ptr(bytes_in)
                ret = acl.rt.memcpy(item["buffer"], item["size"], ptr, item["size"], policy)
                check_ret("acl.rt.memcpy", ret)

            else:
                ptr = dataset[i]["buffer"]
                ret = acl.rt.memcpy(ptr, item["size"], item["buffer"], item["size"], policy)
                check_ret("acl.rt.memcpy", ret)

    def _gen_dataset(self, type_str="input"):
        dataset = acl.mdl.create_dataset()

        if type_str == "in":
            self.load_input_dataset = dataset
            temp_dataset = self.input_data
        else:
            self.load_output_dataset = dataset
            temp_dataset = self.output_data

        for item in temp_dataset:
            data = acl.create_data_buffer(item["buffer"], item["size"])
            _, ret = acl.mdl.add_dataset_buffer(dataset, data)

            if ret != const.ACL_ERROR_NONE:
                ret = acl.destroy_data_buffer(data)
                check_ret("acl.destroy_data_buffer", ret)

    def _data_from_host_to_device(self, input_list):
        # print("data interaction from host to device")
        # copy input data to device
        self._data_interaction(input_list, const.ACL_MEMCPY_HOST_TO_DEVICE)
        # load input data into model
        self._gen_dataset("in")
        # load output data into model
        self._gen_dataset("out")
        # print("data interaction from host to device success")

    def _data_from_device_to_host(self):
        # print("data interaction from device to host")
        res = []
        # copy device to host
        self._data_interaction(res, const.ACL_MEMCPY_DEVICE_TO_HOST)
        # print("data interaction from device to host success")
        result = self.get_result(res)
        return result

    def _set_dynamic_batch_size(self, batch):
        dynamicIdx, ret = acl.mdl.get_input_index_by_name(self.model_desc, "ascend_mbatch_shape_data")
        if ret != const.ACL_SUCCESS:
            print("get_input_index_by_name failed")
            return const.FAILED
        batch_dic, ret = acl.mdl.get_dynamic_batch(self.model_desc)
        if ret != const.ACL_SUCCESS:
            print("get_dynamic_batch failed")
            return const.FAILED
        print("[INFO] get dynamic_batch = ", batch_dic)
        ret = acl.mdl.set_dynamic_batch_size(self.model_id, self.load_input_dataset, dynamicIdx, batch)
        if ret != const.ACL_SUCCESS:
            print("set_dynamic_batch_size failed, ret = ", ret)
            return const.FAILED
        if batch in batch_dic["batch"]:
            return const.SUCCESS
        else:
            assert ret == const.ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID
            print("[INFO] [dynamic batch] {} is not in {}".format(batch, batch_dic["batch"]))
            return const.FAILED

    def run(self, input_list, **kwargs):
        acl.rt.set_context(self.context)
        results = []
        for input_data in input_list:
            if not isinstance(input_data, list):
                input_data = [input_data]
            self._data_from_host_to_device(input_data)
            self.forward(**kwargs)
            tmp_res = self._data_from_device_to_host()
            results.append(tmp_res)
        return results

    def run_with_dynamic_batch_size(self, input_list, batch_size):
        self._data_from_host_to_device(input_list)
        self._set_dynamic_batch_size(batch_size)
        self.forward()
        results = self._data_from_device_to_host()
        return results

    def _model_set_dynamicInfo(self, height, width):
        # Obtains the index of the dynamic resolution input. The input name of the dynamic resolution input is fixed
        # to ascend_mbatch_shape_data.
        # Set the input image resolution. model_id indicates the ID of the successfully loaded model,
        # input indicates the data of the aclmdlDataset type,
        # and index indicates the input index of the dynamic resolution input.
        index, ret = acl.mdl.get_input_index_by_name(self.model_desc, "ascend_mbatch_shape_data")
        check_ret("acl.mdl.get_input_index_by_name", ret)
        ret = acl.mdl.set_dynamic_hw_size(self.model_id, self.load_input_dataset, index, height, width)
        check_ret("acl.mdl.set_dynamic_hw_size", ret)

    @func_time("execute one step")
    def forward(self, **kwargs):
        # Don't support for mindspore (export cannot run with dynamic hw size)
        # width, height = kwargs.get('width'), kwargs.get('height')
        # if width and height:
        #     self._model_set_dynamicInfo(height, width)
        ret = acl.mdl.execute(self.model_id, self.load_input_dataset, self.load_output_dataset)
        check_ret("acl.mdl.execute", ret)
        self._destroy_databuffer()

    def _destroy_databuffer(self):
        for dataset in [self.load_input_dataset, self.load_output_dataset]:
            if not dataset:
                continue
            number = acl.mdl.get_dataset_num_buffers(dataset)
            for i in range(number):
                data_buf = acl.mdl.get_dataset_buffer(dataset, i)
                if data_buf:
                    ret = acl.destroy_data_buffer(data_buf)
                    check_ret("acl.destroy_data_buffer", ret)
            ret = acl.mdl.destroy_dataset(dataset)
            check_ret("acl.mdl.destroy_dataset", ret)

    def get_result(self, output_data):
        dataset = []

        output_tensor_list = self._gen_output_tensor()
        for i, temp in enumerate(output_data):
            size = temp["size"]
            ptr = temp["buffer"]
            bytes_out = acl.util.ptr_to_bytes(ptr, size)
            data = np.frombuffer(bytes_out, dtype=output_tensor_list[i]["dtype"]).reshape(
                output_tensor_list[i]["shape"]
            )
            dataset.append(data)
        return dataset
