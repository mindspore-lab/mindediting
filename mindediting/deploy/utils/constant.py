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

SUCCESS = 0
FAILED = 1

ACL_DEVICE = 0
ACL_HOST = 1

MEMORY_NORMAL = 0
MEMORY_HOST = 1
MEMORY_DEVICE = 2
MEMORY_DVPP = 3
MEMORY_CTYPES = 4

IMAGE_DATA_NUMPY = 0
IMAGE_DATA_BUFFER = 1

READ_VIDEO_OK = 0

# error code
ACL_SUCCESS = 0
ACL_ERROR_INVALID_PARAM = 100000
ACL_ERROR_UNINITIALIZE = 100001
ACL_ERROR_REPEAT_INITIALIZE = 100002
ACL_ERROR_INVALID_FILE = 100003
ACL_ERROR_WRITE_FILE = 100004
ACL_ERROR_INVALID_FILE_SIZE = 100005
ACL_ERROR_PARSE_FILE = 100006
ACL_ERROR_FILE_MISSING_ATTR = 100007
ACL_ERROR_FILE_ATTR_INVALID = 100008
ACL_ERROR_INVALID_DUMP_CONFIG = 100009
ACL_ERROR_INVALID_PROFILING_CONFIG = 100010
ACL_ERROR_INVALID_MODEL_ID = 100011
ACL_ERROR_DESERIALIZE_MODEL = 100012
ACL_ERROR_PARSE_MODEL = 100013
ACL_ERROR_READ_MODEL_FAILURE = 100014
ACL_ERROR_MODEL_SIZE_INVALID = 100015
ACL_ERROR_MODEL_MISSING_ATTR = 100016
ACL_ERROR_MODEL_INPUT_NOT_MATCH = 100017
ACL_ERROR_MODEL_OUTPUT_NOT_MATCH = 100018
ACL_ERROR_MODEL_NOT_DYNAMIC = 100019
ACL_ERROR_OP_TYPE_NOT_MATCH = 100020
ACL_ERROR_OP_INPUT_NOT_MATCH = 100021
ACL_ERROR_OP_OUTPUT_NOT_MATCH = 100022
ACL_ERROR_OP_ATTR_NOT_MATCH = 100023
ACL_ERROR_OP_NOT_FOUND = 100024
ACL_ERROR_OP_LOAD_FAILED = 100025
ACL_ERROR_UNSUPPORTED_DATA_TYPE = 100026
ACL_ERROR_FORMAT_NOT_MATCH = 100027
ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED = 100028
ACL_ERROR_KERNEL_NOT_FOUND = 100029
ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED = 100030
ACL_ERROR_KERNEL_ALREADY_REGISTERED = 100031
ACL_ERROR_INVALID_QUEUE_ID = 100032
ACL_ERROR_REPEAT_SUBSCRIBE = 100033
ACL_ERROR_STREAM_NOT_SUBSCRIBE = 100034
ACL_ERROR_THREAD_NOT_SUBSCRIBE = 100035
ACL_ERROR_WAIT_CALLBACK_TIMEOUT = 100036
ACL_ERROR_REPEAT_FINALIZE = 100037
ACL_ERROR_BAD_ALLOC = 200000
ACL_ERROR_API_NOT_SUPPORT = 200001
ACL_ERROR_INVALID_DEVICE = 200002
ACL_ERROR_MEMORY_ADDRESS_UNALIGNED = 200003
ACL_ERROR_RESOURCE_NOT_MATCH = 200004
ACL_ERROR_INVALID_RESOURCE_HANDLE = 200005
ACL_ERROR_STORAGE_OVER_LIMIT = 300000
ACL_ERROR_INTERNAL_ERROR = 500000
ACL_ERROR_FAILURE = 500001
ACL_ERROR_GE_FAILURE = 500002
ACL_ERROR_RT_FAILURE = 500003
ACL_ERROR_DRV_FAILURE = 500004
ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID = 145013

# data format
ACL_FORMAT_UNDEFINED = -1
ACL_FORMAT_NCHW = 0
ACL_FORMAT_NHWC = 1
ACL_FORMAT_ND = 2
ACL_FORMAT_NC1HWC0 = 3
ACL_FORMAT_FRACTAL_Z = 4
ACL_DT_UNDEFINED = -1
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT8 = 2
ACL_INT32 = 3
ACL_UINT8 = 4
ACL_INT16 = 6
ACL_UINT16 = 7
ACL_UINT32 = 8
ACL_INT64 = 9
ACL_UINT64 = 10
ACL_DOUBLE = 11
ACL_BOOL = 12

# error code
ACL_ERROR_NONE = 0

# rule for mem
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

# images format
IMG_EXT = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP", ".jpeg", ".JPEG"]

VALID_COLORSPACE = {"rgb", "bgr", "lab", "yuv", "ycrcb", "gray3d", "gray", "yuv", "y"}
VALID_MODE = {"train", "eval", "inference", "freeze"}
VALID_PARADIGM = {"dni"}
VALID_DEBUG_MODE = {"zeroin", "intermediate"}
VALID_TASK = {"vsr", "denoise", "face", "hdr", "vfi"}

# HDR
FILE_EXT_TO_PIX_FMT = {
    "exr": "gbrpf32le",
    "png": "bgr24",
}
VALID_FILE_EXT = FILE_EXT_TO_PIX_FMT.keys()


# define task
class Task:
    IMG_SUPER_RESOLUTION = "ImgSuperResolution"
    Video_SUPER_RESOLUTION = "VideoSuperResolution"

    # TODO: add other task


# frontend state
class STATE:
    RUNNING = r"RUNNING"
    PENDING = r"PENDING"
    DONE = r"DONE"


# io backend
class IO_BACKEND:
    DISK = "disk"
    MEMORY = "memory"
    FFMPEG = "ffmpeg"

    @classmethod
    def CHECK_VALID(cls, io_backend):
        assert io_backend in {cls.DISK, cls.MEMORY, cls.FFMPEG}, f"Invalid io backend {io_backend}"