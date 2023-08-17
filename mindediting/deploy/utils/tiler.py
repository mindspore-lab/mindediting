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

import time

import numpy as np


class DefaultTiler:
    def __init__(
        self,
        backend,
        frame_window_size=0,
        frame_overlap=0,
        patch_size=None,
        patch_overlap=0,
        sf=1,
        not_overlap_border=False,
        dtype="float32",
    ):
        self._result = None
        self.sf = sf
        self.frame_window_size = frame_window_size
        self.frame_overlap = frame_overlap
        self.not_overlap_border = not_overlap_border
        self.spatial_size = patch_size
        if isinstance(self.spatial_size, int):
            self.spatial_size = (self.spatial_size, self.spatial_size)  # HxW
        self.spatial_overlap = patch_overlap
        self.backend = backend
        if dtype not in {"float32", "float16"}:
            raise ValueError(f"Invalid dtype: {dtype}")
        self.dtype = np.float32 if dtype == "float32" else np.float16

    def _process_clip(self, input_data, **kwargs):
        if not self.backend:
            print("Error: backed is undefined.")
            return input_data
        spatial_size = self.spatial_size
        sf = self.sf

        # print("Process clip shape = ", input_data.shape)
        # divide the clip to patches (spatially only, tested patch by patch)
        if len(input_data.shape) == 4:
            # Image model, adding T dimension=1
            b, c, h, w = input_data.shape
            d = 1
        elif len(input_data.shape) == 5:
            b, d, c, h, w = input_data.shape
        if spatial_size[0] > 0 and spatial_size[0] < h and spatial_size[1] > 0 and spatial_size[1] < w:
            # do patching
            spatial_overlap = self.spatial_overlap
            stride = spatial_size[0] - spatial_overlap, spatial_size[1] - spatial_overlap
            h_idx_list = list(range(0, h - spatial_size[0], stride[0])) + [max(0, h - spatial_size[0])]
            w_idx_list = list(range(0, w - spatial_size[1], stride[1])) + [max(0, w - spatial_size[1])]
            E = np.zeros((b, d, c, h * sf, w * sf), dtype=self.dtype)
            W = np.zeros_like(E)
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_data[..., h_idx : h_idx + spatial_size[0], w_idx : w_idx + spatial_size[1]]
                    # print("patching shape = ", in_patch.shape)
                    res = self.backend.run(in_patch, **kwargs)
                    # CANN returns list of lists, Mindspore returns just Tensor
                    # in this case where input consists of one element, output list will contain also one element
                    if isinstance(res, list):
                        res = res[0]
                        if isinstance(res, list):
                            # get the first output of the model, currently all models return just one tensor
                            res = res[0]
                    if res.dtype != self.dtype:
                        res = res.astype(self.dtype)
                    if len(res.shape) == 4:
                        # image model, T = 1
                        res = np.expand_dims(res, axis=1)
                    out_patch = np.copy(res)
                    # out_patch_mask = np.ones_like(out_patch)
                    out_patch_mask = np.ones((b, d, c, spatial_size[0] * sf, spatial_size[1] * sf), dtype=self.dtype)
                    not_overlap_border = True
                    if spatial_overlap > 0 and not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -spatial_overlap // 2 :, :] *= 0
                            out_patch_mask[..., -spatial_overlap // 2 :, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -spatial_overlap // 2 :] *= 0
                            out_patch_mask[..., :, -spatial_overlap // 2 :] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., : spatial_overlap // 2, :] *= 0
                            out_patch_mask[..., : spatial_overlap // 2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, : spatial_overlap // 2] *= 0
                            out_patch_mask[..., :, : spatial_overlap // 2] *= 0
                    if spatial_overlap > 0:
                        E[
                            ...,
                            h_idx * sf : (h_idx + spatial_size[0]) * sf,
                            w_idx * sf : (w_idx + spatial_size[1]) * sf,
                        ] += out_patch
                        W[
                            ...,
                            h_idx * sf : (h_idx + spatial_size[0]) * sf,
                            w_idx * sf : (w_idx + spatial_size[1]) * sf,
                        ] += out_patch_mask
                    else:
                        E[
                            ...,
                            h_idx * sf : (h_idx + spatial_size[0]) * sf,
                            w_idx * sf : (w_idx + spatial_size[1]) * sf,
                        ] = out_patch
            if spatial_overlap > 0:
                output = E / W
            else:
                output = E
            self._update_frame_seq_output(output)
        else:
            # no spatial patching
            res = self.backend.run(input_data, **kwargs)
            # CANN returns list of lists, Mindspore returns just Tensor
            # in this case where input consists of one element, output list will contain also one element
            if isinstance(res, list):
                res = res[0]
                if isinstance(res, list):
                    # get the first output of the model, currently all models return just one tensor
                    res = res[0]
            if res.dtype != self.dtype:
                res = res.astype(self.dtype)
            if len(res.shape) == 4:
                # image model, T = 1
                res = np.expand_dims(res, axis=1)
            output = res
            self._update_frame_seq_output(output)

    def _update_frame_seq_output(self, update):
        if update is None:
            return None

        new_frames = update  # NTCHW

        if self._result is None:
            self._result = np.copy(new_frames)
        else:
            if self.frame_overlap == 0:
                self._result = np.concatenate((self._result, new_frames), axis=1)
            else:
                prev_frames = self._result[:, -self.frame_overlap :, :, :, :]
                if self.not_overlap_border:
                    to_zero = self.frame_overlap // 2
                    new_frames[:, :to_zero, ...] = prev_frames[:, :to_zero, ...]
                    prev_frames[:, -to_zero:, ...] = new_frames[:, -2 * to_zero : -to_zero, ...]
                self._result[:, -self.frame_overlap :, :, :, :] = (
                    new_frames[:, : self.frame_overlap, :, :, :].astype(np.float32) + prev_frames.astype(np.float32)
                ) / 2
                self._result = np.concatenate((self._result, new_frames[:, self.frame_overlap :, :, :, :]), axis=1)

    def get_result(self):
        if self._result is not None:
            output = self._result
            if self.frame_overlap > 0:
                self._result = self._result[:, -self.frame_overlap :, :, :, :]
                output = output[:, 0 : -self.frame_overlap, :, :, :]
            else:
                self._result = None
            # to support image models, for which T=1, reduce dimension by 1
            if self.frame_overlap == 0 and len(output.shape) == 5 and output.shape[1] == 1:
                output = np.squeeze(output, axis=1)
            return output

    def __call__(self, input_data, **kwargs):
        assert isinstance(input_data, list) is False
        temporal_size = self.frame_window_size
        # remember frame_overlap here to restore it before next video
        restore_ov = self.frame_overlap
        # print("Process video shape = ", input_data.shape)
        if temporal_size > 1 and temporal_size != input_data.shape[1]:
            temporal_overlap = self.frame_overlap
            d = input_data.shape[1]
            stride = temporal_size - temporal_overlap
            d_idx_list = list(range(0, d - temporal_size, stride)) + [max(0, d - temporal_size)]
            prev_d_idx = 0
            for d_idx in d_idx_list:
                lq_clip = input_data[:, d_idx : d_idx + temporal_size, ...]

                # the following code is nesessary to correctly handle the last frames window,
                # it should be shifted to the lest so that overlap will be equal to dif.
                dif = temporal_size + prev_d_idx - d_idx
                if dif > 0 and dif < temporal_size:
                    self.frame_overlap = dif
                prev_d_idx = d_idx

                self._process_clip(lq_clip, **kwargs)
        else:
            self._process_clip(input_data, **kwargs)
        # in the end of video, do not forget to set frame_overlap to 0,
        # so that get_results function could return all video frames.
        # then restore original frame_overlap for next videos
        self.frame_overlap = 0
        output = self.get_result()
        self.frame_overlap = restore_ov
        return output


class SimpleImageTiler:
    def __init__(self, backend, scale, target_input_index=0):
        self.backend = backend
        self.scale = scale
        self.target_input_index = target_input_index

    def __call__(self, input_data):
        if not isinstance(input_data, (list, tuple)):
            input_data = [input_data]
        elif isinstance(input_data, (list, tuple)) and len(input_data) and isinstance(input_data[0], (list, tuple)):
            input_data = input_data[0]
        tiles, actual_shapes = self.get_tiles(input_data)
        actual_shape = actual_shapes[self.target_input_index]
        model_outputs = []
        for tile in tiles:
            res = self.backend.run([tile])
            while isinstance(res, list):
                res = res[0]
            model_outputs.append(res)
        target_spatial_size = self.get_target_spatial_size(input_data[self.target_input_index])
        return self.make_image_from_tiles(model_outputs, actual_shape, target_spatial_size)

    def get_target_spatial_size(self, input_data):
        h, w = input_data.shape[-2:]
        return h * self.scale, w * self.scale

    def get_tiles(self, input_data):
        tiles = []
        actual_shapes = []
        for i, data in enumerate(input_data):
            shape = self.backend.get_input_shape(i)
            data_tiles, data_shape = self.split_onto_tiles(data, shape)
            tiles.append(data_tiles)
            actual_shapes.append(data_shape)
        assert len(tiles)
        tiles_ = []
        for i in range(len(tiles[0])):
            column = []
            for j in range(len(tiles)):
                column.append(tiles[j][i])
            tiles_.append(column)
        return tiles_, actual_shapes

    def split_onto_tiles(self, image, model_size):
        if len(image.shape) == 5:
            image = image[0]
        b, c, image_h, image_w = image.shape
        model_h, model_w = model_size[-2:]
        if image_h % model_h != 0 or image_w % model_w != 0:
            target_h = (image_h // model_h) * model_h + model_h
            target_w = (image_w // model_w) * model_w + model_w
            image_ = np.zeros((b, c, target_h, target_w), dtype=image.dtype)
            image_[:, :, :image_h, :image_w] = image
        else:
            image_ = image

        crops = []
        for i in range(0, image_h, model_h):
            for j in range(0, image_w, model_w):
                crop = image_[:, :, i : i + model_h, j : j + model_w]
                crops.append(crop)
        return crops, image_.shape

    def make_image_from_tiles(self, tiles, actual_size, target_size):
        target_height, target_width = target_size
        assert len(tiles)
        step_h, step_w = tiles[0].shape[-2:]
        b, c, h, w = actual_size
        h, w = h * self.scale, w * self.scale
        image = np.zeros((b, c, h, w), dtype=tiles[0].dtype)
        k = 0
        for i in range(0, target_height, step_h):
            for j in range(0, target_width, step_w):
                image[:, :, i : i + step_h, j : j + step_w] = tiles[k]
                k += 1
        return image[:, :, :target_height, :target_width]
