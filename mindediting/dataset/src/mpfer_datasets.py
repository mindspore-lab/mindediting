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

import copy
import json
import os
import warnings

import cv2
import mindspore.dataset as ds
import numpy as np
import transformations


def _PoseFromViewDict(view_json):
    """Fills the world from camera transform from the view_json.

    Args:
        view_json: A dictionary of view parameters.

    Returns:
         A 4x4 transform matrix representing the world from camera transform.
    """

    # The camera model transforms the 3d point X into a ray u in the local
    # coordinate system:
    #
    #    u = R * (X[0:2] - X[3] * c)
    #
    # Meaning the world from camera transform is [inv(R), c]

    transform = np.identity(4)
    position = view_json["position"]
    transform[0:3, 3] = (position[0], position[1], position[2])
    orientation = view_json["orientation"]
    angle_axis = np.array([orientation[0], orientation[1], orientation[2]])
    angle = np.linalg.norm(angle_axis)
    epsilon = 1e-7
    if abs(angle) < epsilon:
        # No rotation
        return transform[:3]

    axis = angle_axis / angle
    rot_mat = transformations.quaternion_matrix(transformations.quaternion_about_axis(-angle, axis))
    transform[0:3, 0:3] = rot_mat[0:3, 0:3]
    return np.array(transform[:3])


def _IntrinsicsFromViewDict(view_json):
    """Fills the intrinsics matrix from view_json.

    Args:
        view_json: Dict view parameters.

    Returns:
         A 3x3 matrix representing the camera intrinsics.
    """
    intrinsics = []
    intrinsics.append(view_json["focal_length"])
    intrinsics.append(view_json["focal_length"] * view_json["pixel_aspect_ratio"])
    intrinsics.append(view_json["principal_point"][0])
    intrinsics.append(view_json["principal_point"][1])
    intrinsics = np.array(intrinsics)

    return intrinsics


def ReadView(base_dir, view_json):
    image_path = os.path.join(base_dir, view_json["relative_path"])
    intrinsics = _IntrinsicsFromViewDict(view_json)
    pose = _PoseFromViewDict(view_json)
    view = dict()
    view["image_path"] = image_path
    view["intrinsics"] = intrinsics
    view["pose"] = pose
    view["height"] = int(view_json["height"])
    view["width"] = int(view_json["width"])

    # Fix intrinsics for scenes 9 and 68
    if int(view_json["width"]) == 2048:
        view["intrinsics"] = view["intrinsics"] * 800 / 2048
        view["height"] = round(view["height"] * 800 / 2048)
        view["width"] = round(view["width"] * 800 / 2048)

    return view


def ReadScene(base_dir):
    """Reads a scene from the directory base_dir."""
    with open(os.path.join(base_dir, "models.json")) as f:
        model_json = json.load(f)

    all_views = []
    for views in model_json:
        all_views.append([ReadView(base_dir, view_json) for view_json in views])
    return all_views


def ReadImage(image_path, height):

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if height > img.shape[0]:
        img = cv2.copyMakeBorder(img, 0, height - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, 0)
    else:
        img = img[:height, :, :]

    return np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0


def add_read_shot_noise(imgs, log_sig_read, log_sig_shot, conditional):
    sig_read = 10**log_sig_read
    sig_shot = 10**log_sig_shot

    std = (imgs * (sig_shot**2) + (sig_read**2)) ** 0.5
    noisy = imgs + np.random.randn(*imgs.shape) * std
    noisy_std = (np.clip(noisy, 0.0, 1.0) * (sig_shot**2) + (sig_read**2)) ** 0.5

    input_imgs = np.concatenate([noisy, noisy_std], axis=1) if conditional else noisy
    return input_imgs


class Space_Dataset(object):
    def __init__(
        self,
        data_path,
        valid_scenes,
        position,
        input_indices,
        ref_height,
        log_sig_read,
        log_sig_shot,
        version,
        load_path,
        **kwargs,
    ):
        def load_params(params_filename):
            with open(params_filename) as file:
                params = json.load(file)
            return params

        params = load_params(f"{load_path}/{version}_ms/params.json")

        self.log_sig_read = log_sig_read
        self.log_sig_shot = log_sig_shot
        self.params = params
        self.ref_height = ref_height
        self.all_views = [ReadScene(f"{data_path}/data/scene_{scene:03}") for scene in valid_scenes]
        self.valid_scenes = valid_scenes
        self.position = position
        self.input_indices = input_indices

    def __getitem__(self, index):
        input_views = [self.all_views[index][self.position][i] for i in self.input_indices]

        imgs = np.stack([ReadImage(view["image_path"], self.ref_height) for view in input_views])
        heights = np.stack([view["height"] for view in input_views])
        poses = np.stack([view["pose"] for view in input_views])
        intrinsics = np.stack([view["intrinsics"] for view in input_views])

        input_imgs = add_read_shot_noise(imgs, self.log_sig_read, self.log_sig_shot, True)
        scene = self.valid_scenes[index]

        ref_idx = self.params["ref_idx"]
        fov_factor = self.params["fov_factor"]

        ref_pose = copy.deepcopy(self.all_views[index][self.position][ref_idx]["pose"])
        ref_intrinsics = copy.deepcopy(self.all_views[index][self.position][ref_idx]["intrinsics"])
        ref_intrinsics[0] *= fov_factor
        ref_intrinsics[1] *= fov_factor

        far_depth = self.params["far_depth"]
        mid_depth = self.params["mid_depth"]
        near_depth = self.params["near_depth"]
        num_planes_far = self.params["num_planes_far"]
        num_planes_near = self.params["num_planes_near"]

        disparities = np.linspace(1.0 / far_depth, 1.0 / mid_depth, num_planes_far)
        depths = 1.0 / disparities
        if num_planes_near > 0:
            disparities_near = np.linspace(1.0 / mid_depth, 1.0 / near_depth, num_planes_near + 1)
            depths_near = 1.0 / disparities_near
            depths = np.concatenate([depths, depths_near[1:]])

        return input_imgs, scene, imgs, heights, poses, intrinsics, ref_pose, ref_intrinsics, depths

    def __len__(self):
        return len(self.valid_scenes)


def create_space_dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=1,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    **kwargs,
):
    result = ""
    # Iterating over the keys of the Python kwargs dictionary
    for arg in kwargs:
        result += arg + " "
    source = Space_Dataset(dataset_dir, **kwargs)
    dataset = ds.GeneratorDataset(
        source=source,
        column_names=[
            "input_imgs",
            "scene",
            "imgs",
            "heights",
            "poses",
            "intrinsics",
            "ref_pose",
            "ref_intrinsics",
            "depths",
        ],
        sampler=sampler,
        num_samples=num_samples,
        num_parallel_workers=num_parallel_workers,
        num_shards=num_shards,
        shard_id=shard_id,
        shuffle=shuffle,
    )

    return dataset
