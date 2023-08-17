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

from mindediting.models.common.grid_sample import grid_sample_2d


def ms_broadcast_to_match(a, b, ignore_axes=0):
    """Returns (a', b') which are the inputs broadcast up to have the same shape.

    Suppose you want to apply an operation to tensors a and b but it doesn't
    support broadcasting. As an example maybe we have tensors of these shapes:
      a    [5, 1, 3, 4]
      b [2, 1, 8, 4, 2]
    Considering the last two dimensions as matrices, we may want to multiply
    a by b to get a tensor [2, 5, 8, 3, 2] of (2x3) matrices. However, tf.matmul
    doesn't support this because the outer dimensions don't match. Calling
    tf.matmul(a, b) directly will fail.

    However, the dimensions do match under broadcasting, so we can do the
    multiplication like this:
      a, b = broadcast_to_match(a, b, ignore_axes=2)
      c = tf.matmul(a, b)
    The ignore_axes parameter tells us to ignore the last two dimensions of a
    and b and just make the rest match.

    Args:
      a: Any shape
      b: Any shape
      ignore_axes: If present, broadcasting will not apply to the final this many
        axes. For example, if you are planning to call tf.matmul(a, b) on the
        result, then set ignore_axes=2 because tf.matmul operates on the last two
        axes, only the rest need to match. To ignore a different number of axes
        for inputs a and b, pass a pair of number to ignore_axes.

    Returns:
      a', b': Identical to the two inputs except tiled so that the shapes
          match. See https://www.tensorflow.org/performance/xla/broadcasting.
          If the shapes already match, no tensorflow graph operations are added,
          so this is cheap.
    """
    # a = torch.tensor(a)
    # b = torch.tensor(b)
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    # Extract the part of the shape that is required to match.
    if isinstance(ignore_axes, tuple) or isinstance(ignore_axes, list):
        ignore_a = ignore_axes[0]
        ignore_b = ignore_axes[1]
    else:
        ignore_a = ignore_axes
        ignore_b = ignore_axes
    if ignore_a:
        a_shape = a_shape[:-ignore_a]
    if ignore_b:
        b_shape = b_shape[:-ignore_b]
    if a_shape == b_shape:
        return (a, b)
    # Addition supports broadcasting, so add a tensor of zeroes.
    za = ms.ops.Zeros()(tuple(a_shape + [1] * ignore_b), ms.float32)  # .to(b)
    zb = ms.ops.Zeros()(tuple(b_shape + [1] * ignore_a), ms.float32)  # .to(a)
    # print("a+zb: ", a.shape, zb.shape)
    a = a + zb
    b = b + za

    a_new_shape = list(a.shape)
    b_new_shape = list(b.shape)
    if ignore_a:
        a_new_shape = a_new_shape[:-ignore_a]
    if ignore_b:
        b_new_shape = b_new_shape[:-ignore_b]
    assert a_new_shape == b_new_shape
    return (a, b)


def ms_broadcasting_matmul(a, b):
    (a, b) = ms_broadcast_to_match(a, b, ignore_axes=2)
    return ms.ops.matmul(a, b)


def check_input_shape(name, tensor, axis, value):
    """Utility function for checking tensor shapes."""
    shape = list(tensor.shape)
    if shape[axis] != value:
        raise ValueError('Input "%s": dimension %d should be %s. Shape = %s' % (name, axis, value, shape))
    return


def check_input_m34(name, tensor):
    check_input_shape(name, tensor, -1, 4)
    check_input_shape(name, tensor, -2, 3)
    return


def ms_mat34_pose_inverse(matrix):
    """Invert a 3x4 matrix.

    Args:
      matrix: [..., 3, 4] matrix where [..., 3, 3] is a rotation matrix

    Returns:
      The inverse matrix, of the same shape as the input. It is computed as
      if we added an extra row with values [0, 0, 0, 1], inverted the
      matrix, and removed the row again.

    Raises:
      ValueError: if input is not a 3x4 matrix.
    """
    check_input_m34("matrix", matrix)
    rest = matrix[..., :3]
    translation = matrix[..., 3:]
    inverse = rest.swapaxes(-2, -1)
    inverse_translation = -ms.ops.matmul(inverse, translation)
    return ms.ops.concat([inverse, inverse_translation], axis=-1)


def ms_mat34_product(a, b):
    """Returns the product of a and b, 3x4 matrices.

    Args:
      a: [..., 3, 4] matrix
      b: [..., 3, 4] matrix

    Returns:
      The product ab. The product is computed as if we added an extra row
      [0, 0, 0, 1] to each matrix, multiplied them, and then removed the extra
      row. The shapes of a and b must match, either directly or via
      broadcasting.

    Raises:
      ValueError: if a or b are not 3x4 matrices.
    """
    check_input_m34("a", a)
    check_input_m34("b", b)

    # print("in product: ", a.shape, b.shape)
    (a, b) = ms_broadcast_to_match(a, b, ignore_axes=2)
    # Split translation part off from the rest
    a33 = a[..., :3]
    a_translate = a[..., 3:]
    b33 = b[..., :3]
    b_translate = b[..., 3:]
    # Compute parts of the product
    ab33 = ms.ops.matmul(a33, b33)
    ab_translate = a_translate + ms.ops.matmul(a33, b_translate)
    # Assemble
    return ms.ops.concat([ab33, ab_translate], axis=-1)


def ms_build_matrix(elements):
    """Stacks elements along two axes to make a tensor of matrices.

    Args:
      elements: [n, m] matrix of tensors, each with shape [...].

    Returns:
      [..., n, m] tensor of matrices, resulting from concatenating
        the individual tensors.
    """
    rows = [ms.ops.stack(row_elements, axis=-1) for row_elements in elements]
    return ms.ops.stack(rows, axis=-2)


def ms_intrinsics_matrix_spaces(intrinsics):
    """Make a matrix mapping camera space to homogeneous texture coords.

    Args:
      intrinsics: [..., 4] intrinsics. Last dimension (fx, fy, cx, cy)

    Returns:
      [..., 3, 3] matrix mapping camera space to image space.
    """
    fx = intrinsics[..., 0]
    fy = intrinsics[..., 1]
    cx = intrinsics[..., 2]
    cy = intrinsics[..., 3]
    zero = ms.ops.ZerosLike()(fx)
    one = ms.ops.OnesLike()(fx)
    return ms_build_matrix([[fx, zero, cx], [zero, fy, cy], [zero, zero, one]])


def ms_inverse_intrinsics_matrix_spaces(intrinsics):
    """Return the inverse of the intrinsics matrix..

    Args:
      intrinsics: [..., 4] intrinsics. Last dimension (fx, fy, cx, cy)

    Returns:
      [..., 3, 3] matrix mapping homogeneous texture coords to camera space.
    """
    fxi = 1.0 / intrinsics[..., 0]
    fyi = 1.0 / intrinsics[..., 1]
    cx = intrinsics[..., 2]
    cy = intrinsics[..., 3]
    zero = ms.ops.ZerosLike()(cx)
    one = ms.ops.OnesLike()(cx)
    return ms_build_matrix([[fxi, zero, -cx * fxi], [zero, fyi, -cy * fyi], [zero, zero, one]])


def ms_get_homographies(pose, intrinsics, ref_pose, ref_intrinsics, depths, dataset):

    size1 = list(pose.shape[:-2]) + [len(depths)] + [1, 1]
    size2 = list(pose.shape[:-2]) + [1, 1, 1]
    pose = pose[..., None, :, :]  # [..., 1, 3, 4]
    intrinsics = intrinsics[..., None, :]  # [..., 1, 4]
    ref_pose = ref_pose[..., None, :, :]  # [..., 1, 3, 4]
    ref_intrinsics = ref_intrinsics[..., None, :]  # [..., 1, 4]

    rel_pose = ms_mat34_product(ms_mat34_pose_inverse(pose), ref_pose)
    R = rel_pose[..., :3]
    t = rel_pose[..., 3:]

    n = ms.ops.tile(ms.Tensor([0.0, 0.0, 1.0]), tuple(size1))
    d = ms.ops.tile(-depths[..., None, None], tuple(size2))
    K = ms_intrinsics_matrix_spaces(intrinsics)
    K_ref = ms_inverse_intrinsics_matrix_spaces(ref_intrinsics)

    H = ms_broadcasting_matmul(K, ms_broadcasting_matmul(R - ms.ops.Div()(ms_broadcasting_matmul(t, n), d), K_ref))

    return H


def ms_pixel_center_grid(corners, upfactor):
    """Produce a grid of (x,y) texture-coordinate pairs of pixel centers.

    Args:
      height: (integer) height, not a tensor
      width: (integer) width, not a tensor

    Returns:
      A tensor of shape [height, width, 2] where each entry gives the (x,y)
      texture coordinates of the corresponding pixel center. For example, for
      pixel_center_grid(2, 3) the result is:
         [[[1/6, 1/4], [3/6, 1/4], [5/6, 1/4]],
          [[1/6, 3/4], [3/6, 3/4], [5/6, 3/4]]]
    """
    h0, h1, w0, w1 = corners
    H = int((h1 - h0) * upfactor)
    W = int((w1 - w0) * upfactor)
    ys = ms.ops.tile(
        ms.ops.linspace(ms.Tensor(h0 + 0.5 / upfactor), ms.Tensor(h1 - 0.5 / upfactor), H).reshape(-1, 1), (1, W)
    )
    xs = ms.ops.tile(
        ms.ops.linspace(ms.Tensor(w0 + 0.5 / upfactor), ms.Tensor(w1 - 0.5 / upfactor), W).reshape(1, -1), (H, 1)
    )

    grid = ms.ops.stack((xs, ys), axis=-1)
    assert list(grid.shape) == [H, W, 2]
    return grid


def ms_homogenize(coords):
    """Convert (x, y) to (x, y, 1), or (x, y, z) to (x, y, z, 1)."""
    ones = ms.ops.OnesLike()(coords[..., :1])
    return ms.ops.concat([coords, ones], axis=-1)


def ms_dehomogenize(coords):
    """Convert (x, y, w) to (x/w, y/w) or (x, y, z, w) to (x/w, y/w, z/w)."""
    # return np.nan_to_num(coords[..., :-1] / coords[..., -1:], nan=0.0, posinf=0.0, neginf=0.0)
    return coords[..., :-1] / coords[..., -1:]


def ms_collapse_dim(tensor, axis):
    """Collapses one axis of a tensor into the preceding axis.

    This is a fast operation since it just involves reshaping the
    tensor.

    Example:
      a = [[[1,2], [3,4]], [[5,6], [7,8]]]

      collapse_dim(a, -1) = [[1,2,3,4], [5,6,7,8]]
      collapse_dim(a, 1) = [[1,2], [3,4], [5,6], [7,8]]

    Args:
      tensor: a tensor of shape [..., Di-1, Di, ...]
      axis: the axis to collapse, i, in the range (-n, n). The first axis may not
        be collapsed.

    Returns:
      a tensor of shape [..., Di-1 * Di, ...] containing the same values.
    """
    # tensor = torch.tensor(tensor)
    shape = list(tensor.shape)
    # We want to extract the parts of the shape that should remain unchanged.
    # Naively one would write shape[:axis-1] or shape[axis+1:] for this, but
    # this will be wrong if, for example, axis is -1. So the safe way is to
    # first slice using [:axis] or [axis:] and then remove an additional element.
    newshape = shape[:axis][:-1] + [-1] + shape[axis:][1:]
    return tensor.reshape(newshape)


def ms_split_dim(tensor, axis, factor):
    """Splits a dimension into two dimensions.

    Opposite of collapse_dim.

    Args:
      tensor: an n-dimensional tensor of shape [..., Di, ...]
      axis: the axis to split, i, in the range [-n, n)
      factor: the size of the first of the two resulting axes. Must divide Di.

    Returns:
      an (n+1)-dimensional tensor of shape [..., factor, Di / factor, ...]
      containing the same values as the input tensor.
    """
    # tensor = torch.tensor(tensor)
    shape = list(tensor.shape)
    newshape = shape[:axis] + [factor, shape[axis] // factor] + shape[axis:][1:]
    return tensor.reshape(newshape)


def ms_apply_homography(homography, coords, corners):
    """Transform grid of (x,y) texture coordinates by a homography.

    Args:
      homography: [..., 3, 3]
      coords: [..., H, W, 2] (x,y) texture coordinates

    Returns:
      [..., H, W, 2] transformed coordinates.
    """
    height = coords.shape[-3]
    coords = ms_homogenize(ms_collapse_dim(coords, -2))  # [..., H*W, 3]
    # Instead of transposing the coords, transpose the homography and
    # swap the order of multiplication.
    coords = ms_broadcasting_matmul(coords, homography.swapaxes(-2, -1))
    # coords is now [..., H*W, 3]
    coords = ms_split_dim(ms_dehomogenize(coords), -2, height)

    h0, h1, w0, w1 = corners
    coords -= ms.Tensor([w0, h0], ms.float32)
    coords /= ms.Tensor([w1 - w0, h1 - h0], ms.float32)
    return coords


def ms_sample_image(image, coords):
    """Sample points from an image, using bilinear filtering.

    Args:
      image: [B0, ..., Bn-1, channels, height, width] image data
      coords: [B0, ..., Bn-1, ..., 2] (x,y) texture coordinates
      clamp: if True, coordinates are clamped to the coordinates of the corner
        pixels -- i.e. minimum value 0.5/width, 0.5/height and maximum value
        1.0-0.5/width or 1.0-0.5/height. This is equivalent to extending the image
        in all directions by copying its edge pixels. If False, sampling values
        outside the image will return 0 values.

    Returns:
      [B0, ..., Bn-1, ..., channels, h, w] image data, in which each value is sampled
      with bilinear interpolation from the image at position indicated by the
      (x,y) texture coordinates. The image and coords parameters must have
      matching batch dimensions B0, ..., Bn-1.

    Raises:
      ValueError: if shapes are incompatible.
    """
    # check_input_shape('coords', coords, -1, 2)
    coords = 2.0 * coords - 1.0

    # ms.ops.grid_sample expects image to be [batch, channels, height, width] and
    # pixel_coords to be [batch, ..., 2]. So we need to reshape, perform the
    # resampling, and then reshape back to what we had.
    image_shape = image.shape
    coord_shape = coords.shape
    batched_image = image.view(-1, *image_shape[-3:])
    batched_coords = coords.view(-1, *coord_shape[-3:])

    resampled = grid_sample_2d(
        batched_image, batched_coords, interpolation_mode="bilinear", align_corners=False, padding_mode="zeros"
    )

    # Convert back to the right shape to return
    resampled_shape = image_shape[:-2] + coord_shape[-3:-1]
    resampled = resampled.view(*resampled_shape)

    return resampled
