"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-01-12
 @desc Point cloud utilities
"""

import numpy as np
import open3d as o3d


def assert_array_shape(xyz, shapes=((-1, 3),)):
    """check array shape

    Args:
        xyz (np.ndarray): array
        shape (tuple of tuple of ints, optional): possible target shapes, -1 means arbitrary. Defaults to ((-1, 3)).

    Raises:
        ValueError.
    """
    flags = {x: True for x in range(len(shapes))}
    for idx, shape in enumerate(shapes):
        if len(xyz.shape) != len(shape):
            flags[idx] = False

        for dim, num in enumerate(shape):
            if num == -1:
                continue
            elif xyz.shape[dim] != num:
                flags[idx] = False
    if sum(flags.values()) == 0:  # None of the possible shape works
        raise ValueError(f"Input array {xyz.shape} is not in target shapes {shapes}!")


def np_to_o3d_pointcloud(xyz, color=None):
    """convert numpy array to open3d point cloud

    Args:
        xyz (np.ndarray): point cloud
        color (np.ndarray, optional): colors of input point cloud. Can be N*3 or 3. Defaults to None.

    Returns:
        o3d.geometry.PointCloud: open3d point cloud
    """
    assert_array_shape(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # add color
    if color is not None:
        if len(color.shape) == 2:
            # same number of colors as the points
            assert_array_shape(color, shapes=(xyz.shape,))
            pcd.colors = o3d.utility.Vector3dVector(color)
        elif len(color.shape) == 1:
            # N*3
            color = np.tile(color, (xyz.shape[0], 1))
            assert_array_shape(color, shapes=(xyz.shape,))
            pcd.colors = o3d.utility.Vector3dVector(color)
        else:
            raise ValueError(f"Bad color with shape {color.shape}")

    return pcd


def normalize_pointcloud(xyz, padding=0.0):
    """normalize point cloud to [-0.5, 0.5]

    Args:
        xyz (np.ndarray): input point cloud, N*3
        padding (float, optional): padding. Defaults to 0.0.

    Returns:
        np.ndarray: normalized point cloud
        np.ndarray: original center
        float: original scale
    """
    assert_array_shape(xyz)
    bound_max = xyz.max(0)
    bound_min = xyz.min(0)
    center = (bound_max + bound_min) / 2
    scale = (bound_max - bound_min).max()
    scale = scale * (1 + padding)
    normalized_xyz = (xyz - center) / scale
    return normalized_xyz, center, scale


def sample_pointcloud(xyz, color=None, num_points=2048):
    """random subsample point cloud

    Args:
        xyz (np.ndarray): input point cloud of N*3.
        color (np.ndarray, optional): color of the points, N*3 or None. Defaults to None.
        num_points (int, optional): number of subsampled point cloud. Defaults to 2048.

    Returns:
        np.ndarray: subsampled point cloud
    """
    assert_array_shape(xyz)
    replace = num_points > xyz.shape[0]
    sample_idx = np.random.choice(np.arange(xyz.shape[0]), size=(num_points,), replace=replace)

    if color is not None:
        assert_array_shape(color, shapes=(xyz.shape,))
        color = color[sample_idx]
    xyz = xyz[sample_idx]
    return xyz, color
