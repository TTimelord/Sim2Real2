"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-01-11
 @desc Point cloud IO
"""

import os

import numpy as np
import open3d as o3d
from numpy.lib import shape_base

from utils3d.pointcloud.utils import assert_array_shape, np_to_o3d_pointcloud


def read_pointcloud(path):
    """read point cloud

    Args:
        path (str): path to point cloud file

    Raises:
        NotImplementedError: if the loader for certain type of file is not implemented

    Returns:
        np.ndarry: loaded point cloud.
        np.ndarry or None: loaded color.
    """
    _, ext = os.path.splitext(path)
    ext = ext[1:]  # remove .
    if ext in ["txt", "pts"]:
        xyz = np.loadtxt(path)
    elif ext in ["npy"]:
        xyz = np.load(path)
    elif ext in ["ply", "pcd"]:
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points)
        if pcd.colors is not None:
            color = np.asarray(pcd.colors)
            xyz = np.concatenate((xyz, color), axis=1)
    else:
        raise NotImplementedError(f"No loader for {ext} files. Can't load {path}")
    assert_array_shape(xyz, shapes=((-1, 3), (-1, 6)))

    if xyz.shape[1] > 3:  # with color
        color = xyz[:, 3:]
        xyz = xyz[:, :3]
    else:
        color = None

    return xyz, color


def write_pointcloud(xyz, path, color=None):
    """save point cloud

    Args:
        xyz (np.ndarray): point cloud to save, N*3 or N*6
        path (str): path to point cloud file

    Raises:
        NotImplementedError: if the saver for certain type of file is not implemented

    """
    if color is not None:
        assert_array_shape(color, shapes=((xyz.shape,)))
        xyz = np.concatenate((xyz, color), axis=1)

    _, ext = os.path.splitext(path)
    ext = ext[1:]  # remove .
    if ext in ["txt", "pts"]:
        np.savetxt(path, xyz)
    elif ext in ["npy"]:
        np.save(path, xyz)
    elif ext in ["ply", "pcd"]:
        if xyz.shape[1] == 6:
            color = xyz[:, 3:]
            xyz = xyz[:, :3]
        else:
            color = None
        pcd = np_to_o3d_pointcloud(xyz, color=color)
        o3d.io.write_point_cloud(path, pcd)
    else:
        raise NotImplementedError(f"No saver for {ext} files. Can't save {path}")
