"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-03-03
 @desc RGBD utils
"""
from bdb import set_trace

import numpy as np
import open3d as o3d

from utils3d.rgbd.fusion import TSDFVolume


def backproject_rgbd(rgbd, intrinsics):
    """back project rgbd to point cloud

    Args:
        rgbd (o3d.geometry.RGBDImage): RGBD Image
        intrinsics (dict): intrinsics with keys: width, height, fx, fy, cx, cy

    Returns:
        o3d.geometry.PointCloud: back projected point cloud
    """
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(**intrinsics)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics_o3d)
    return pcd


def tsdf_fusion(rgbd_list, transform_list, intrinsics, bounds=None, voxel_size=0.0):
    """fuse multiview rgbd into tsdsf

    Args:
        rgbd_list (list[o3d.geometry.RGBDImage]): list of RGBD images
        transform_list (list[np.ndarray]): list of transformations (4X4)
        intrinsics (dict): _description_
        bounds (np.ndarray, optional): scene boundary, 3X2. Defaults to None.
        voxel_size (float, optional): size of voxel. Defaults to 0.

    Returns:
        TSDFVolumn: tsdf volumn
    """
    if bounds is None:
        # automatic determine bounds
        pc_list = []
        for rgbd, transform in zip(rgbd_list, transform_list):
            pcd = backproject_rgbd(rgbd, intrinsics)
            pcd.transform(transform)
            pc_list.append(np.asarray(pcd.points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(pc_list, axis=0))
        bounds = np.stack((pcd.get_min_bound(), pcd.get_max_bound()), axis=1)
    # automatic determine voxel size
    if voxel_size == 0:
        voxel_size = (bounds[:, 1] - bounds[:, 0]).max() / 100
    tsdf = TSDFVolume(bounds, voxel_size, use_gpu=True)

    intrinsics_mat = np.eye(3)
    intrinsics_mat[0, 0] = intrinsics["fx"]
    intrinsics_mat[1, 1] = intrinsics["fy"]
    intrinsics_mat[0, 2] = intrinsics["cx"]
    intrinsics_mat[1, 2] = intrinsics["cy"]
    for rgbd, transform in zip(rgbd_list, transform_list):
        rgb = rgbd.color
        depth = rgbd.depth
        tsdf.integrate(np.asarray(rgb), np.asarray(depth), intrinsics_mat, transform)
    return tsdf
