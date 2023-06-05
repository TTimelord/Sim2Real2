"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-03-03
 @desc RGBD IO
"""

import os

import numpy as np
import open3d as o3d
from PIL import Image


def read_rgbd(rgb_path, depth_path, depth_scale=1000, depth_trunc=1):
    """read rgbd images

    Args:
        rgb_path (str): path to rgb image
        depth_path (str): path to depth image
        depth_scale (float, optional): The scale of depth, multiplicative. Defaults to 0.001.
        depth_trunc (float, optional): The truncate scale of depth. Defaults to 0.7.

    Returns:
        o3d.geometry.RGBDImage: Merged RGBD Image
    """
    depth_img = np.array(Image.open(depth_path))
    depth_img[depth_img == depth_img.max()] == 0
    depth_img = depth_img.astype(np.float32)
    if len(depth_img.shape) == 3:
        depth_img = np.ascontiguousarray(depth_img[:, :, 0])

    rgb_img = np.array(Image.open(rgb_path))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_img),
        o3d.geometry.Image(depth_img),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    return rgbd
