"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-01-14
 @desc
"""

import numpy as np

from utils3d.utils.transform import Rotation, Transform


def get_pose(distance, center=np.zeros(3), ax=0, ay=0, az=0):
    """generate camera pose from distance, center and euler angles

    Args:
        distance (float): distance from camera to center
        center (np.ndarray, optional): the look at center. Defaults to np.zeros(3).
        ax (float, optional): euler angle x. Defaults to 0.
        ay (float, optional): euler angle around y axis. Defaults to 0.
        az (float, optional): euler angle around z axis. Defaults to 0.

    Returns:
        np.ndarray: camera pose of 4*4 numpy array
    """
    rotation = Rotation.from_euler("xyz", (ax, ay, az))
    vec = np.array([0, 0, distance])
    translation = rotation.as_matrix().dot(vec) + center
    camera_pose = Transform(rotation, translation).as_matrix()
    return camera_pose
