"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-01-11
 @desc Point cloud interactive visualization utils
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D

from utils3d.pointcloud.utils import assert_array_shape, np_to_o3d_pointcloud


def visualize_3d_point_cloud_mpl(
    xyz,
    show=True,
    show_axis=True,
    in_u_sphere=False,
    marker=".",
    s=8,
    alpha=0.8,
    figsize=(5, 5),
    elev=10,
    azim=240,
    axis=None,
    title=None,
    lim=None,
    *args,
    **kwargs,
):
    """Visualize 3d point cloud with matplotlib.

    Args:
        xyz (np.ndarray): input point cloud as N*3 numpy array.
        show (bool, optional): show current plot. Defaults to True.
        show_axis (bool, optional): show axis grid. Defaults to True.
        in_u_sphere (bool, optional): plot in unit sphere. Defaults to False.
        marker (str, optional): marker shape. Defaults to '.'.
        s (int, optional): point size. Defaults to 8.
        alpha (float, optional): point transparency. Defaults to .8.
        figsize (tuple, optional): figure size. Defaults to (5, 5).
        elev (int, optional): elevation of view angle. Defaults to 10.
        azim (int, optional): azimuth of view angle. Defaults to 240.
        axis (mpl_toolkits.mplot3d.axes3d.Axes3D, optional): plot on given axis instead of creating a new. Defaults to None.
        title (str of None, optional): plot title. Defaults to None.
        lim (list of tuples of floats, optional): limits in three dimension. Defaults to None.

    Returns:
        matplotlib.figure.Figure: axis or figure where the point cloud is plotted on.
    """

    assert_array_shape(xyz)
    # extract x y z from input points
    x = xyz.T[0]
    y = xyz.T[1]
    z = xyz.T[2]

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if lim:
        ax.set_xlim3d(*lim[0])
        ax.set_ylim3d(*lim[1])
        ax.set_zlim3d(*lim[2])
    elif in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        lim = (min(np.min(x), np.min(y), np.min(z)), max(np.max(x), np.max(y), np.max(z)))
        ax.set_xlim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_ylim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_zlim(1.3 * lim[0], 1.3 * lim[1])
        plt.tight_layout()

    if not show_axis:
        plt.axis("off")

    if show:
        plt.show()

    return fig


def visualize_3d_point_cloud_o3d(xyz, color=None, **kwargs):
    """Visualize 3d point cloud with matplotlib.

    Args:
        xyz (np.ndarray): input point cloud as N*3 numpy array.
        color (np.ndarray, optional): colors of input point cloud. Can be N*3 or 3. Defaults to None.
    """
    pcd = np_to_o3d_pointcloud(xyz, color)
    o3d.visualization.draw_geometries([pcd], **kwargs)
