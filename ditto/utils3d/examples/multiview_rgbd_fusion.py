import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from utils3d.pointcloud.visualization import visualize_3d_point_cloud_o3d
from utils3d.rgbd.io import read_rgbd
from utils3d.rgbd.utils import tsdf_fusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="../data/rgbd",
        help="Input rgbd path.",
    )

    parser.add_argument(
        "-t",
        "--transform",
        type=str,
        default="../data/rgbd/transforms.json",
        help="Input transformation matrix.",
    )

    args = parser.parse_args()

    rgb_path_dict = {}
    depth_path_dict = {}

    with open(args.transform) as f:
        transform_dict = json.load(f)

    rgbd_list = []
    transform_list = []

    for idx in range(10):
        # rgb_p = f'../data/rgbd/r_{idx}.png'
        # depth_p = f'../data/rgbd/r_{idx}_depth_0019.png'
        rgb_p = f"/home/zhenyu/Downloads/real_music_box/imgs_1/{idx}_color.png"
        depth_p = f"/home/zhenyu/Downloads/real_music_box/imgs_1/{idx}_depth.png"
        rgbd_list.append(read_rgbd(rgb_p, depth_p, depth_scale=1000, depth_trunc=1))
        transform_list.append(np.array(transform_dict["frames"][idx]["transform_matrix"]))
    # print(np.unique(np.asanyarray(rgbd_list[0].depth)))
    # compute intrinsics
    camera_angle_x = transform_dict["camera_angle_x"]
    img = Image.open(rgb_p)
    w, h = img.size
    fx = w / 2.0 / np.tan(camera_angle_x / 2)
    fy = h / 2.0 / np.tan(camera_angle_x / 2) * (w / h)
    intrinsics = {"height": h, "width": w, "fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2}

    # fusion
    tsdf_vol = tsdf_fusion(rgbd_list, transform_list, intrinsics)
    # verts, faces, norms, colors = tsdf_vol.get_mesh()
    point_cloud = tsdf_vol.get_point_cloud()

    visualize_3d_point_cloud_o3d(point_cloud[:, :3], point_cloud[:, 3:6])


if __name__ == "__main__":
    main()
