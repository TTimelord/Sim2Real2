import argparse

from utils3d.pointcloud.io import read_pointcloud
from utils3d.pointcloud.utils import normalize_pointcloud
from utils3d.pointcloud.visualization import (
    visualize_3d_point_cloud_mpl,
    visualize_3d_point_cloud_o3d,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tool", type=str, choices=["o3d", "mpl"], help="Which tool for visualization."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="../data/pointcloud_color.pcd",
        help="Input point cloud path.",
    )
    parser.add_argument("--norm", action="store_true", help="Normalize data.")
    args = parser.parse_args()

    xyz, color = read_pointcloud(args.input)

    # normalize point cloud
    if args.norm:
        xyz, _, _ = normalize_pointcloud(xyz)

    if args.tool == "o3d":
        visualize_3d_point_cloud_o3d(xyz, color=color)
    else:
        visualize_3d_point_cloud_mpl(xyz, color=color)


if __name__ == "__main__":
    main()
