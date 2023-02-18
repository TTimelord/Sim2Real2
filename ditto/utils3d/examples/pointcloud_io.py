import argparse

import numpy as np

from utils3d.pointcloud.io import read_pointcloud, write_pointcloud
from utils3d.pointcloud.utils import sample_pointcloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="../data/pointcloud_color.pcd",
        help="Input point cloud path.",
    )
    parser.add_argument("-o", "--output", type=str, help="Output point cloud path.")
    args = parser.parse_args()

    xyz, color = read_pointcloud(args.input)
    xyz, color = sample_pointcloud(xyz, color, 2048)

    print(f"Saving point cloud of shape {xyz.shape} to {args.output}")
    write_pointcloud(xyz, args.output, color=color)


if __name__ == "__main__":
    main()
