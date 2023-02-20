import argparse

import matplotlib.pyplot as plt
import numpy as np

from utils3d.mesh.io import read_mesh
from utils3d.render.nvisii import NViSIIRenderer
from utils3d.utils.utils import get_pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="../data/Squirrel_visual.obj",
        help="Input mesh path.",
    )
    args = parser.parse_args()

    NViSIIRenderer.init()
    renderer = NViSIIRenderer()

    mesh_pose_dict = {"mesh": (args.input, [1.0] * 3, np.eye(4))}

    mesh = read_mesh(args.input)
    center = mesh.bounds.mean(0)
    scale = np.sqrt(((mesh.bounds[1] - mesh.bounds[0]) ** 2).sum())
    camera_pose = get_pose(scale * 1, center=center, ax=np.pi / 3, az=np.pi / 3)
    # camera_pose = get_pose(scale * 1, center=center, ax=0, az=np.pi)
    light_pose = get_pose(scale * 2, center=center, ax=np.pi / 3, az=np.pi / 3)

    renderer.reset(camera_pose, light_pose)
    renderer.update_objects(mesh_pose_dict)
    img = renderer.render()

    fig, ax = plt.subplots(1, 1, dpi=150)
    ax.imshow(img)
    plt.show()

    NViSIIRenderer.deinit()


if __name__ == "__main__":
    main()
