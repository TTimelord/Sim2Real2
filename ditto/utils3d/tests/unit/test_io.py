import os
import tempfile

import pytest
import trimesh

from utils3d.mesh.io import read_mesh, write_mesh
from utils3d.pointcloud.io import read_pointcloud, write_pointcloud
from utils3d.pointcloud.utils import sample_pointcloud


@pytest.mark.parametrize("num_point", [1, 2048])
def test_pointcloud_io(num_point):
    xyz, color = read_pointcloud("data/pointcloud_color.pcd")
    xyz, color = sample_pointcloud(xyz, color, 2048)
    with tempfile.TemporaryDirectory() as tmpdirname:
        output = os.path.join(tmpdirname, "tmp.npy")
        write_pointcloud(xyz, output, color=color)

        output = os.path.join(tmpdirname, "tmp.pcd")
        write_pointcloud(xyz, output, color=color)

        output = os.path.join(tmpdirname, "tmp.txt")
        write_pointcloud(xyz, output, color=color)

        output = os.path.join(tmpdirname, "tmp.pts")
        write_pointcloud(xyz, output, color=color)


def test_mesh_io():
    mesh = read_mesh("data/Squirrel_visual.obj")
    with tempfile.TemporaryDirectory() as tmpdirname:
        output = os.path.join(tmpdirname, "tmp.stl")
        write_mesh(mesh, output)
