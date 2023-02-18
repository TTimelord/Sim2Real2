"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-01-12
 @desc
"""

import trimesh


def read_mesh(path):
    return trimesh.load(path)


def write_mesh(mesh, path):
    mesh.export(path)
