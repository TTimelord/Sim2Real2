import os

# import torch
from tqdm import tqdm

import open3d as o3d

from env import Env
from camera import Camera
from utils import get_actor_meshes_visual


def get_mesh_list(env: Env):
    mesh_list = []
    movable_links = []
    for j in env.active_joints:
        movable_links.append(j.get_child_link())

    # for our objects we can find base_link simply:
    base_link = env.active_joints[0].get_parent_link()
    mesh = get_actor_meshes_visual(base_link)
    mesh_list.append(mesh)

    # movable_links
    for link in movable_links:
        mesh = get_actor_meshes_visual(link)
        mesh_list.append(mesh)
    return mesh_list

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

data_root = 'assets/urdf'
category = 'Faucet'
        # setup env

for shape_id in tqdm(os.listdir(os.path.join(data_root, category))):
    env = Env(show_gui=False)
    cam = Camera(env, random_position=True)

    # load shape
    data_dir = os.path.join(data_root, category, shape_id)
    out_dir = os.path.join(data_dir, 'part_mesh_fixed')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    object_urdf_fn = os.path.join(data_dir, 'syn.urdf')
    object_material = env.get_material(0.0, 0.0, 0.01)

    env.load_object(object_urdf_fn, object_material, scale = 0.5)
    mesh_list = get_mesh_list(env)
    for i, mesh in enumerate(mesh_list):
        if i == 0:
            obj_name = 'base_link'
        else:
            obj_name = 'link_' + str(i-1)
        # o3d.visualization.draw_geometries([mesh])
        o3d.io.write_triangle_mesh(os.path.join(out_dir ,obj_name + '.obj'), mesh)
