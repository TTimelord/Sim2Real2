import os
import sys
import subprocess
from glob import glob

# sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import copy

import cv2
import hydra
import numpy as np
import open3d as o3d
import torch
import trimesh
import yaml
from hydra.experimental import (
    compose,
    initialize,
    # initialize_config_dir,
    # initialize_config_module,
)

from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
from src.utils.joint_estimation import aggregate_dense_prediction_r
from src.utils.misc import sample_point_cloud
from utils3d.utils3d.mesh.utils import as_mesh
from utils3d.utils3d.render.pyrender import PyRenderer
from utils3d.utils3d.utils.utils import get_pose

# from src.utils.joint_estimation import (
#     aggregate_dense_prediction_r,
#     eval_joint_p,
#     eval_joint_r,
# )
# %matplotlib nbagg

def downsample_points(pcd, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)

def vector_to_rotation(vector):
    z = np.array(vector)
    z = z / np.linalg.norm(z)
    x = np.array([1, 0, 0])
    x = x - z*(x.dot(z)/z.dot(z))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]

def add_r_joint_to_scene(scene,
                             axis,
                             pivot_point,
                             length,
                             radius=0.01,
                             joint_color=[200, 0, 0, 180],
                             recenter=False):
    if recenter:
        pivot_point = np.cross(axis, np.cross(pivot_point, axis))
    rotation_mat = vector_to_rotation(axis)
    screw_tran = np.eye(4)
    screw_tran[:3, :3] = rotation_mat
    screw_tran[:3, 3] = pivot_point

    axis_cylinder = trimesh.creation.cylinder(radius, height=length)
    axis_arrow = trimesh.creation.cone(radius * 2, radius * 4)
    arrow_trans = np.eye(4)
    arrow_trans[2, 3] = length / 2
    axis_arrow.apply_transform(arrow_trans)
    axis_obj = trimesh.Scene((axis_cylinder, axis_arrow))
    screw = as_mesh(axis_obj)

    # screw.apply_translation([0, 0, 0.1])
    screw.apply_transform(screw_tran)
    screw.visual.face_colors = np.array(joint_color, dtype=np.uint8)
    scene.add_geometry(screw)
    return screw


def norm(a, axis=-1):
    return np.sqrt(np.sum(a ** 2, axis=axis))


# load data
root_path = os.getcwd()
category = 'drawer'
model_id='video_1'
realdata_path = os.path.join(root_path,'real_test/real_datasets',category)

target_qpos = 0.09

pcd_path = os.path.join(realdata_path,model_id)
results_path = os.path.join(pcd_path,'digital_twin')
if not os.path.exists(results_path):
        os.makedirs(results_path)
urdf_path = os.path.join(results_path, category + '_' + model_id)
if not os.path.exists(urdf_path):
        os.makedirs(urdf_path)

pcd_1 = o3d.io.read_point_cloud(os.path.join(pcd_path,'pcd_2.pcd'))
pcd_2 = o3d.io.read_point_cloud(os.path.join(pcd_path,'pcd_1.pcd'))

ROTATE = True # True for drawer and faucet
if ROTATE:
    R = np.array([[-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]])
    pcd_1 = pcd_1.rotate(R, center=[0,0,0])
    pcd_2 = pcd_2.rotate(R, center=[0,0,0])

# src_pcd = downsample_points(pcd_1, 0.001, 50, 0.1)
# dst_pcd = downsample_points(pcd_2, 0.001, 50, 0.1)
src_pcd = pcd_1
dst_pcd = pcd_2


# Run model

with initialize(config_path='../configs/'):
    config = compose(
        config_name='config',
        overrides=[
            'experiment=sapien.yaml',
        ], return_hydra_config=True)
config.datamodule.opt.train.data_dir = '../data/'
config.datamodule.opt.val.data_dir = '../data/'
config.datamodule.opt.test.data_dir = '../data/'
model = hydra.utils.instantiate(config.model)

ckpt = torch.load('****.ckpt') # load your model here

# device = torch.device(0)
device = torch.device('cpu')

model.load_state_dict(ckpt['state_dict'], strict=True)
model = model.eval().to(device)
generator = Generator3D(
    model.model,
    device=device,
    threshold=0.4,
    seg_threshold=0.5,
    input_type='pointcloud',
    refinement_step=0,
    padding=0.1,
    resolution0=32
)

pc_start = np.asarray(src_pcd.points)
pc_end = np.asarray(dst_pcd.points)

bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
norm_center = (bound_max + bound_min) / 2
norm_scale = (bound_max - bound_min).max() * 1.1
pc_start = (pc_start - norm_center) / norm_scale
pc_end = (pc_end - norm_center) / norm_scale

pc_start, _ = sample_point_cloud(pc_start, 8192)
pc_end, _ = sample_point_cloud(pc_end, 8192)
sample = {
    'pc_start': torch.from_numpy(pc_start).unsqueeze(0).to(device).float(),
    'pc_end': torch.from_numpy(pc_end).unsqueeze(0).to(device).float()
}
mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)
with torch.no_grad():
    joint_type_logits, joint_param_revolute, joint_param_prismatic = model.model.decode_joints(mobile_points_all, c)

renderer = PyRenderer(light_kwargs={'color': np.array([1., 1., 1.]), 'intensity': 9})
# compute articulation model
mesh_dict[1].visual.face_colors = np.array([84, 220, 83, 255], dtype=np.uint8)
joint_type_prob = joint_type_logits.sigmoid().mean()

# articulation evaluation

if joint_type_prob.item()< 0.5:
    # axis voting
    joint_r_axis = (
        normalize(joint_param_revolute[:, :, :3], -1)[0].cpu().numpy()
    )
    joint_r_t = joint_param_revolute[:, :, 3][0].cpu().numpy()
    print('joint_r_t:',joint_r_t)

    joint_r_p2l_vec = (
        normalize(joint_param_revolute[:, :, 4:7], -1)[0].cpu().numpy()
    )
    joint_r_p2l_dist = joint_param_revolute[:, :, 7][0].cpu().numpy()
    p_seg = mobile_points_all[0].cpu().numpy()

    pivot_point = p_seg + joint_r_p2l_vec * joint_r_p2l_dist[:, np.newaxis]
    (
        joint_axis_pred,
        pivot_point_pred,
        config_pred,
    ) = aggregate_dense_prediction_r(
        joint_r_axis, pivot_point, joint_r_t, method="mean"
    )
    #-------------------------------------------------------------------------
# prismatic
else:
    # axis voting
    joint_p_axis = (
        normalize(joint_param_prismatic[:, :, :3], -1)[0].cpu().numpy()
    )
    # print('joint_p_axis：',joint_p_axis)
    joint_axis_pred = joint_p_axis.mean(0)
    joint_p_t = joint_param_prismatic[:, :, 3][0].cpu().numpy()
    print("joint_P_t：",joint_p_t)
    # print(joint_p_t.shape)
    config_pred = joint_p_t.mean()
    pivot_point_pred = mesh_dict[1].bounds.mean(0)



#------------------------------------------------------------------------------------------------
scene = trimesh.Scene()
static_part = mesh_dict[0].copy()
mobile_part = mesh_dict[1].copy()
static_part_simp = static_part.simplify_quadratic_decimation(10000)
mobile_part_simp = mobile_part.simplify_quadratic_decimation(10000)
mobile_part_simp.visual.face_colors = np.array(
                [84, 220, 83, 255], dtype=np.uint8
            )

mesh_0 = copy.deepcopy(static_part_simp)
mesh_0 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh_0.vertices),o3d.utility.Vector3iVector(mesh_0.faces))

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh_0.cluster_connected_triangles())
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
# print(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)
# print(cluster_area)
triangles_to_remove = cluster_n_triangles[triangle_clusters]<cluster_n_triangles.max()
mesh_0.remove_triangles_by_mask(triangles_to_remove)

mesh_1 = copy.deepcopy(mobile_part_simp)
mesh_1 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh_1.vertices),o3d.utility.Vector3iVector(mesh_1.faces))
# mesh_1.paint_uniform_color([0,1,0])
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh_1.cluster_connected_triangles())
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
# print(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)
# print(cluster_area)
triangles_to_remove = cluster_n_triangles[triangle_clusters]<cluster_n_triangles.max()
mesh_1.remove_triangles_by_mask(triangles_to_remove)

mesh0_file = os.path.join(urdf_path, 'mesh_0.obj')
o3d.io.write_triangle_mesh(mesh0_file, mesh_0)
# pcd_file = os.path.join(results_path, 'urdf', 'cmp_pcd_0.ply')
# src_pcd.translate(-norm_center)
# src_pcd.scale(1.0/(norm_scale), np.zeros(3))
# o3d.io.write_point_cloud(pcd_file, src_pcd)

mesh1_file = os.path.join(urdf_path, 'mesh_1.obj')
o3d.io.write_triangle_mesh(mesh1_file, mesh_1)

def to_trimesh(o3d_mesh):
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces)

tri_mesh_0 = to_trimesh(mesh_0)
tri_mesh_1 = to_trimesh(mesh_1)
tri_mesh_1.visual.face_colors = np.array([84, 220, 83, 255], dtype=np.uint8)

scene.add_geometry(tri_mesh_0)
scene.add_geometry(tri_mesh_1)
# pivot_point_pred[2] -= 0.2 # for drawer figure rendering
add_r_joint_to_scene(scene, joint_axis_pred, pivot_point_pred, 1.0, recenter=True)
# add_r_joint_to_scene(scene, gt_axis, pivot_point_pred, 1.0, joint_color=[0, 0, 255, 255],recenter=True)

#-------------------------------------------------------------
# write result URDF

with open(os.path.join(root_path, "template.urdf")) as f:
    urdf_txt = f.read()

if joint_type_prob.item() < 0.5:
    joint_type = "revolute"
else:
    joint_type = "prismatic"
urdf_txt = urdf_txt.replace("joint_type", joint_type)

joint_position_r_txt = " ".join([str(x) for x in -pivot_point_pred])
urdf_txt = urdf_txt.replace("joint_position_r", joint_position_r_txt)

joint_position_txt = " ".join([str(x) for x in pivot_point_pred])
urdf_txt = urdf_txt.replace("joint_position", joint_position_txt)

joint_axis_txt = " ".join([str(x) for x in joint_axis_pred])
urdf_txt = urdf_txt.replace("joint_axis", joint_axis_txt)
urdf_txt = urdf_txt.replace("joint_state_lower", str(-3.14))
urdf_txt = urdf_txt.replace("joint_state_upper", str(+3.14))
# if config_pred > 0:
#     urdf_txt = urdf_txt.replace("joint_state_lower", str(-config_pred))
#     urdf_txt = urdf_txt.replace("joint_state_upper", str(2*config_pred))
# else:
#     urdf_txt = urdf_txt.replace("joint_state_upper", str(-config_pred))
#     urdf_txt = urdf_txt.replace("joint_state_lower", str(2*config_pred))
with open(os.path.join(urdf_path, "out.urdf"), "w") as f:
    f.write(urdf_txt)

if ROTATE:
    root_ang = np.pi
    root_pos = (-1, -1, 1) * norm_center
else:
    root_ang = 0
    root_pos = norm_center

# write digital_twin config
digital_twin_config = {
    'name':category,
    'urdf_path':os.path.join(urdf_path, 'out.urdf'),
    'multiple_collisions': True,
    'scale':float(norm_scale),
    'root_pos':root_pos.tolist(),
    'root_ang':root_ang,
    'init_qpos':0.0,
    'target_qpos':target_qpos,
    'target_joint_idx':0,
    'target_joint_axis':joint_axis_pred.tolist()
}

with open(str(urdf_path) + ".yaml", 'w') as f:
    yaml.dump(digital_twin_config, f)

# save a copy of config file to CEM assets
CEM_config_path = '~/Sim2Real2/CEM/mani_skill2/assets/config_files/digital_twins/'
with open(CEM_config_path + category + '_' + model_id + ".yaml", 'w') as f:
    yaml.dump(digital_twin_config, f)

# render result mesh
if ROTATE:
    pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=-1*np.pi/4 + np.pi)
else:
    pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=-1*np.pi/4)
camera_pose = pose
light_pose = pose
rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)
rgb = rgb.copy()

rgb_name = os.path.join(results_path,'rgb.png')

cv2.imwrite(rgb_name,rgb)

try:
    completed = subprocess.run(args="~/Sim2Real2/ditto/real_test/./TestVHACD " + str(os.path.join(urdf_path, "mesh_0.obj")), shell=True)
    completed = subprocess.run(["~/Sim2Real2/ditto/real_test/./TestVHACD " + str(os.path.join(urdf_path, "mesh_1.obj"))], shell=True)
except:
    print("failed to run VHACD, please install VHACD")

# rename the decomposed file
files = os.listdir(urdf_path)
for file in files:
    if 'decomp' in file and 'mesh_0' in file:
        os.rename(os.path.join(urdf_path, file), os.path.join(urdf_path, 'mesh_0_decomp.obj'))
    if 'decomp' in file and 'mesh_1' in file:
        os.rename(os.path.join(urdf_path, file), os.path.join(urdf_path, 'mesh_1_decomp.obj'))

