import os
import sys
from logging import root
from ntpath import join
from unittest import result

from genericpath import isdir

# sys.path.append('../')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import copy
import json
import math
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyrender
import torch
import trimesh
from hydra.experimental import (
    compose,
    initialize,
    initialize_config_dir,
    initialize_config_module,
)
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf
from PIL import Image

from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D
from src.utils.joint_estimation import aggregate_dense_prediction_r
from src.utils.misc import sample_point_cloud
from utils3d.mesh.utils import as_mesh
from utils3d.render.pyrender import PyRenderer
from utils3d.utils.utils import get_pose


# from src.utils.joint_estimation import (
#     aggregate_dense_prediction_r,
#     eval_joint_p,
#     eval_joint_r,
# )
# %matplotlib nbagg
def plot_3d_point_cloud(x,
                        y,
                        z,
                        show=True,
                        show_axis=True,
                        in_u_sphere=False,
                        marker='.',
                        s=8,
                        alpha=.8,
                        figsize=(5, 5),
                        elev=10,
                        azim=240,
                        axis=None,
                        title=None,
                        lim=None,
                        *args,
                        **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
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
        lim = (min(np.min(x), np.min(y),
                   np.min(z)), max(np.max(x), np.max(y), np.max(z)))
        ax.set_xlim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_ylim(1.3 * lim[0], 1.3 * lim[1])
        ax.set_zlim(1.3 * lim[0], 1.3 * lim[1])
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if show:
        plt.show()

    return fig
def read_depth(depth_path):
    depth_img = np.array(Image.open(depth_path))
    depth_img = depth_img.astype(np.float32) * 0.001
    print(depth_img.dtype)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.empty_like(depth_img)),
        o3d.geometry.Image(depth_img),
        depth_scale=1.0,
        depth_trunc=2.0,
        convert_rgb_to_intensity=False,
    )
    return rgbd

def sum_downsample_points(point_list, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    points = np.concatenate([np.asarray(x.points) for x in point_list], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
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

def eval_joint_r(pred_axis,pred_angle,gt_axis,gt_angle):
    # pred_axis, pred_pivot_point
    # gt_axis, gt_pivot_point
    # add ambiguity
    print('gt_angle：', gt_angle)   #弧度
    print("pre_angle：", pred_angle)  #弧度
    print('gt_angle：', gt_angle*180/np.pi)   #弧度
    print("pre_angle：", pred_angle*180/np.pi)  #弧度
    axis_ori_1 = np.arccos(np.dot(pred_axis, gt_axis))  #弧度
    axis_ori_2 = np.arccos(np.dot(-pred_axis, gt_axis))
    print('axis_ori_1:',axis_ori_1)
    print('axis_ori_2:', axis_ori_2)
    if axis_ori_1 < axis_ori_2:
        axis_ori = axis_ori_1
        # config = np.abs(gt_angle - pred_angle) * 180 / np.pi   #度
        config = np.abs(gt_angle - pred_angle)
    else:
        axis_ori = axis_ori_2
        # config = np.abs(gt_angle + pred_angle) * 180 / np.pi
        config = np.abs(gt_angle + pred_angle)


    # pred_moment = np.cross(pred_pivot_point, pred_axis)
    # gt_moment = np.cross(gt_pivot_point, gt_axis)

    #异面直线距离公式
    # if np.abs(axis_ori) < 0.001:
    #     dist = np.cross(gt_axis, (pred_moment - gt_moment))
    #     dist = norm(dist)
    # else:
    #     dist = np.abs(gt_axis.dot(pred_moment) + pred_axis.dot(gt_moment)) / norm(
    #         np.cross(gt_axis, pred_axis)
    #     )
    # axis_ori = axis_ori * 180 / np.pi   #度

    return axis_ori, config

def eval_joint_p(pred, gt):
    pred_axis, pred_d = pred
    gt_axis, gt_d = gt
    # add ambiguity
    axis_ori_1 = np.arccos(np.dot(pred_axis, gt_axis))
    axis_ori_2 = np.arccos(np.dot(-pred_axis, gt_axis))  #弧度
    print('axis_ori_1:',axis_ori_1)
    print('axis_ori_2：',axis_ori_2)
    print('gt_d：',gt_d)
    print('pred_d:',pred_d)
    if axis_ori_1 < axis_ori_2:
        axis_ori = axis_ori_1
        config_err = np.abs(gt_d - pred_d)    #米
    else:
        axis_ori = axis_ori_2
        config_err = np.abs(gt_d + pred_d)
    return axis_ori, config_err


def cal_ang(point_a, point_b, point_c, flag):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0] #点a,b,c的x坐标
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1] #点a,b,c的y坐标
    if len(point_a) == len(point_b) == len(point_c) == 3:
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]   #点a, b, c的z坐标
    else:
        a_z, b_z, c_z = 0, 0, 0   #二维形式
    #向量 m=(x1,y1,z1), n=(x2,y2,z2)
    x1, y1, z1 = (a_x-b_x), (a_y-b_y), (a_z-b_z)
    x2, y2, z2 = (c_x-b_x), (c_y-b_y), (c_z-b_z)
    ## 两个向量的夹角，即角点b的夹角余弦值
    cos_b = (x1*x2+y1*y2+z1*z2)/(math.sqrt(x1**2+y1**2+z1**2)*(math.sqrt(x2**2+y2**2+z2**2)))
    # B = math.degrees(math.acos(cos_b))  #角点b的夹角值   度
    B = flag * math.acos(cos_b)     #弧度
    return B

#--------------主要修改的参数为:data_model/model_id/ckpt----------------------------
# load data
root_path = os.getcwd()
data_model = 'drawer'
realdata_path = os.path.join(root_path,'notebooks/real_datasets',data_model)
# realdata_path = os.path.join(root_path,'notebooks/Ditto_real_data',model)

model_id='2'
result_dict = {}
# axis_ori_err_all = 0
# config_err_all = 0
# correct_all = 0
# num = 0
# for model_id in os.listdir(realdata_path):
# num += 1
# if model_id==1:
pcd_path = os.path.join(realdata_path,model_id)
results_path = os.path.join(pcd_path,'result_test_cleandepth')
if not os.path.exists(results_path):
        os.makedirs(results_path)
# pcd_2_path = 'Ditto_real_data/laptop_black/1/pcd_2.pcd'
pcd_1 = o3d.io.read_point_cloud(os.path.join(pcd_path,'pcd_1.pcd'))
pcd_2 = o3d.io.read_point_cloud(os.path.join(pcd_path,'pcd_2.pcd'))
# crop out the object
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0.2, -0.6, -0.1), max_bound=(1.2, 0.4, 0.55))
# print(bbox)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)
# o3d.visualization.draw_geometries([pcd_1, pcd_2, mesh_frame, bbox])
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)
# o3d.visualization.draw_geometries([pcd_2, mesh_frame, bbox])
pcd_1 = pcd_1.crop(bbox)
pcd_2 = pcd_2.crop(bbox)


# find bounding box of data
src_pc_list = []
src_fused_pc = np.asarray(pcd_1.points)
center = (np.min(src_fused_pc, 0) + np.max(src_fused_pc, 0)) / 2
scale = (np.max(src_fused_pc, 0) - np.min(src_fused_pc, 0)).max()
scale *= 1.1

# back project and normalize point cloud

src_pcd_list = []
pcd = pcd_1
center_transform = np.eye(4)
center_transform[:3, 3] = -center
pcd.transform(center_transform)
pcd.scale(1 / scale, np.zeros((3, 1)))
src_pcd_list.append(pcd)

dst_pcd_list = []
pcd = pcd_2
center_transform = np.eye(4)
center_transform[:3, 3] = -center
pcd.transform(center_transform)
pcd.scale(1 / scale, np.zeros((3, 1)))
dst_pcd_list.append(pcd)

src_pcd = sum_downsample_points(src_pcd_list, 0.02, 50, 0.1)
dst_pcd = sum_downsample_points(dst_pcd_list, 0.02, 50, 0.1)

# visualize crop results
# tune crop box to get better isolated objects

# o3d.visualization.draw_geometries([src_pcd])
# o3d.visualization.draw_geometries([dst_pcd])

# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# plot_3d_point_cloud(*np.asarray(src_pcd.points).T,
#                     axis=ax,
#                     azim=30,
#                     elev=30,
#                     lim=[(-0.5, 0.5)] * 3)

# ax = fig.add_subplot(1, 2, 2, projection='3d')
# plot_3d_point_cloud(*np.asarray(dst_pcd.points).T,
#                     axis=ax,
#                     azim=30,
#                     elev=30,
#                     lim=[(-0.5, 0.5)] * 3)

# plt.show()
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
ckpt = torch.load('****.ckpt')
device = torch.device(0)
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
if data_model=='drawer':
    joint_type = 1
else:
    joint_type = 0

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
    start_data_file = Path(os.path.join(pcd_path,'start_data.txt'))
    end_data_file = Path(os.path.join(pcd_path,'end_data.txt'))
    if start_data_file.exists():
        start_data = np.loadtxt(os.path.join(pcd_path,'start_data.txt'),dtype=np.float32, delimiter=',')
        if data_model=='faucet':
            flag = start_data[0][0]
            start_state = cal_ang(start_data[1], start_data[2], start_data[3], flag)
        else:
            flag = 1
            start_state = cal_ang(start_data[0], start_data[1], start_data[2], flag)
    else:
        start_state = 0

    if end_data_file.exists():
        end_data = np.loadtxt(os.path.join(pcd_path,'end_data.txt'),dtype=np.float32, delimiter=',')
        if data_model=='faucet':
            flag = end_data[0][0]
            # print('flag：', flag)
            end_state = cal_ang(end_data[1], end_data[2], end_data[3], flag)
        else:
            flag = 1
            end_state = cal_ang(end_data[0], end_data[1], end_data[2], flag)
    else:
        end_state = 0

    points = np.loadtxt(os.path.join(pcd_path,'picking_list.txt'),dtype=np.float32, delimiter=',')
    gt_angle = end_state - start_state   #弧度 角度
    screw_axis = torch.from_numpy(points[1]-points[0])
    gt_axis = normalize(screw_axis,-1)

    axis_ori_err, config_err = eval_joint_r(
    joint_axis_pred, config_pred, gt_axis, gt_angle)

    correct = (joint_type_prob < 0.5).long().item() == joint_type

    result_dict["articulation_{}".format(model_id)] = {
        "revolute": {
            "axis_orientation": axis_ori_err,
            # "axis_displacement": axis_displacement,
            "angle_err": config_err,
        },
        "prismatic": None,
        "joint_type": {"accuracy": correct},
    }
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

    start_data = np.loadtxt(os.path.join(pcd_path,'start_data.txt'),dtype=np.float32, delimiter=',')
    end_data = np.loadtxt(os.path.join(pcd_path,'end_data.txt'),dtype=np.float32, delimiter=',')
    start_state = math.sqrt((start_data[0,0]-start_data[1,0])**2+(start_data[0,1]-start_data[1,1])**2
                +(start_data[0,2]-start_data[1,2])**2)
    end_state = math.sqrt((end_data[0,0]-end_data[1,0])**2+(end_data[0,1]-end_data[1,1])**2
                +(end_data[0,2]-end_data[1,2])**2)
    points = np.loadtxt(os.path.join(pcd_path,'picking_list.txt'),dtype=np.float32, delimiter=',')
    gt_t = end_state - start_state
    screw_axis = torch.from_numpy(points[1]-points[0])
    # print(screw_axis)
    gt_axis = normalize(screw_axis,-1)
    axis_ori_err, config_err = eval_joint_p(
            (joint_axis_pred, config_pred), (gt_axis, gt_t)
        )
    correct = (joint_type_prob > 0.5).long().item() == joint_type

    result_dict["articulation_{}".format(model_id)] = {
        "prismatic": {
            "axis_orientation": axis_ori_err,
            "config_err": config_err,
        },
        "revolute": None,
        "joint_type": {"accuracy": correct},
    }
    # axis_ori_err_all += axis_ori_err
    # config_err_all += config_err
    # correct_all += correct

#------------------------------------------------------------------------------------------------
scene = trimesh.Scene()
static_part = mesh_dict[0].copy()
mobile_part = mesh_dict[1].copy()
static_part_simp = static_part.simplify_quadratic_decimation(10000)  #网格抽取算法用于简化网格
mobile_part_simp = mobile_part.simplify_quadratic_decimation(10000)
mobile_part_simp.visual.face_colors = np.array(
                [84, 220, 83, 255], dtype=np.uint8
            )

mesh_0 = copy.deepcopy(static_part_simp)
mesh_0 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh_0.vertices),o3d.utility.Vector3iVector(mesh_0.faces))
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    #每一个三角形的索引、每一个集群中三角形的数量以及集群的表面积
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh_0.cluster_connected_triangles())   #将每一个三角形分配给一个连接的三角集群
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
    #每一个三角形的索引、每一个集群中三角形的数量以及集群的表面积
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh_1.cluster_connected_triangles())   #将每一个三角形分配给一个连接的三角集群
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
# print(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)
# print(cluster_area)
triangles_to_remove = cluster_n_triangles[triangle_clusters]<cluster_n_triangles.max()
mesh_1.remove_triangles_by_mask(triangles_to_remove)

mesh0_file = os.path.join(results_path,'mesh_0.obj')
o3d.io.write_triangle_mesh(mesh0_file, mesh_0)

mesh1_file = os.path.join(results_path, 'mesh_1.obj')
o3d.io.write_triangle_mesh(mesh1_file, mesh_1)

# _ = static_part.export(os.path.join(pcd_path, "static.obj"))
# _ = mobile_part.export(os.path.join(pcd_path, "mobile.obj"))
scene.add_geometry(static_part)
scene.add_geometry(mobile_part)
add_r_joint_to_scene(scene, joint_axis_pred, pivot_point_pred, 1.0, recenter=True)
add_r_joint_to_scene(scene, gt_axis, pivot_point_pred, 1.0, joint_color=[0, 0, 255, 255],recenter=True)


#--------------------add gt axis-----------------------
# data = np.loadtxt(os.path.join(pcd_path,'picking_list.txt'),dtype=np.float32, delimiter=',')
# print(data)
# joint_axis_gt = data[1]-data[0]
# pivot_point_gt = (data[1]+data[0])/2
# add_r_joint_to_scene(scene, joint_axis_gt, pivot_point_gt,1.0, joint_color =[1,1,1],recenter=True)

# axis_ori_err, axis_displacement = eval_joint_r(
#             joint_axis_pred, pivot_point_pred,
#             joint_axis_gt, pivot_point_gt)
#-----------------------------------------------------------

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
if config_pred > 0:
    urdf_txt = urdf_txt.replace("joint_state_lower", "0.0")
    urdf_txt = urdf_txt.replace("joint_state_upper", str(3*config_pred))
else:
    urdf_txt = urdf_txt.replace("joint_state_upper", "0.0")
    urdf_txt = urdf_txt.replace("joint_state_lower", str(3*config_pred))
with open(os.path.join(results_path, "out.urdf"), "w") as f:
    f.write(urdf_txt)

# render result mesh
pose = get_pose(1.5, ax=np.pi / 3, ay=0, az=-1*np.pi/4)
camera_pose = pose
light_pose = pose
rgb, depth = renderer.render_mesh(scene, camera_pose, light_pose)
rgb = rgb.copy()
# print(rgb.shape)
rgb_name = os.path.join(results_path,'rgb.png')
cv2.putText(rgb, 'axis_ori_err:{}({})'.format(axis_ori_err,axis_ori_err*180/np.pi), (65, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)
if data_model=='drawer':
    cv2.putText(rgb, 'config_err:{}'.format(config_err), (65, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)
else:
    cv2.putText(rgb, 'config_err:{}({})'.format(config_err,config_err*180/np.pi), (65, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)

cv2.imwrite(rgb_name,rgb)
# Image.fromarray(rgb)

axis_ori_err_showmodel_path = f"{realdata_path}/axis_ori_err_cleandepth.txt"
f = open(axis_ori_err_showmodel_path, 'a')
print(axis_ori_err,file=f)
f.close()

config_err_showmodel_path = f"{realdata_path}/config_err_cleandepth.txt"
f = open(config_err_showmodel_path, 'a')
print(config_err,file=f)
f.close()

type_val_showmodel_path = f"{realdata_path}/type_val_cleandepth.txt"
f = open(type_val_showmodel_path, 'a')
print(correct+0,file=f)
f.close()

result_showmodel_path = f"{realdata_path}/result_showmodel_cleandepth.yaml"
f = open(result_showmodel_path, 'a')
print(result_dict,file=f)
print('**********************************************************************************',file=f)
f.close()
