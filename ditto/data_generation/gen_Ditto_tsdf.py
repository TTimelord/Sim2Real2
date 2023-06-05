# from ntpath import join
import sys
import os
import os.path as osp

from bleach import clean
from cv2 import PCA_DATA_AS_COL
from mani_skill2.utils.sapien_utils import get_articulation_state, get_entity_by_name
from mani_skill2.utils.trimesh_utils import get_actor_meshes, merge_meshes, get_actor_meshes_visual
from PIL import Image
# import torch
from mani_skill2.utils.urdf.urdf import URDF
import numpy as np
import sapien.core as sapien
from sapien.core import Pose
import trimesh
from tqdm import tqdm
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from mani_skill2 import ASSET_DIR, ROOT_DIR
from mani_skill2.envs.fixed_single_articulation.open_cabinet_Shape2motion import (
    OpenCabinetDoorSensor,
)
import transforms3d as t3d
from mani_skill2.agents.camera import get_texture
import json
import open3d as o3d
from mani_skill2.utils.contrib import (
    apply_pose_to_points,
    normalize_and_clip_in_interval,
    o3d_to_trimesh,
    trimesh_to_o3d,
)
from mani_skill2.utils.o3d_utils import merge_mesh

import random

# _ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), '../..'))
ManiSkill_path = ROOT_DIR.parent.resolve()
sys.path.insert(0, str(ManiSkill_path))
Ditto_path = Path('~/Ditto').expanduser().resolve()
sys.path.append(str(Ditto_path))
# try:
#     from src.third_party.ConvONets.utils.libmesh.inside_mesh import check_mesh_contains
# except:
#     print('import libmesh failed!')
# from mani_skill2.agents.base_agent import get_camera_poses
# OUTPUT_DIR = Path("/media/DATA/LINUX_DATA/mani_skill2022/data/cabinet_drawer_handle")
# OUTPUT_DIR = Path("/home/mj/dataset_single_real/data/cabinet_drawer_handle")

#-------------------------------改动classes、sub_dataset、nums、angles_num以及start/end_open_extent的大小--------------------------------------
classes = 'drawer'
dataset_name = 'Ditto_' + classes + '_dataset'
config_folder_name = 'Ditto_'+classes+'_filtered'
sub_dataset = 'train'
state_nums = 5
pose_nums = 10
angles_num = 'stereo_multipose_multistate'    #single_state or multi_state or stereo_depth
depth_name = 'clean_depth'   #clean_depth or stereo_depth

OUTPUT_DIR = ManiSkill_path / 'data/Shape2motion_{}'.format(angles_num)/classes/sub_dataset
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

NUM_POS_PER_SCENE = 2   # 同一视角下柜门不同角度

MAX_DEPTH = 2.0



# ----------------------------------------------------------
def gen_npz(npy_path, npz_path, model_name):
    # for model in os.listdir(npy_path):
    out_dict = {}
    # model_path = os.path.join(npy_path,model)
    for file in os.listdir(npy_path):
        file_name = file.split('.')[0]
        file_path = os.path.join(npy_path, file)
        file_npy = np.load(file_path, allow_pickle=True)
        out_dict[file_name] = file_npy
    np.savez(os.path.join(npz_path, '{}.npz'.format(model_name)), **out_dict)
############################rgbd2pc#########################
# 读取rgbd图片   由rgb和depth组成


def read_rgbd(rgb_path, depth_path):
    depth_img = np.array(Image.open(depth_path))  # 加载depth
    depth_img = depth_img.astype(np.float32)*0.001  # 转换成相应的类型
    rgb_img = np.array(Image.open(rgb_path))  # 加载rgb
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_img),
        o3d.geometry.Image(depth_img),
        depth_scale=1.0,
        depth_trunc=0.7,
        convert_rgb_to_intensity=False,
    )
    return rgbd

#---------读取depth tsdf--------------------
def read_depth(depth_path):
    depth_img = np.array(Image.open(depth_path))
    depth_img = depth_img.astype(np.float32) * 0.001

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.empty_like(depth_img)),
        o3d.geometry.Image(depth_img),
        depth_scale = 1.0,
        depth_trunc = 2.0,
        convert_rgb_to_intensity=False,
    )
    return rgbd

# 降采样


def sum_downsample_points(point_list, voxel_size=0.01, nb_neighbors=20, std_ratio=2.0):
    # 按行进行拼接 len(point_list)个点
    points = np.concatenate([np.asarray(x.points) for x in point_list], axis=0)
    # 转换成点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 降采样
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # 去掉离群点
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

# 标准化
# def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
#     return tensor / ((tensor**2).sum(dim, keepdim=True).sqrt() + 1.0e-5)

# 向量转换成旋转矩阵


def vector_to_rotation(vector):
    z = np.array(vector)
    z = z/np.linalg.norm(z)  # 归一化
    x = np.array([1, 0, 0])
    x = x - z*(x.dot(z)/z.dot(z))
    x = x/np.linalg.norm(x)
    y = np.cross(z, x)
    return np.c_[x, y, z]

############################################################


def visualize_depth(depth):
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(float) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth


def gen_single_scene(config_name):
    np.set_printoptions(suppress=True, precision=4)
    env = OpenCabinetDoorSensor(
        articulation_config_path=ASSET_DIR/ f"Ditto_dataset_configs/{config_folder_name}/{sub_dataset}/{config_name}.yml",
        # is_action_normalized=False,
    )
    env.reset()
    env._init_open_extent_range = 1.0
    env._scene.set_ambient_light([0.5, 0.5, 0.5])
    env._scene.add_point_light([-0.3, -0.3, 2.5], [30, 30, 30])
    env._scene.add_point_light([2, -2, 2.5], [10, 10, 10])
    env._scene.add_point_light([-2, 2, 2.5], [10, 10, 10])
    env._scene.add_point_light([2, 2, 2.5], [10, 10, 10])
    env._scene.add_point_light([-2, -2, 2.5], [10, 10, 10])


    z = np.linspace(env._articulation_init_rot_min_z,env._articulation_init_rot_max_z,pose_nums//2)
    z1 = np.linspace(env._articulation_init_rot_min_z1,env._articulation_init_rot_max_z1,pose_nums//2)
    # 给关节物体一个随机的姿势
    def set_root_random_pose(j):
        pos_x = env._episode_rng.uniform(
            env._articulation_init_pos_min_x, env._articulation_init_pos_max_x
        )
        pos_y = env._episode_rng.uniform(
            env._articulation_init_pos_min_y, env._articulation_init_pos_max_y
        )
        if j<(pose_nums/2):
            if(sub_dataset=='test'):
                rot_z = env._episode_rng.uniform(
                    env._articulation_init_rot_min_z, env._articulation_init_rot_max_z
                )
            else:
                rot_z = z[j]

        else:
            if(sub_dataset=='test'):
                rot_z = env._episode_rng.uniform(
                    env._articulation_init_rot_min_z1, env._articulation_init_rot_max_z1
                )
            else:
                rot_z = z1[j-(pose_nums//2)]

        print('rot_z：',rot_z)
        pose = Pose(
            [
                pos_x,
                pos_y,
                -env._articulation_config.scale
                * env._articulation_config.bbox_min[2],
            ],
            [np.sqrt(1 - rot_z ** 2), 0, 0, rot_z],
        )
        env._articulation.set_root_pose(pose)
        return pose
    
    def get_random_qpos():
        [[lmin, lmax]] = env._target_joint.get_limits()  #(-3.14, 3.14)
        # start_open_extent = env._episode_rng.uniform(0.5,0.7)  #(0.5，1)# uniform distribution from 0 to 1     #cabinet refrigerator
        start_open_extent = env._episode_rng.uniform(0.1,0.55)  # uniform distribution from 0 to 1       #laptop  start_open_extent=0.5的时候，对应start_qpos=0  电脑打开90度 向内开start_qpos<0,反之大于0
        # start_open_extent = env._episode_rng.uniform(0.47,0.75)  # stapler
        # start_open_extent = env._episode_rng.uniform(0.15,0.4)  # uniform distribution from 0 to 1       #laptop  start_open_extent=0.5的时候，对应start_qpos=0  电脑打开90度 向内开start_qpos<0,反之大于0
        # start_open_extent = env._episode_rng.uniform(0,0.55)  # uniform distribution from 0 to 1       #drawer
        # start_open_extent = env._episode_rng.uniform(0.25,0.75)  # uniform distribution from 0 to 1       #faucet

        print("start_open_extent:",start_open_extent)
        # the min difference of open extent before and after the interaction
        # open_extent_diff =  0.1 #0.02 #0.1    #cabinet laptop refqrigerator
        # open_extent_diff = 0.05     #oven eyeglasses
        # open_extent_diff = 0.03   #stapler
        open_extent_diff = 0.1   #stapler faucet

        while True:
            # end_open_extent = env._episode_rng.uniform(0.5,0.7)   #（0.5，1)#cabinet refrigerator
            end_open_extent = env._episode_rng.uniform(0.1,0.55)    #laptop
            # end_open_extent = env._episode_rng.uniform(0.15,0.4)    #laptop

            # end_open_extent = env._episode_rng.uniform(0.51,0.53)  #oven
            # end_open_extent = env._episode_rng.uniform(-0.2,0.1)   #eyeglasses
            # end_open_extent = env._episode_rng.uniform(0.47,0.75)   #stapler
            # end_open_extent = env._episode_rng.uniform(0,0.55)   #drawer
            # end_open_extent = env._episode_rng.uniform(0.25,0.75)   #faucet

            print('end_open_extent：',end_open_extent)
            if abs(start_open_extent - end_open_extent) > open_extent_diff:
                break
        start_qpos = np.zeros(env._articulation.dof)
        end_qpos = np.zeros(env._articulation.dof)
        for i in range(env._articulation.dof):
            start_qpos[i] = env._articulation.get_active_joints()[i].get_limits()[0][0]
            end_qpos[i] = env._articulation.get_active_joints()[i].get_limits()[0][0]
        start_qpos[env._articulation_config.target_joint_idx] = (
            lmin + (lmax - lmin) * start_open_extent
            # lmin + (lmax - lmin) * 0.4   #pai

        )
        end_qpos[env._articulation_config.target_joint_idx] = (
            lmin + (lmax - lmin) * end_open_extent
            # lmin + (lmax - lmin) * 0.6   #0

        )
        # print(start_qpos)
        # print(end_qpos)
        return start_qpos, end_qpos


    # root_pose: Pose = set_root_random_pose()  # root_pose
    model_name = config_name.split('-', 1)[0]
    demo_name = config_name

    # mlq test

    def gen_parts_mesh():
        # mesh_path = OUTPUT_DIR/f"urdfs"/f"{model_name}"/f"part_objs"
        mesh_path = ASSET_DIR/"Ditto_datasets"/f'{dataset_name}'/f'{sub_dataset}'/f"{model_name}"/f'part_mesh'
        # if os.path.exists(mesh_path):
        #     return

        if not os.path.exists(mesh_path):
            os.makedirs(mesh_path)
    
        target_link_name = env._target_link.name
        for link in env._articulation.get_links():
            link_name = link.name
            # target_joint_idx = env._articulation.get_active_joints().index(target_joint)
            # if link_name =='base':
            #     continue
            link_mesh = get_actor_meshes_visual(link)
            # if link_name == target_link_name:
            #     mesh_file = os.path.join(mesh_path, 'target_part.obj')
            # else:
            mesh_file = os.path.join(mesh_path, '{}.obj'.format(link_name))
            # link_mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries([link_mesh])
            o3d.io.write_triangle_mesh(mesh_file, link_mesh)

    # gen_parts_mesh()
    for j in range(pose_nums):
        if j==0:
            gen_parts_mesh()
        root_pose: Pose = set_root_random_pose(j)  # root_pose
        for i in range(state_nums):
            different_angle_demo_name = demo_name+'-'+ str(j)+'_'+str(i)
         
            demo_dir = OUTPUT_DIR/f"sensors"/f"{demo_name}"/f"{different_angle_demo_name}"
            npy_data_dir = OUTPUT_DIR/f"data_npy"/f"{demo_name}"/f"{different_angle_demo_name}"
            npz_data_dir = OUTPUT_DIR/f"data_npz"/f"{demo_name}"
            # mesh_root = OUTPUT_DIR/f"urdfs"/f"{model_name}"/f"part_objs"
            mesh_root = ASSET_DIR/"Ditto_datasets"/f'{dataset_name}'/f'{sub_dataset}'/f"{model_name}"/f'part_mesh'
        


            if not os.path.exists(npy_data_dir):
                os.makedirs(npy_data_dir)
            if not os.path.exists(npz_data_dir):
                os.makedirs(npz_data_dir)

            object_root_path = ASSET_DIR
            object_path_env = env._articulation_config.urdf_path
            object_path = os.path.join(object_root_path, object_path_env)
            np.save(os.path.join(npy_data_dir, 'object_path'), object_path)
            target_joint_index = env._articulation_config.target_joint_idx

            np.save(os.path.join(npy_data_dir, 'joint_index'), target_joint_index)
            target_joint_type = env._articulation.get_active_joints()[target_joint_index].type
            if target_joint_type == 'prismatic':
                joint_type = 1
            elif target_joint_type == 'revolute':
                joint_type = 0
            np.save(os.path.join(npy_data_dir, 'joint_type'), joint_type)


        #########occupancy related code###############
            def get_mesh_pose_list_from_world():
                # gen_parts_mesh()
                # mesh_root = OUTPUT_DIR/f"urdfs"/f"{model_name}"/f"part_objs"
                mesh_root = ASSET_DIR/"Ditto_datasets"/f'{dataset_name}'/f'{sub_dataset}'/f"{model_name}"/f'part_mesh'

                mesh_pose_list = []
                mesh_pose_dict = {}

                for link in env._articulation.get_links():
                    link_name = link.name
                    target_link_name = env._target_link.name

                    if(link_name=='base_link'):
                        index = 'link_' + '-1'
                    else:
                        index = 'link_' + str(int(link_name)-1)
                    # if(link_name == target_link_name):
                    #     mesh_path = os.path.join(mesh_root, 'target_part.obj')
                    # else:
                    mesh_path = os.path.join(mesh_root, '{}.obj'.format(link_name))
                    pose = link.get_pose().to_transformation_matrix()
                    scale_float = env._articulation_config.scale
                    scale = np.repeat(scale_float, 3)
                    mesh_mumber = [(mesh_path, scale, pose)]
                    mesh_pose_list.append((mesh_path, scale, pose))
                    # link_index应和joint的index对应
                    mesh_pose_dict[index] = mesh_mumber
                    # mesh_pose_dict[link_name] = mesh_mumber
                return mesh_pose_list, mesh_pose_dict

            # ---------------------------------------
            # 基于open3d的occ
            def get_occ_open3d(mesh_pose_dict_name, occ_list_name, p_occ_name, n_iou_points):
                mesh_pose_list, mesh_pose_dict = get_mesh_pose_list_from_world()
                np.save(os.path.join(npy_data_dir, mesh_pose_dict_name), mesh_pose_dict)

                mesh_list = get_meshes_open3d(mesh_pose_list)

                points_occ, occ, occ_list, bounds = sample_iou_points_open3d(
                    mesh_list, n_iou_points)
                # points_occ, occ_list = sample_iou_points(mesh_list, n_iou_points)

                np.save(os.path.join(npy_data_dir, p_occ_name), points_occ)
                np.save(os.path.join(npy_data_dir, occ_list_name), occ_list)
                return bounds

            # get mesh from the .obj files exported earlier
            def get_meshes_open3d(mesh_pose_list):
                mesh_list = []

                for mesh_path, scale, pose in mesh_pose_list:
                    mesh = o3d.io.read_triangle_mesh(mesh_path)
                    # mesh.scale(scale,center = mesh.get_center())
                    mesh.transform(pose) 
                    mesh_list.append(mesh)
                return mesh_list

            def sample_iou_points_open3d(mesh_list, num_point, padding=0.02, uniform=False, size=0.3):
                mesh_merged = o3d.geometry.TriangleMesh()
                for mesh in mesh_list:
                    mesh_merged += mesh
                bbox = mesh_merged.get_axis_aligned_bounding_box()
                bounds = np.vstack((bbox.get_min_bound().astype(
                    np.float32), bbox.get_max_bound().astype(np.float32)))
                # print(bounds)

                occ_list = []
                points = np.random.rand(num_point, 3).astype(np.float32)
                if uniform:
                    points *= size + 2 * padding
                    points -= padding
                else:
                    points = points * (bounds[[1]] + 2 * padding - bounds[[0]]) + bounds[[0]] - padding
                occ = np.zeros(num_point, dtype=bool)
                for mesh in mesh_list:
                    occi = check_mesh_contains_open3d(mesh, points)

                    # visualization ######################################################################
                    # pcd_part = o3d.geometry.PointCloud()
                    # pcd_all = o3d.geometry.PointCloud()
                    # pcd_part.points = o3d.utility.Vector3dVector(points[occi])
                    # pcd_all.points = o3d.utility.Vector3dVector(points)
                    # pcd_all.paint_uniform_color([1, 0.706, 0])
                    # pcd_part.paint_uniform_color([1, 0, 0])
                    # origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)

                    # o3d.visualization.draw_geometries([bbox, mesh, origin_frame, pcd_all, pcd_part])
                    # o3d.visualization.draw_geometries([origin_frame, pcd_all, pcd_part])
                    # visualization ######################################################################

                    occ_list.append(occi)
                    occ = occ | occi
                return points, occ, occ_list, bounds

            def check_mesh_contains_open3d(mesh, points):
                t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
                # Create a scene and add the triangle mesh
                scene = o3d.t.geometry.RaycastingScene()
                # we do not need the geometry ID for mesh
                _ = scene.add_triangles(t_mesh)
                occ_label = scene.compute_occupancy(points)
                return occ_label.numpy().astype(bool)

            # ---------------------------------------

            # --------------------------------------------------------------------------
            # 添加螺旋关节

            def get_screw():
                cabinet: sapien.Articulation = env.articulation
                global_pose = env._target_joint.get_global_pose()
                # print('global_pose:', global_pose)

                rot_mat: np.ndarray = t3d.quaternions.quat2mat(global_pose.q)
                # axis_frame.rotate(rot_mat)
                # print('rot_mat：',rot_mat)

                screw_axis = rot_mat[:3, 0]
                # print('screw_axis：',screw_axis)
                pivot_point = global_pose.p
                # print('povot_point：',pivot_point)
                screw_moment = np.cross(pivot_point, screw_axis)
                # print('screw_moment：',screw_moment)
                return screw_axis, screw_moment

        # ----------------------------------------------------------------------------------------------------------------------------------
        # root_pose: Pose = set_root_random_pose()  # root_pose

            start_qpos, end_qpos = get_random_qpos()
            print("start_qpos：",start_qpos)
            screw_axis, screw_moment = get_screw()
            np.save(os.path.join(npy_data_dir, 'screw_axis'), screw_axis)
            np.save(os.path.join(npy_data_dir, 'screw_moment'), screw_moment)
            for qpos_idx in tqdm(range(NUM_POS_PER_SCENE)):
                qpos_dir = demo_dir / f"imgs_{qpos_idx}"
                qpos_dir.mkdir(parents=True, exist_ok=True)

                if qpos_idx == 0:
                    env._articulation.set_qpos(start_qpos)
                    state = start_qpos[env._articulation_config.target_joint_idx]
                    print("state:",state)
                    # print(start_qpos[env._articulation_config.target_joint_idx])

                    mesh_pose_dict_name = 'start_mesh_pose_dict'
                    occ_list_name = 'start_occ_list'
                    p_occ_name = 'start_p_occ'
                    state_name = 'state_start'
                    pc_name = 'pc_start'
                    pc_seg_name = 'pc_seg_start'

                else:
                    env._articulation.set_qpos(end_qpos)
                    state = end_qpos[env._articulation_config.target_joint_idx]

                    mesh_pose_dict_name = 'end_mesh_pose_dict'
                    occ_list_name = 'end_occ_list'
                    p_occ_name = 'end_p_occ'
                    state_name = 'state_end'
                    pc_name = 'pc_end'
                    pc_seg_name = 'pc_seg_end'

                # # debug:
                # viewer = env.render()
                # print('Press [e] to continue')
                # while True:
                #     if viewer.window.key_down('e'):
                #         break
                #     env.render()

                np.save(os.path.join(npy_data_dir, state_name), state)
                # print(state)

                n_iou_points = 100000
                # get_occ_from_world(mesh_pose_dict_name,occ_list_name,p_occ_name,n_iou_points)
                bounds = get_occ_open3d(mesh_pose_dict_name,occ_list_name, p_occ_name, n_iou_points)

                # box_bound_dir = os.path.join('/home/mj/ManiSkill2022/mani_skill2/assets/partnet_mobility_dataset/',model_name)
                box_bound_dir = ASSET_DIR / 'Ditto_datasets'/f'{dataset_name}'/f'{sub_dataset}' / f'{model_name}'
                with open(os.path.join(box_bound_dir, 'bounding_box.json'), 'r') as f:
                    box_bound = json.load(f)
                # print(box_bound)
                with open(os.path.join(demo_dir, 'bounding_box.json'), 'w') as f:
                    json.dump(box_bound, f)
                # print(box_bound)
                views = env._agent.get_images(depth=True)
                camera_intrinsic = views["base_d415"]['camera_intrinsic']
                # print(camera_intrinsic)
                f = open(qpos_dir / "camera_intrinsic.json", 'w')
                print(camera_intrinsic.tolist(), file=f)
                f.close()

                camera_pose = np.linalg.inv(views["base_d415"]['camera_extrinsic_base_frame'])  # mlq: camera -> base
                camera_pose[0, 3] += -0.6
                camera_pose[1, 3] += 0.4  # base -> world

                # print(camera_pose)
                f1 = open(qpos_dir/"camera_pose.json", 'w')
                print(camera_pose.tolist(), file=f1)
                f1.close()

                rgb = views["base_d415"]["rgb"][..., ::-1]
                # clean_depth = views["base_d415"]["clean_depth"][:, :, 0]
                if depth_name== 'clean_depth':
                    depth = views["base_d415"][depth_name][:, :, 0]
                else:
                    depth = views["base_d415"][depth_name]


                # if i==0:
                cv2.imwrite(str(qpos_dir / "rgb.png"),views["base_d415"]["rgb"][..., ::-1])
                if depth_name=='clean_depth':
                    cv2.imwrite(
                        str(qpos_dir / f"{depth_name}.png"),
                        (views["base_d415"][depth_name][:, :, 0] * 1000.0).astype(np.uint16),
                    )
                else:
                    cv2.imwrite(
                        str(qpos_dir / f"{depth_name}.png"),
                        (views["base_d415"][depth_name] * 1000.0).astype(np.uint16),
                    )
                cv2.imwrite(str(qpos_dir/f"{depth_name}_colored.png"),
                            visualize_depth(views['base_d415'][depth_name]),
                    )

                """
                -----------------------------由深度图像和rgb图像生成点云-------------------------------
                """
                intrinsics = o3d.camera.PinholeCameraIntrinsic(  # mlq: 应该用sapien获取的内参？
                    width=1920,
                    height=1080,
                    fx=1387.5,
                    fy=1386.2,
                    cx=982.59,
                    cy=565.32,
                )
                data = []
                num_point_per_depth = 2048
                if qpos_idx == 0:
                    img_dir = os.path.join(demo_dir, 'imgs_0')
                    # print(src_img_dir)
                else:
                    img_dir = os.path.join(demo_dir, 'imgs_1')
                with open(os.path.join(img_dir, 'camera_pose.json')) as f:
                    #    print(f)
                    src_transform = json.load(f)  # camera_pose

                # rgbd = read_rgbd(os.path.join(img_dir, f'rgb.png'),
                #                 os.path.join(img_dir, f'clean_depth.png'))
                rgbd = read_depth(os.path.join(img_dir, f'{depth_name}.png'))
                data.append((rgbd, np.array(src_transform)))

                # data.append((depth, camera_pose))

                # 裁剪物体,确定边界框的大小
                with open(os.path.join(demo_dir, f'bounding_box.json')) as f:
                    bound_box = json.load(f)
                
                # ---------------------laptop-------------------------------
                min_bound = np.asarray(bound_box["min"]) - np.asarray([0.2, 0.2, 0])
                max_bound = np.asarray(bound_box["max"]) + np.asarray([0.2, 0.2, 0.4])
                #-------------------stapler----------------------
                # min_bound = np.asarray(bound_box["min"]) - np.asarray([0.2, 0.2, 0])
                # max_bound = np.asarray(bound_box["max"]) + np.asarray([0.2, 0.2, 0.4])
                #--------------------drawer------------------------------------
                # min_bound = np.asarray(bound_box["min"]) - np.asarray([0.3, 0.3, 0])
                # max_bound = np.asarray(bound_box["max"]) + np.asarray([0.3, 0.3, 0.1])
                # # #--------------------faucet------------------------------------
                # min_bound = np.asarray(bound_box["min"]) - np.asarray([0.3, 0.3, 0])
                # max_bound = np.asarray(bound_box["max"]) + np.asarray([0.3, 0.3, 0.15])

                # min_bound = np.asarray(bound_box["min"]) 
                # max_bound = np.asarray(bound_box["max"]) 
            
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
                bbox = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
                bbox.scale(env._articulation_config.scale, np.zeros(3))
                bbox.translate(root_pose.p)
                bbox.rotate(root_pose.to_transformation_matrix()[:3, :3])

    #------------------------------tsdf-----------------------------
                # volume = o3d.pipelines.integration.UniformTSDFVolume(
                #     length=0.3,
                #     resolution = 120,
                #     sdf_trunc=0.01,
                #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
                # )
                pc_list = []

                volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length = 0.3/120,
                    sdf_trunc=0.04,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
                )
                volume.integrate(rgbd,intrinsics,np.linalg.inv(camera_pose))
                # pc = volume.extract_point_cloud()
                pc = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), intrinsics, np.linalg.inv(camera_pose))
                pc = pc.crop(bbox)
                pc_list.append(pc)
                pcd = sum_downsample_points(pc_list, 0.005, 50, 0.1)  # 交互前

                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)
                o3d.visualization.draw_geometries([pcd, mesh_frame])
                pc = np.asarray(pcd.points)

                # pc_end = np.asarray(dst_pcd.points)
                np.save(os.path.join(npy_data_dir, pc_name), pc)
                # np.save(os.path.join(npy_data_dir,'pc_end'),pc_end)
                # pc_seg, pc_seg_points = get_pc_seg(mesh_root, pc.shape[0])
                # np.save(os.path.join(npy_data_dir,pc_seg_name), pc_seg)
                # np.save(os.path.join(npy_data_dir, 'pc_seg_points_from_mesh'), pc_seg_points)

            gen_npz(npy_path=npy_data_dir, npz_path=npz_data_dir, model_name=different_angle_demo_name)
    env.close()


if __name__ == "__main__":
    config_dir = ASSET_DIR/ f"Ditto_dataset_configs/{config_folder_name}/{sub_dataset}"
    # config_name_without_scale = 'a'
    # for s in sorted(config_dir.glob("*")):
    #     gen_single_scene(config_name=s.name[:-4])

    s = sorted(config_dir.glob('*'))[0]
    gen_single_scene(config_name=s.name[:-4])

    # gen_single_scene(config_name='00010-1-0')
