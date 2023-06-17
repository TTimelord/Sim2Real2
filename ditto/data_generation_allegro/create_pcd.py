"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import shutil
import numpy as np
from argparse import ArgumentParser
import open3d as o3d
import transforms3d as t3d

import sapien.core as sapien
from sapien.core import Pose
from env import AllegroEnv, ContactError
from camera import AllegroCamera
from sensor import AllegroSensor
from utils import get_actor_meshes_visual


def get_random_qpos():
    start_open_extent = np.random.rand()
    print("start_open_extent:", start_open_extent)

    open_extent_diff_min = 0.05
    open_extent_diff_max = 0.4

    while True:
        end_open_extent = np.random.rand()
        print('end_open_extent：', end_open_extent)
        if abs(start_open_extent - end_open_extent) > open_extent_diff_min \
                and abs(start_open_extent - end_open_extent) < open_extent_diff_max:
            break

    start_qpos = np.zeros(env.object.dof)
    end_qpos = np.zeros(env.object.dof)
    for i in range(env.object.dof):
        start_qpos[i] = env.object.get_active_joints()[i].get_limits()[0][0]
        end_qpos[i] = env.object.get_active_joints()[i].get_limits()[0][0]
    lmin = env.target_joint.get_limits()[0][0]
    lmax = env.target_joint.get_limits()[0][1]
    start_qpos[env.target_joint_id] = (
            lmin + (lmax - lmin) * start_open_extent
    )
    end_qpos[env.target_joint_id] = (
            lmin + (lmax - lmin) * end_open_extent
    )
    print('start_qpos: %f, end_qpos: %f' % (start_qpos[env.target_joint_id], end_qpos[env.target_joint_id]))
    return start_qpos, end_qpos


def get_mesh_pose_list():
    mesh_pose_list = []
    movable_links = []
    for j in env.active_joints:
        movable_links.append(j.get_child_link())

    # for our objects we can find base_link simply:
    base_link = env.active_joints[0].get_parent_link()
    mesh = get_actor_meshes_visual(base_link)
    pose = base_link.get_pose().to_transformation_matrix()
    mesh_pose_list.append((mesh, pose))

    # movable_links
    for link in movable_links:
        mesh = get_actor_meshes_visual(link)
        pose = link.get_pose().to_transformation_matrix()
        mesh_pose_list.append((mesh, pose))
    return mesh_pose_list


def get_mesh_pose_list_from_pre_generated_mesh(use_fixed):
    if use_fixed:
        mesh_dir = 'assets/urdf/%s/%s/part_mesh_fixed/' % (category, shape_id)
    else:
        mesh_dir = 'assets/urdf/%s/%s/part_mesh/' % (category, shape_id)
    mesh_pose_list = []
    movable_links = []
    for j in env.active_joints:
        movable_links.append(j.get_child_link())

    # for our objects we can find base_link simply:
    base_link = env.active_joints[0].get_parent_link()
    mesh = o3d.io.read_triangle_mesh(mesh_dir + 'base_link.obj')
    print('watertight:', mesh.is_watertight())
    pose = base_link.get_pose().to_transformation_matrix()
    mesh_pose_list.append((mesh, pose))

    # movable_links
    for i, link in enumerate(movable_links):
        mesh = o3d.io.read_triangle_mesh(mesh_dir + 'link_%s.obj' % i)
        pose = link.get_pose().to_transformation_matrix()
        mesh_pose_list.append((mesh, pose))
    return mesh_pose_list


def get_occ(n_iou_points):
    mesh_pose_list = get_mesh_pose_list()
    # mesh_pose_list = get_mesh_pose_list_from_pre_generated_mesh(use_fixed=True)
    mesh_list = get_meshes(mesh_pose_list)
    points_occ, occ, occ_list, bounds = sample_iou_points(
        mesh_list, n_iou_points)
    return points_occ, occ_list


def get_meshes(mesh_pose_list):
    mesh_list = []
    for mesh, pose in mesh_pose_list:
        # mesh: o3d mesh
        mesh.transform(pose)
        mesh_list.append(mesh)
    return mesh_list


def sample_iou_points(mesh_list, num_point, padding=0.1, uniform=False, size=0.3):
    mesh_merged = o3d.geometry.TriangleMesh()
    for mesh in mesh_list:
        mesh_merged += mesh
    bbox = mesh_merged.get_axis_aligned_bounding_box()
    bounds = np.vstack((bbox.get_min_bound().astype(
        np.float32), bbox.get_max_bound().astype(np.float32)))

    occ_list = []
    points = np.random.rand(num_point, 3).astype(np.float32)
    if uniform:
        points *= size + 2 * padding
        points -= padding
    else:
        # points = points * (bounds[[1]] + 2 * padding - bounds[[0]]) + bounds[[0]] - padding
        points = points * np.max(bounds[1] + 2 * padding - bounds[0]) + bounds[[0]] - padding

    occ = np.zeros(num_point, dtype=bool)
    for mesh in mesh_list:
        occi = check_mesh_contains(mesh, points)

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


def check_mesh_contains(mesh, points):
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
    global_pose = env.target_joint.get_global_pose()

    rot_mat: np.ndarray = t3d.quaternions.quat2mat(global_pose.q)

    screw_axis = rot_mat[:3, 0]
    pivot_point = global_pose.p
    screw_moment = np.cross(pivot_point, screw_axis)
    return screw_axis, screw_moment


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('shape_id', type=str)
    parser.add_argument('category', type=str)
    parser.add_argument('cnt_id', type=int)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--stereo_out_dir', type=str)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
    args = parser.parse_args()

    pcd_path = f"real_test/real_datasets_allegro/{args.category.lower()}/video_{args.cnt_id}"
    if not os.path.exists(pcd_path):
        os.mkdir(pcd_path)
        os.mkdir(os.path.join(pcd_path, "digital_twin"))

    shape_id = args.shape_id
    # out_dir = args.out_dir
    category = args.category
    # stereo_out_dir = args.stereo_out_dir
    # if args.no_gui:
    #     out_name = out_dir + '/%s_%s_%d.npz' % (shape_id, category, args.cnt_id)
    #     stereo_out_name = stereo_out_dir + '/%s_%s_%d.npz' % (shape_id, category, args.cnt_id)
    # else:
    #     out_name = 'results' + '/%s_%s_%d.npz' % (shape_id, category, args.cnt_id)
    #     stereo_out_name = 'results/stereo' + '/%s_%s_%d.npz' % (shape_id, category, args.cnt_id)
    # out_dict = dict()
    # out_dict_stereo_depth = dict()

    # set random seed
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        # out_dict['random_seed'] = args.random_seed

    # setup env
    env = AllegroEnv(show_gui=(not args.no_gui), object_position_offset=0.5)

    # load shape
    urdf_name = [i for i in os.listdir('assets/urdf/%s/%s' % (args.category, shape_id)) if i[-4:] == 'urdf'][0]
    object_urdf_fn = 'assets/urdf/%s/%s/%s' % (args.category, shape_id, urdf_name)
    object_material = env.get_material(0.0, 0.0, 0.01)
    # state = 'closed'
    # state = 'random-open-single'
    # if np.random.random() < 0.5:
    #     state = 'closed'

    # env.load_object(object_urdf_fn, object_material, scale=0.5, q=[0, 0, 0, 1])
    env.load_object(object_urdf_fn, object_material, scale=0.5)
    # setup camera
    # cam = Camera(env, fixed_position=True)
    cam = AllegroSensor(env, fixed_position=True)
    # env.take_rgb_picture(None)
    if not args.no_gui:
        # env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)
        env.set_controller_camera_pose(1.0, 0.0, 1.2, np.pi, -0.5)
    # env.take_rgb_picture(None)
    # out_dict['joint_index'] = env.target_joint_id
    target_joint_type = env.target_joint.type
    joint_type = {
        'prismatic': 1,
        'revolute': 0
    }.get(target_joint_type)
    # out_dict['joint_type'] = joint_type

    start_qpos, end_qpos = get_random_qpos()
    screw_axis, screw_moment = get_screw()
    # out_dict['screw_axis'] = screw_axis
    # out_dict['screw_moment'] = screw_moment

    # generate data before and after interaction
    for qpos_idx in range(2):
        if qpos_idx == 0:
            env.object.set_qpos(start_qpos)
            state = start_qpos[env.target_joint_id]
            occ_list_name = 'start_occ_list'
            p_occ_name = 'start_p_occ'
            state_name = 'state_start'
            pc_name = 'pc_start'
            pc_seg_name = 'pc_seg_start'

        else:
            env.object.set_qpos(end_qpos)
            state = end_qpos[env.target_joint_id]
            occ_list_name = 'end_occ_list'
            p_occ_name = 'end_p_occ'
            state_name = 'state_end'
            pc_name = 'pc_end'
            pc_seg_name = 'pc_seg_end'

        # for debug:
        # viewer = env.render()
        # print('Press [e] to continue')
        # while True:
        #     if viewer.window.key_down('e'):
        #         break
        #     env.render()

        # out_dict[state_name] = state

        n_iou_points = 100000
        points_occ, occ_list = get_occ(n_iou_points)

        # out_dict[p_occ_name] = points_occ
        # out_dict[occ_list_name] = occ_list

        # get pc from depth sensor
        env.render()
        # rgb, depth, points = cam.get_observation()
        clean_depth, stereo_depth, rgb = cam.get_observation()
        intrinsics = cam.get_intrinsic()
        extrinsics = cam.get_extrinsic()

        intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsics_o3d.intrinsic_matrix = intrinsics

        # volume = o3d.pipelines.integration.ScalableTSDFVolume(
        #     voxel_length = 0.3/120,
        #     sdf_trunc=0.04,
        #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        # )
        # volume.integrate(rgbd,intrinsics,np.linalg.inv(camera_pose))
        # pc = volume.extract_point_cloud()

        pcd_clean = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(clean_depth), intrinsics_o3d,
                                                                    extrinsics)
        pcd_filename = os.path.join(pcd_path, f"pcd_{qpos_idx + 1}.pcd")
        o3d.io.write_point_cloud(pcd_filename, pcd_clean)
    #     pcd_stereo = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(stereo_depth), intrinsics_o3d,
    #                                                                  extrinsics)
    #
    #     pcd_clean = pcd_clean.voxel_down_sample(voxel_size=0.002)
    #     pcd_stereo = pcd_stereo.voxel_down_sample(voxel_size=0.002)
    #
    #     cl, ind = pcd_clean.remove_statistical_outlier(nb_neighbors=80,
    #                                                    std_ratio=2.0)
    #     pcd_clean = pcd_clean.select_by_index(ind)
    #
    #     cl, ind = pcd_stereo.remove_statistical_outlier(nb_neighbors=80,
    #                                                     std_ratio=2.0)
    #     pcd_stereo = pcd_stereo.select_by_index(ind)
    #
    #     # manual scale:
    #     direction = -cam.pos / np.linalg.norm(cam.pos)
    #     if args.category == 'Drawer':
    #         pcd_stereo.translate(direction * 0.015)
    #     else:
    #         pcd_stereo.translate(direction * 0.003)
    #
    #     # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0]*3)
    #     # o3d.visualization.draw_geometries([pcd_clean, mesh_frame])
    #     # o3d.visualization.draw_geometries([pcd_stereo, mesh_frame])
    #     # import cv2
    #     # cv2.imshow('asd', rgb)
    #     # cv2.waitKey(0)
    #     # cv2.destroyWindow('asd')
    #
    #     pc_clean = np.asarray(pcd_clean.points)
    #     pc_stereo = np.asarray(pcd_stereo.points)
    #     print('pc_clean.shape[0]:', pc_clean.shape[0])
    #     print('pc_stereo.shape[0]:', pc_stereo.shape[0])
    #
    #     if pc_clean.shape[0] > 10000:
    #         index = np.random.choice(pc_clean.shape[0], 10000, replace=False)
    #         pc_clean = pc_clean[index]
    #
    #     if pc_stereo.shape[0] > 10000:
    #         index = np.random.choice(pc_stereo.shape[0], 10000, replace=False)
    #         pc_stereo = pc_stereo[index]
    #
    #     out_dict[pc_name] = pc_clean
    #     out_dict_stereo_depth[pc_name] = pc_stereo
    #
    # for key in out_dict:
    #     if key not in out_dict_stereo_depth:
    #         out_dict_stereo_depth[key] = out_dict[key]
    #
    # # save results
    # np.savez(out_name, **out_dict)
    # np.savez(stereo_out_name, **out_dict_stereo_depth)
    env.close()  # segmentation fault
