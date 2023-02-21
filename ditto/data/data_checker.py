import numpy as np
import open3d as o3d
from pathlib import Path

data_dir = Path('~/Sim2Real2/ditto/data').expanduser().resolve()
data = np.load(data_dir/'faucet_stereo/train/scenes/0011_Faucet_0.npz', allow_pickle=True)
# data = np.load(data_dir/'laptop_stereo/train/scenes/10211_Laptop_0.npz', allow_pickle=True)
# data = np.load(data_dir/'drawer_stereo/train/scenes/0007_Drawer_0.npz', allow_pickle=True)


def visualize_point_cloud(points, name='point_cloud'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)
    o3d.visualization.draw_geometries([pcd, mesh_frame], window_name=name)

def load(name):
    return data[name]

def show_start_occ():
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)
    start_occ = load('start_p_occ')
    pcd_start_occ = o3d.geometry.PointCloud()
    pcd_start_occ.points = o3d.utility.Vector3dVector(start_occ)
    pcd_start_occ.paint_uniform_color([0,1,0])
    o3d.visualization.draw_geometries([pcd_start_occ, mesh_frame])
    for i in load('start_occ_list'):
        # visualize_point_cloud(load('start_p_occ')[i])
        pcd_part = o3d.geometry.PointCloud()
        pcd_all = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(load('start_p_occ')[i])
        pcd_all.points = o3d.utility.Vector3dVector(load('pc_start'))
        pcd_all.paint_uniform_color([1, 0.706, 0])
        pcd_part.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd_start_occ, pcd_all, pcd_part, mesh_frame])

def show_start_seg():
    joint_index = load('joint_index') + 1
    occ_list = load('start_occ_list')
    p_occ = load('start_p_occ')
    seg_label = occ_list[joint_index]
    occ_label = np.zeros(occ_list.shape[1], dtype=bool)
    for i in range(occ_list.shape[0]):
        if i == joint_index:
            continue
        occ_label = np.bitwise_or(occ_list[i], occ_label)
    pcd_part = o3d.geometry.PointCloud()
    pcd_all = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_occ[seg_label])
    pcd_all.points = o3d.utility.Vector3dVector(p_occ[occ_label])
    print('seg:', p_occ[seg_label].shape)
    print('occ:', p_occ[occ_label].shape)
    pcd_all.paint_uniform_color([1, 0.706, 0])
    pcd_part.paint_uniform_color([1, 0, 0])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)
    o3d.visualization.draw_geometries([pcd_all, pcd_part, mesh_frame])
    pcd_start = o3d.geometry.PointCloud()
    pcd_start.points = o3d.utility.Vector3dVector(load('pc_start'))
    o3d.visualization.draw_geometries([pcd_start,pcd_all, pcd_part, mesh_frame])
    

def show_end_seg():
    joint_index = load('joint_index') + 1
    occ_list = load('end_occ_list')
    p_occ = load('end_p_occ')
    seg_label = occ_list[joint_index]
    occ_label = np.zeros(occ_list.shape[1], dtype=bool)
    for i in range(occ_list.shape[0]):
        if i == joint_index:
            continue
        occ_label = np.bitwise_or(occ_list[i], occ_label)
    pcd_part = o3d.geometry.PointCloud()
    pcd_all = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_occ[seg_label])
    pcd_all.points = o3d.utility.Vector3dVector(p_occ[occ_label])
    pcd_all.paint_uniform_color([1, 0.706, 0])
    pcd_part.paint_uniform_color([1, 0, 0])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)
    o3d.visualization.draw_geometries([pcd_all, pcd_part, mesh_frame])
    pcd_end = o3d.geometry.PointCloud()
    pcd_end.points = o3d.utility.Vector3dVector(load('pc_end'))
    o3d.visualization.draw_geometries([pcd_end,pcd_all, pcd_part, mesh_frame])


def show_start_axis():
    start_pcd = o3d.geometry.PointCloud()
    start_pcd.points = o3d.utility.Vector3dVector(load('pc_start'))
    screw_axis = load('screw_axis')
    screw_moment = load('screw_moment')
    pivot_point = np.cross(screw_axis, screw_moment)
    axis_0_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=pivot_point)
    axis_1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=pivot_point + 0.3 * screw_axis)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0]*3)
    o3d.visualization.draw_geometries([start_pcd, origin_frame, axis_0_frame, axis_1_frame])

# show_start_occ()
visualize_point_cloud(load('pc_start'))
visualize_point_cloud(load('pc_end'))

show_start_occ()

show_start_seg()
show_end_seg()
# show_end_seg()
# show_start_occ()
# show_start_axis()

# visualize_point_cloud(load('pc_end'))


# print(load('start_mesh_pose_dict'))

# print(load('start_occ_list'))
# print(load('end_occ_list'))
