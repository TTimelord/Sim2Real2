import os
import sys
import shutil
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import utils
import open3d as o3d
from pointnet2_ops.pointnet2_utils import furthest_point_sample


# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--model_version', type=str)
parser.add_argument('--model_epoch', type=int, help='epoch')
parser.add_argument('--pointcloud_name', type=str, help='name of the real point cloud')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')
eval_conf = parser.parse_args()


# load train config
train_conf = torch.load(os.path.join('logs', eval_conf.exp_name, 'conf.pth'))

# load model
model_def = utils.get_model_module(eval_conf.model_version)

# set up device
device = torch.device(eval_conf.device)
print(f'Using device: {device}')

# check if eval results already exist. If so, delete it.
result_dir = os.path.join('logs', eval_conf.exp_name, f'visu_action_heatmap_proposals-{eval_conf.pointcloud_name}-model_epoch_{eval_conf.model_epoch}-{eval_conf.result_suffix}')
if os.path.exists(result_dir):
    if not eval_conf.overwrite:
        response = input('Eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
        if response != 'y':
            sys.exit()
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
print(f'\nTesting under directory: {result_dir}\n')

# create models
network = model_def.Network(train_conf.feat_dim, train_conf.rv_dim, train_conf.rv_cnt)

# load pretrained model
print('Loading ckpt from ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), eval_conf.model_epoch)
data_to_restore = torch.load(os.path.join('logs', eval_conf.exp_name, 'ckpts', '%d-network.pth' % eval_conf.model_epoch))
network.load_state_dict(data_to_restore, strict=False)
print('DONE\n')

# send to device
network.to(device)

# set models to evaluation mode
network.eval()


pointcloud_fn = '../data/real_pointcloud/faucet/%s.pcd' % eval_conf.pointcloud_name
pcd = o3d.io.read_point_cloud(pointcloud_fn)
base_cam_extrinsic = np.asarray(
    [[ 0.4426, -0.8964,  0.0217, -0.5123],
    [-0.4364, -0.2365, -0.8681,  0.4499],
    [ 0.7833,  0.3748, -0.4959,  0.4564],
    [ 0.,      0.,      0.,      1.    ]])
pcd.transform(base_cam_extrinsic)

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd, world_frame])
pc = np.asarray(pcd.points, dtype=np.float32)
# permutation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
pc[:,[0, 1, 2]] = pc[:,[2, 0, 1]]
pc[:,[1, 2]] = -pc[:,[1, 2]]
pc_cam = pc.copy()
pc_center = pc.mean(axis=0)
pc -= pc_center
pc = torch.from_numpy(pc).unsqueeze(0).to(device)
input_pcid = furthest_point_sample(pc, train_conf.num_point_per_shape).long().reshape(-1)
pc = pc[:, input_pcid, :3]  # 1 x N x 3
# push through unet
feats = network.pointnet2(pc.repeat(1, 1, 2))[0].permute(1, 0)    # N x F

with torch.no_grad():
    # push through the network
    pred_action_score_map = network.inference_action_score(pc)[0] # N
    pred_action_score_map = pred_action_score_map.cpu().numpy()

    cmap = plt.cm.get_cmap("jet")

    # save action_score_map
    fn = os.path.join(result_dir, 'action_score_map_full')
    pc_np = pc[0].cpu().numpy()
    utils.render_pts_label_png(fn,  pc_np, pred_action_score_map)
    resultcolors = cmap(pred_action_score_map)[:, :3]
    result_pccolors = resultcolors
    utils.export_pts_color_obj(fn,  pc_np, result_pccolors)

    best_action_point_index = np.argsort(pred_action_score_map)[-1]
    best_action_point = pc_np[best_action_point_index]

    pred_6d = network.inference_actor(pc, index=best_action_point_index)[0]  # RV_CNT x 6
    pred_Rs = network.actor.bgs(pred_6d.reshape(-1, 3, 2))    # RV_CNT x 3 x 3

    result_score = 0
    best_gripper_direction_camera = None
    best_gripper_forward_direction_camera = None
    for i in range(train_conf.rv_cnt):
        gripper_direction_camera = pred_Rs[i:i+1, :, 0]
        gripper_forward_direction_camera = pred_Rs[i:i+1, :, 1]
        result = network.inference_critic(pc, gripper_direction_camera, gripper_forward_direction_camera, abs_val=True, index=best_action_point_index).item()
        if result > result_score:
            result_score = result
            best_gripper_direction_camera = gripper_direction_camera.squeeze()
            best_gripper_forward_direction_camera = gripper_forward_direction_camera.squeeze()
    best_gripper_direction_camera = best_gripper_direction_camera.cpu().numpy()
    best_gripper_forward_direction_camera = best_gripper_forward_direction_camera.cpu().numpy()
    print('critic_score=', result)
print('best_gripper_direction_camera', best_gripper_direction_camera)
print('best_gripper_forward_direction_camera', best_gripper_forward_direction_camera)

# visualize result with open3d
pcd_result = o3d.geometry.PointCloud()
pcd_result.points = o3d.utility.Vector3dVector(pc_np)
pcd_result.colors = o3d.utility.Vector3dVector(result_pccolors)
pcd_result.estimate_normals()

up = np.array(best_gripper_direction_camera, dtype=np.float32)
forward = best_gripper_forward_direction_camera
left = np.cross(up, forward)
left /= np.linalg.norm(left)
forward = np.cross(left, up)
forward /= np.linalg.norm(forward)
rotmat = np.eye(4).astype(np.float32) # mlq:this is actually a transformation matrix
rotmat[:3, 0] = forward
rotmat[:3, 1] = left
rotmat[:3, 2] = up

gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=np.zeros(3))
final_rotmat = rotmat.copy()
final_rotmat[:3, 3] = best_action_point
gripper_frame.transform(final_rotmat)
o3d.visualization.draw_geometries([pcd_result, gripper_frame])

# transform to world

# best_action_point = np.append(best_action_point, 1)
# position_world = (cam.get_metadata()['mat44'] @ best_action_point)[:3]
# print('position_world:', position_world)

# action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ best_gripper_direction_camera
# print('action_direction_world:', action_direction_world)
# action_forward_direction_world = cam.get_metadata()['mat44'][:3, :3] @ best_gripper_forward_direction_camera

# compute final pose
# up = np.array(action_direction_world, dtype=np.float32)
# forward = action_forward_direction_world
# left = np.cross(up, forward)
# left /= np.linalg.norm(left)
# forward = np.cross(left, up)
# forward /= np.linalg.norm(forward)
# rotmat = np.eye(4).astype(np.float32) # mlq:this is actually a transformation matrix
# rotmat[:3, 0] = forward
# rotmat[:3, 1] = left
# rotmat[:3, 2] = up


# final_dist = 0.0
# if primact_type == 'pushing-left' or primact_type == 'pushing-up':
#     final_dist = 0.01

# final_rotmat = np.array(rotmat, dtype=np.float32)
# final_rotmat[:3, 3] = position_world - action_direction_world * final_dist
# final_root_rotmat = final_rotmat.copy()
# final_root_rotmat[:3, 3] = final_rotmat[:3, 3] - action_direction_world * 0.225
# final_root_pose = Pose().from_transformation_matrix(final_root_rotmat)

# start_rotmat = np.array(rotmat, dtype=np.float32)
# start_rotmat[:3, 3] = position_world - action_direction_world * 0.15
# start_root_rotmat = start_rotmat.copy()
# start_root_rotmat[:3, 3] = start_rotmat[:3, 3] - action_direction_world * 0.225
# start_root_pose = Pose().from_transformation_matrix(start_root_rotmat)

# action_direction = None
# if 'left' in primact_type:
#     action_direction = forward
# elif 'up' in primact_type:
#     action_direction = left
# if primact_type == 'pushing':
#     action_direction = up

# if action_direction is not None:
#     end_rotmat = np.array(rotmat, dtype=np.float32)
#     end_rotmat[:3, 3] = position_world - action_direction_world * final_dist + action_direction * 0.1





