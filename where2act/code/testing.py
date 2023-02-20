import os
from argparse import ArgumentParser
import numpy as np
import torch
import utils
from utils import get_global_position_from_camera
from sapien.core import Pose
from env import Env, ContactError
from camera import Camera
from robots.robotiq_robot import Robot
import open3d as o3d
import matplotlib.pyplot as plt

from pointnet2_ops.pointnet2_utils import furthest_point_sample


# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--model_version', type=str)
parser.add_argument('--model_epoch', type=int, help='epoch')
parser.add_argument('--shape_id', type=str, help='shape id')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')
parser.add_argument('--primact_type', type=str)
testing_conf = parser.parse_args()

primact_type = testing_conf.primact_type

# load train config
train_conf = torch.load(os.path.join('logs', testing_conf.exp_name, 'conf.pth'))

# load model
model_def = utils.get_model_module(testing_conf.model_version)

# set up device
device = torch.device(testing_conf.device)
print(f'Using device: {device}')

# create models
network = model_def.Network(train_conf.feat_dim, train_conf.rv_dim, train_conf.rv_cnt)

# load pretrained model
print('Loading ckpt from ', os.path.join('logs', testing_conf.exp_name, 'ckpts'), testing_conf.model_epoch)
data_to_restore = torch.load(os.path.join('logs', testing_conf.exp_name, 'ckpts', '%d-network.pth' % testing_conf.model_epoch))
network.load_state_dict(data_to_restore, strict=False)
print('DONE\n')

# send to device
network.to(device)

# set models to evaluation mode
network.eval()

# setup env
env = Env(zero_gravity=True)

# setup camera
cam = Camera(env, random_position=True)
# cam = Camera(env, fixed_position=True)
#cam = Camera(env)
env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)
mat33 = cam.mat44[:3, :3]

# load shape
# object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % eval_conf.shape_id
object_urdf_fn = '../urdf/selected_data/faucet/%s/mobility.urdf' % testing_conf.shape_id
object_material = env.get_material(4, 4, 0.01)
state = 'random-middle'
# state = 'closed'
# state = 'random-open-single'
# if np.random.random() < 0.5:
#     state = 'closed'
print('Object State: %s' % state)
env.load_object(object_urdf_fn, object_material, state=state, scale = 0.2)
cur_qpos = env.get_object_qpos()

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    cur_new_qpos = env.get_object_qpos()
    invalid_contact = False
    for c in env.scene.get_contacts():
        for p in c.points:
            if abs(p.impulse @ p.impulse) > 1e-4:
                invalid_contact = True
                break
        if invalid_contact:
            break
    if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_qpos = cur_new_qpos
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, 'Object Not Still!')
    flog.close()
    env.close()
    exit(1)


### use the GT vision
rgb, depth, _ = cam.get_observation()
object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
gt_handle_mask = cam.get_handle_mask()

# prepare input pc
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
mask = (cam_XYZA[:, :, 3] > 0.5)
pc = cam_XYZA[mask, :3]
grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448
idx = np.arange(pc.shape[0])
np.random.shuffle(idx)
while len(idx) < 30000:
    idx = np.concatenate([idx, idx])
idx = idx[:30000-1]
pc = pc[idx, :]
# pc[:, 0] -= 5
center = pc.mean(axis=0)
pc -= center
# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pc)
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=0.5, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd, mesh_frame])

pc = torch.from_numpy(pc).unsqueeze(0).to(device)

input_pcid = furthest_point_sample(pc, train_conf.num_point_per_shape).long().reshape(-1)
pc = pc[:, input_pcid, :3]  # 1 x N x 3

# push through unet
feats = network.pointnet2(pc.repeat(1, 1, 2))[0].permute(1, 0)    # N x F

# setup robot
robot_urdf_fn = './robots/robotiq_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))

# object_link_ids = env.movable_link_ids
# gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
# env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1])

with torch.no_grad():
    # push through the network
    pred_action_score_map = network.inference_action_score(pc)[0] # N
    pred_action_score_map = pred_action_score_map.cpu().numpy()

    best_action_point_index = np.argsort(pred_action_score_map)[-1]
    best_action_point = pc[0].cpu().numpy()[best_action_point_index]
    best_action_point += center

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

best_action_point = np.append(best_action_point, 1)
position_world = (cam.get_metadata()['mat44'] @ best_action_point)[:3]
print('position_world:', position_world)

action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ best_gripper_direction_camera
print('action_direction_world:', action_direction_world)
action_forward_direction_world = cam.get_metadata()['mat44'][:3, :3] @ best_gripper_forward_direction_camera

# compute final pose
up = np.array(action_direction_world, dtype=np.float32)
forward = action_forward_direction_world
left = np.cross(up, forward)
left /= np.linalg.norm(left)
forward = np.cross(left, up)
forward /= np.linalg.norm(forward)
rotmat = np.eye(4).astype(np.float32) # mlq:this is actually a transformation matrix
rotmat[:3, 0] = forward
rotmat[:3, 1] = left
rotmat[:3, 2] = up


final_dist = 0.0
if primact_type == 'pushing-left':
    final_dist = -0.01
    foward_offset = 0.1

final_rotmat = np.array(rotmat, dtype=np.float32)
final_rotmat[:3, 3] = position_world - action_direction_world * final_dist - forward * foward_offset
final_root_rotmat = final_rotmat.copy()
final_root_rotmat[:3, 3] = final_rotmat[:3, 3] - action_direction_world * 0.225
final_root_pose = Pose().from_transformation_matrix(final_root_rotmat)

start_rotmat = np.array(rotmat, dtype=np.float32)
start_rotmat[:3, 3] = position_world - action_direction_world * 0.15 - forward * foward_offset
start_root_rotmat = start_rotmat.copy()
start_root_rotmat[:3, 3] = start_rotmat[:3, 3] - action_direction_world * 0.225
start_root_pose = Pose().from_transformation_matrix(start_root_rotmat)

action_direction = None
if 'left' in primact_type:
    action_direction = forward
elif 'up' in primact_type:
    action_direction = left
if primact_type == 'pushing':
    action_direction = up

if action_direction is not None:
    end_rotmat = np.array(rotmat, dtype=np.float32)
    end_rotmat[:3, 3] = position_world - action_direction_world * final_dist + action_direction * 0.04

# visualize result with open3d
cmap = plt.cm.get_cmap("jet")
resultcolors = cmap(pred_action_score_map)[:, :3]
result_pccolors = resultcolors
pc_cam = pc[0].cpu().numpy() + center

pcd_result = o3d.geometry.PointCloud()
pcd_result.points = o3d.utility.Vector3dVector(pc_cam)
pcd_result.transform(cam.get_metadata()['mat44'])
pcd_result.colors = o3d.utility.Vector3dVector(result_pccolors)
pcd_result.estimate_normals()

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.01, origin=np.zeros(3))
gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1, origin=np.zeros(3))
gripper_frame.translate(position_world)
gripper_frame.rotate(final_root_rotmat[:3, :3], center=position_world)
o3d.visualization.draw_geometries([pcd_result, gripper_frame, world_frame])

### viz the EE gripper position

# move to the final pose
robot.robot.set_root_pose(final_root_pose)
env.render()

# env.wait_to_start()

# move back
robot.robot.set_root_pose(start_root_pose)
env.render()

# env.wait_to_start()


### main steps

success = True

try:
    if 'pushing' in primact_type:
        robot.close_gripper()
    elif 'pulling' in primact_type:
        robot.open_gripper()

    # approach
    robot.move_to_target_pose(final_rotmat, 2000)
    robot.wait_n_steps(2000)

    if 'pulling' in primact_type:
        robot.close_gripper()
        robot.wait_n_steps(2000)
    
    if 'left' in primact_type or 'up' in primact_type:
        robot.move_to_target_pose(end_rotmat, 2000)
        robot.wait_n_steps(2000)
    
    if primact_type == 'pushing':
        robot.move_to_target_pose(end_rotmat, 2000)
        robot.wait_n_steps(2000)

    if primact_type == 'pulling':
        robot.move_to_target_pose(start_rotmat, 2000)
        robot.wait_n_steps(2000)

except ContactError:
    success = False

env.wait_to_start()

# close env
env.close()

