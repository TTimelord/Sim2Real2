from mani_skill2.envs.CEM_env import CEMEnv
from mani_skill2.envs.CEM_tool_env import CEMToolEnv

from argparse import ArgumentParser

from mani_skill2 import DIGITAL_TWIN_CONFIG_DIR

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import os

def main():
    digital_twin_config_path = os.path.join(DIGITAL_TWIN_CONFIG_DIR, 'drawer_video_1.yaml')
    # trajectory_path = os.path.join(Path(__file__).parent, 'trajectory', 'drawer_video_1.h5')
    trajectory_path = '/home/guest2/Documents/Sim2Real2/CEM/ManiSkill2-Learn/work_dirs/CEM-v0/20230520_180119/trajectory.h5'

    f = h5py.File(trajectory_path, 'r')
    states = f['/traj_0/dict_str_env_states']

    traj = states[:, 26:34]

    use_tool = False
    if use_tool:
        env = CEMToolEnv(
            articulation_config_path=digital_twin_config_path,
            obs_mode="state_dict",
            reward_mode="dense",
            sim_freq=500,
            control_freq=20
        )
    else:
        env = CEMEnv(
            articulation_config_path=digital_twin_config_path,
            obs_mode="state_dict",
            reward_mode="dense",
            sim_freq=500,
            control_freq=20
        )
    env.reset()
    viewer = env.render()
    # viewer.set_camera_xyz()
    # viewer.set_camera_rpy()
    # print(env._target_joint.get_global_pose().to_transformation_matrix()[:3, 0])
    # action = {'control_mode':'pd_joint_pos', 'action':np.zeros(9)}
    # obs, rew, done, info = env.step(action)
    # print('obs:', obs)
    # print('reward:', rew)

    # visualize point cloud in sapien
    # import open3d as o3d
    # N = 2000
    # pcd = o3d.io.read_point_cloud('pcd/faucet_exp_21/pcd_2.pcd')
    # rotate = True
    # if rotate:
    #     R = np.array([[-1, 0, 0],
    #                 [0, -1, 0],
    #                 [0, 0, 1]])
    #     # pcd = pcd.rotate(R, center=[0,0,0])
    #     # pcd = pcd.translate(np.array([2, 2, 0])*np.array(env._articulation_config.root_pos))
    # pc_np = np.array(pcd.points)
    # np.random.shuffle(pc_np)
    # pc_np = pc_np[:2000]
    # colors = np.zeros((N, 4))
    # colors[:, 0] = 1
    # colors[:, 1] = 0.5
    # scales = np.ones(N) * 0.01
    # sapien_pcd = env._scene.add_particle_entity(pc_np.astype(np.float32))
    # sapien_pcd.visual_body.set_attribute("color", colors.astype(np.float32))
    # sapien_pcd.visual_body.set_attribute("scale", scales.astype(np.float32))

    print("Press [e] to execute traj")
    while True:
        if viewer.window.key_down("e"):
            break
        env.render()
        env.step(None)

    env.agent._robot.set_qpos(np.append(traj[0], traj[0, -1]))
    env.agent._robot.set_qvel(np.zeros(9))

    curr_qpos = env.agent._robot.get_qpos()[:7]
    for qpos in tqdm(traj):
        target_qpos = qpos[:7]
        for substep in range(20):
            action = {}
            action['control_mode'] = 'pd_joint_pos'
            action['action'] = np.append(curr_qpos + (substep/20.0)*(target_qpos - curr_qpos), qpos[-1])
            # if substep // 1 == 0:
            env.render()
            env.step(action)
        curr_qpos = target_qpos
        # while True:
        #     env.render()
        #     if viewer.window.key_press("e"):
        #         break

    while True:
        env.render()
        if viewer.window.key_press("e"):
            break


if __name__ == "__main__":
    main()