import copy
from collections import OrderedDict
from typing import Dict

import numpy as np
import sapien.core as sapien

from sapien.core import Pose

from mani_skill2.envs.fixed_xmate3_env import (
    FixedXmate3RobotiqEnv,
)
from mani_skill2.utils.common import (
    convert_np_bool_to_float,
    flatten_state_dict,
    register_gym_env,
)
from mani_skill2.utils.contrib import (
    apply_pose_to_points,
    normalize_and_clip_in_interval,
    o3d_to_trimesh,
    trimesh_to_o3d,
)
from mani_skill2.utils.geometry import angle_distance
from mani_skill2.utils.o3d_utils import merge_mesh, np2mesh
from mani_skill2.utils.sapien_utils import get_pad_articulation_state
from mani_skill2.utils.tmu import register_gym_env_for_tmu


@register_gym_env("CEM_ablation-v0", max_episode_steps=500)
# @register_gym_env_for_tmu("CEM_@-v0")
class CEMAblationEnv(FixedXmate3RobotiqEnv):
    # SUPPORTED_OBS_MODES = ("state", "state_dict", "rgbd", "pointcloud")
    SUPPORTED_OBS_MODES = ("state", "state_dict")
    SUPPORTED_REWARD_MODES = ("dense", "sparse")

    # _open_extent = 0.1
    max_v = 0.1
    max_ang_v = 1.0
    _require_grasp = False

    _robot_init_qpos: np.ndarray

    def __init__(
        self,
        articulation_config_path,
        obs_mode=None,
        reward_mode=None,
        sim_freq=500,
        control_freq=20,
    ):
        super().__init__(
            articulation_config_path, obs_mode, reward_mode, sim_freq, control_freq
        )

    def _initialize_articulation(self):
        root_pos = self._articulation_config.root_pos
        rot_z = np.sin(self._articulation_config.root_ang/2)
        self._articulation.set_root_pose(
            Pose(
                root_pos,
                [np.sqrt(1 - rot_z**2), 0, 0, rot_z],
            )
        )

        [[lmin, lmax]] = self._target_joint.get_limits()
        qpos = np.zeros(self._articulation.dof)
        qpos[0] = self._articulation_config.init_qpos
        self._articulation.set_qpos(qpos)

        self.target_joint_idx = self._articulation_config.target_joint_idx
        self.target_joint_type = self._target_joint.type
        self.digital_twin_target_qpos = self._articulation_config.target_qpos
        self.digital_twin_init_qpos = self._articulation_config.init_qpos

        # get digital twin target link center (center translation in the link frame)
        visual_body = self._target_link.get_visual_bodies()[0]
        render_shape = visual_body.get_render_shapes()[0]
        vertices = apply_pose_to_points(
            render_shape.mesh.vertices * visual_body.scale,
            self._target_link.get_pose() * visual_body.local_pose,
        )
        mesh = np2mesh(vertices, render_shape.mesh.indices.reshape(-1, 3))
        bbox = mesh.get_oriented_bounding_box()

        target_link_center_pos = bbox.get_center()
        target_link_frame_pos = self._target_link.get_pose().p
        target_link_frame_rot = self._target_link.get_pose().to_transformation_matrix()[:3, :3]
        self._target_link_center_to_frame = np.linalg.inv(target_link_frame_rot) @ (target_link_center_pos-target_link_frame_pos)
        
        # get points far from joint axis
        pointing_dir = vertices - target_link_frame_pos # 30000 X 3
        axis_dir = target_link_frame_rot[:, 0] # 3
        distance = np.sqrt(np.linalg.norm(pointing_dir, axis=1)**2 - np.dot(pointing_dir, axis_dir)**2)
        index = np.argsort(distance)[-3000:]
        target_link_far_pos = np.mean(vertices[index], axis=0)
        self._target_link_far_point_to_frame = np.linalg.inv(target_link_frame_rot) @ (target_link_far_pos - target_link_frame_pos)

        # print(target_link_far_pos)
        # print(self._target_link_far_point_to_frame)

        # o3d visualization
        # import open3d as o3d
        # bbox.color = [1,0,0]
        # o3d.visualization.draw_geometries([mesh, bbox])

        # set physical properties for all the joints
        self._joint_friction_range = (0.8, 1.0) # if self.target_joint_type=='prismatic' else (0.8, 0.25)
        self._joint_stiffness_range = (0.0, 0.0)
        # self._joint_damping_range = (20.0, 30.0) # laptop
        # self._joint_damping_range = (70.0, 80.0) # drawer
        self._joint_damping_range = (4.0, 5.0) # faucet


        joint_friction = self._episode_rng.uniform(
            self._joint_friction_range[0], self._joint_friction_range[1]
        )
        joint_stiffness = self._episode_rng.uniform(
            self._joint_stiffness_range[0], self._joint_stiffness_range[1]
        )
        joint_damping = self._episode_rng.uniform(
            self._joint_damping_range[0], self._joint_damping_range[1]
        )

        for joint in self._articulation.get_active_joints():
            joint.set_friction(joint_friction)
            joint.set_drive_property(joint_stiffness, joint_damping)

    def _initialize_agent(self):
        # qpos = np.array([1.4, -1.053, -2.394, 1.662, 1.217, 1.05, -0.8, 0.0, 0.0])
        # qpos = np.array([1.302, -0.541, -2.935, 1.440, 1.560, 1.553, -0.804, 0.0, 0.0])
        # qpos = np.array([1.12187,-0.384234,-2.53946,1.12766,1.32165,1.5631,-0.0644407, 0.0, 0.0])
        # qpos = np.array([-0.488665,0.340129,-1.14341,1.18789,0.346452,1.73497,0.5321,0.0,0.0]) # drawer
        qpos = np.array([-0.488665,0.340129,-1.14341,1.18789,0.346452,1.73497,-1,0.0,0.0]) # laptop and faucet

        # qpos[:-2] += self._episode_rng.normal(0, 0.02, len(qpos) - 2)

        self._robot_init_qpos = qpos
        self._agent.reset(qpos)
        self._agent._robot.set_pose(Pose([0, 0, 0]))

    def _get_obs_state_dict(self) -> OrderedDict:
        state_dict = OrderedDict()
        agent_dict = self._agent.get_proprioception()
        agent_dict["ee_pos_base"] = (
            self._agent._robot.get_root_pose().inv() * self.grasp_site.get_pose()
        ).p
        agent_dict["ee_vel_base"] = self.grasp_site.get_velocity()
        state_dict.update(
            agent=agent_dict,  # proprioception
        )
        extra_dict = OrderedDict()
        # extra_dict["task"] = self._target_indicator
        extra_dict["articulation"] = get_pad_articulation_state(
            self.articulation, self._max_dof
        )
        state_dict["extra"] = extra_dict

        return state_dict

    def _get_obs_rgbd(self) -> OrderedDict:
        self.update_render()
        images = self._agent.get_images(depth=True, visual_seg=True, actor_seg=True)
        # generate masks
        for cam_name, cam_images in images.items():
            visual_id_seg = cam_images["visual_seg"]  # (n, m)
            actor_id_seg = cam_images["actor_seg"]  # (n, m)
            masks = [np.zeros(visual_id_seg.shape, dtype=np.bool) for _ in range(3)]
            for visual_id in self._handle_info["visual_body_ids"]:
                masks[0] = masks[0] | (visual_id_seg == visual_id)
            masks[1] = actor_id_seg == self._target_link.get_id()
            for actor_id in self._agent_actor_ids:
                masks[2] = masks[2] | (actor_id_seg == actor_id)

            images[cam_name]["handle_mask"] = masks[0]
            images[cam_name]["link_mask"] = masks[1]
            images[cam_name]["robot_mask"] = masks[2]

        rgbd_dict = OrderedDict()
        rgbd_dict["image"] = images
        state_dict = self._get_obs_state_dict()
        rgbd_dict["agent"] = state_dict["agent"]
        rgbd_dict["extra"] = self._get_visual_state()
        return rgbd_dict

    def _get_obs_pointcloud(self) -> OrderedDict:
        """get pointcloud from each camera, transform them to the *world* frame, and fuse together"""
        pcd_dict = OrderedDict()
        self.update_render()

        fused_pcd = self._agent.get_fused_pointcloud(visual_seg=True, actor_seg=True)

        pcds = []
        for _, camera in self._cameras.items():
            camera.take_picture()
            pcd = get_camera_pcd(camera, visual_seg=True, actor_seg=True)
            T = camera.get_model_matrix()
            pcd["xyz"] = transform_points(T, pcd["xyz"])
            pcds.append(pcd)

        if len(pcds) > 0:
            fused_pcd = merge_dicts([fused_pcd, merge_dicts(pcds, True)], True)

        # generate masks
        handle_mask = np.zeros(fused_pcd["actor_seg"].shape, dtype=np.bool)
        for visual_id in self._handle_info["visual_body_ids"]:
            handle_mask = handle_mask | (fused_pcd["visual_seg"] == visual_id)
        fused_pcd["handle_mask"] = handle_mask

        fused_pcd["link_mask"] = fused_pcd["visual_seg"] == self._target_link.get_id()
        actor_seg = fused_pcd["actor_seg"][:, None]
        diff = actor_seg == self._agent_actor_ids[None, :]
        agent_mask = np.logical_or.reduce(diff, axis=1)
        fused_pcd["robot_mask"] = agent_mask

        if self.hide_agent_in_pcd:
            for k, v in fused_pcd.items():
                fused_pcd[k] = v[np.logical_not(agent_mask)]

        pcd_dict["pointcloud"] = fused_pcd
        state_dict = self._get_obs_state_dict()
        pcd_dict["agent"] = state_dict["agent"]
        pcd_dict["extra"] = self._get_visual_state()
        return pcd_dict

    def get_obs(self):
        if self._obs_mode == "state":
            state_dict = self._get_obs_state_dict()
            return flatten_state_dict(state_dict)
        elif self._obs_mode == "state_dict":
            state_dict = self._get_obs_state_dict()
            return state_dict
        elif self._obs_mode == "rgbd":
            return self._get_obs_rgbd()
        elif self._obs_mode == "pointcloud":
            return self._get_obs_pointcloud()
        else:
            raise NotImplementedError(self._obs_mode)

    def _get_visual_state(self) -> OrderedDict:  # from ManiSkill1
        joint_pos = (
            self._articulation.get_qpos()[self._target_joint_idx] / self.digital_twin_target_qpos
        )
        current_handle = apply_pose_to_points(
            self._handle_info["pcd"], self.target_link.get_pose()
        ).mean(0)

        flag = self.compute_other_flag_dict()
        ee_close_to_handle = flag["ee_close_to_handle"]

        visual_state = OrderedDict()
        visual_state["joint_pos"] = np.array([joint_pos])
        visual_state["open_enough"] = convert_np_bool_to_float(joint_pos > 1.0)
        visual_state["ee_close_to_handle"] = convert_np_bool_to_float(
            ee_close_to_handle
        )
        visual_state["current_handle"] = current_handle

        return visual_state

    def compute_other_flag_dict(self):
        ee_cords = self._agent.sample_ee_coords()  # [2, 10, 3]
        # current_handle = apply_pose_to_points(
        #     self._handle_info["pcd"], self._target_link.get_pose()
        # )  # [200, 3]
        # ee_to_handle = ee_cords.mean(0)[:, None] - current_handle
        # dist_ee_to_handle = np.linalg.norm(ee_to_handle, axis=-1).min(-1).mean(-1)

        # handle_mesh = trimesh.Trimesh(
        #     vertices=apply_pose_to_points(
        #         np.asarray(self._handle_info["mesh"].vertices),
        #         self._target_link.get_pose(),
        #     ),
        #     faces=np.asarray(np.asarray(self._handle_info["mesh"].triangles)),
        # )

        # handle_obj = trimesh.proximity.ProximityQuery(handle_mesh)
        # sd_ee_mid_to_handle = handle_obj.signed_distance(ee_cords.mean(0)).max()
        # sd_ee_to_handle = (
        #     handle_obj.signed_distance(ee_cords.reshape(-1, 3)).reshape(2, -1).max(1)
        # )

        # Grasp = mid close almost in cvx and both sides has similar sign distance.
        close_to_grasp = sd_ee_to_handle.min() > -1e-2
        ee_in_grasp_pose = sd_ee_mid_to_handle > -1e-2
        grasp_happen = ee_in_grasp_pose and close_to_grasp

        other_info = {
            "dist_ee_to_handle": dist_ee_to_handle,
            "sd_ee_mid_to_handle": sd_ee_mid_to_handle,
            "sd_ee_to_handle": sd_ee_to_handle,
            "ee_close_to_handle_pre_rot": dist_ee_to_handle <= 0.06,
            "ee_close_to_handle": dist_ee_to_handle <= 0.03,
            "grasp_happen": grasp_happen,
        }
        other_info["dist_ee_to_handle_l1"] = (
            np.abs(ee_to_handle).sum(axis=-1).min(-1).min(-1)
        )  # [2]
        return other_info

    def compute_eval_flag_dict(self):
        flag_dict = OrderedDict()

        digital_twin_qpos = self._articulation.get_qpos()[self.target_joint_idx]

        if self.target_joint_type == "prismatic":
            flag_dict["target_achieved"] = np.isclose(digital_twin_qpos, self.digital_twin_target_qpos, 0, 0.005)
        else:
            flag_dict["target_achieved"] = np.isclose(digital_twin_qpos, self.digital_twin_target_qpos, 0, 0.02)

        # flag_dict["cabinet_static"] = convert_np_bool_to_float(
        #     self.check_actor_static(
        #         self._target_link, max_v=self.max_v, max_ang_v=self.max_ang_v
        #     )
        # )
        # flag_dict["open_enough"] = convert_np_bool_to_float(
        #     self._articulation.get_qpos()[self.target_joint_idx]
        #     >= self.target_qpos
        # )
        # if self._require_grasp:
        #     flag_dict["grasp_handle"] = self.compute_other_flag_dict()[
        #         "ee_close_to_handle"
        #     ]
        flag_dict["success"] = convert_np_bool_to_float(all(flag_dict.values()))
        return flag_dict

    def check_success(self):
        flag_dict = self.compute_eval_flag_dict()
        return flag_dict["success"]

    def compute_dense_reward(self):
        flag_dict = self.compute_eval_flag_dict()
        target_achieved_rew = flag_dict['target_achieved']
        digital_twin_qpos = self._articulation.get_qpos()[self.target_joint_idx]

        # contact reward
        contact_rew = 0.0
        contacts = self._scene.get_contacts()
        # contact_force = 0.0
        for contact in contacts:
            if (
                contact.actor0 in self.agent._robot.get_links()
                and contact.actor1 in self._articulation.get_links()
            ) or (
                contact.actor1 in self.agent._robot.get_links()
                and contact.actor0 in self._articulation.get_links()
            ):
                if (
                    (contact.actor0 == self.agent.finger1_link
                    and contact.actor1 == self.target_link)
                    or
                    (contact.actor0 == self.agent.finger2_link
                    and contact.actor1 == self.target_link)
                    or
                    (contact.actor1 == self.agent.finger1_link
                    and contact.actor0 == self.target_link)
                    or
                    (contact.actor1 == self.agent.finger2_link
                    and contact.actor0 == self.target_link)
                ):
                    contact_force = []
                    for point in contact.points:
                        if contact.actor0 == self.target_link:
                            contact_force.append(point.impulse * self.sim_freq)
                        else:
                            contact_force.append(-point.impulse * self.sim_freq)

                    if len(contact_force):  # contact_force may be empty
                        contact_force = np.vstack(contact_force)
                        norm = np.linalg.norm(contact_force, axis=1)

                        # deal with 0 contact force
                        contact_force = contact_force[norm>0]
                        if len(contact_force):  # contact force may be empty again.
                            if np.abs(digital_twin_qpos - self.digital_twin_target_qpos) < np.abs(self.digital_twin_init_qpos - self.digital_twin_target_qpos) and not target_achieved_rew:
                                contact_rew = 1.0
                    continue
                contact_rew = -2.0 # -6.0
                break

        # contact force reward
        contact_direction_rew = -self.accumulated_contact_direction_err # average force per sim step
        self.accumulated_contact_direction_err = 0.0

        # print contact force debug info
        # if np.random.rand() < 0.0005:
        #     print('accumu:', self.accumulated_contact_force)
        #     print('rew', contact_force_rew)


        # close to digital twin target_link reward
        grasp_frame_translation = self.grasp_site.get_pose().p
        digital_twin_target_pos = self._target_link.get_pose().p
        digital_twin_target_rot = self._target_link.get_pose().to_transformation_matrix()[:3, :3]
        # digital_twin_target_center = digital_twin_target_rot @ self._target_link_center_to_frame + digital_twin_target_pos
        digital_twin_target_center = digital_twin_target_rot @ (self._target_link_center_to_frame if self.target_joint_type == "prismatic" else self._target_link_far_point_to_frame) + digital_twin_target_pos
        close_to_target_link_rew = -np.clip(np.linalg.norm(grasp_frame_translation - digital_twin_target_center)**2, a_min=0.0, a_max=np.inf) # move to target_link. [-oo, 0]

        # close to target reward
            # we want the reward to be constant no matter the difference between target qpos and initial qpos
        close_to_target_qpos_rew = -np.abs(
                                            (digital_twin_qpos - self.digital_twin_target_qpos)/
                                            (self.digital_twin_init_qpos - self.digital_twin_target_qpos), 
                                            dtype=np.float32)

        # q_vel and q_acc punishment
        robot_q_acc = self.agent._robot.get_qacc() # [9, ]
        robot_q_acc_coefficient = np.array([2, 3, 2, 2, 2, 1, 1, 15, 15]) # [9, ]
        robot_q_acc_rew = -np.sum(np.abs(robot_q_acc)*robot_q_acc_coefficient)
        robot_q_vel = self.agent._robot.get_qvel() # [9, ]
        robot_q_vel_coefficient = np.array([2, 3, 2, 2, 2, 1, 1, 15, 15]) # [9, ]
        robot_q_vel_rew = -np.sum(np.abs(robot_q_vel)*robot_q_vel_coefficient)

        # digital twin qvel punishment
        digital_twin_qvel = self.articulation.get_qvel()[self.target_joint_idx]
        digital_twin_qvel_rew = -np.abs(digital_twin_qvel)

        # attract to initial qpos if success

        reward_dict = {
            "close_to_digital_twin_reward": close_to_target_link_rew * 10, # 10
            "close_to_target_reward": close_to_target_qpos_rew * 50, # 50
            "target_achieved_reward": target_achieved_rew * 20, # 20
            "contact_reward": contact_rew * 10, # 10
            "robot_q_vel_reward": robot_q_vel_rew * 0.03, # 0.03,
            "robot_q_acc_reward": robot_q_acc_rew * 0.01, # 0.01,
            "digital_twin_q_vel_reward:": digital_twin_qvel_rew * 0 # 5
        }
        reward = sum(reward_dict.values())
        # print('reward:', reward)
        info_dict = {}
        # info_dict["digital_twin_target_center"] = digital_twin_target_center
        info_dict['step_in_ep'] = np.array(self.step_in_ep, dtype=float) # convert to float for visualization in maniskill2_learn
        info_dict['accumulated_contact_direction_err'] = np.array(self.accumulated_contact_direction_err)
        for key in reward_dict:
            info_dict[key] = np.array(reward_dict[key])
        info_dict['total_reward']=reward
        if self.target_joint_type == "prismatic":
            info_dict['qpos_error(m)']=np.abs(self.digital_twin_target_qpos-digital_twin_qpos)
        else:
            info_dict['qpos_error(deg)']=np.rad2deg(np.abs(self.digital_twin_target_qpos-digital_twin_qpos))
        # info_dict['robot_q_vel_rew']=robot_q_vel_rew*0.3
        # info_dict['robot_q_acc_rew']=robot_q_acc_rew*0.02
        # info_dict.update(flag_dict)
        # info_dict.update(other_info)
        self._cache_info = info_dict
        return reward

    def get_reward(self):
        if self._reward_mode == "sparse":
            return float(self.check_success())
        elif self._reward_mode == "dense":
            return self.compute_dense_reward()
        else:
            raise NotImplementedError(self._reward_mode)

    def get_info(self):
        info = super().get_info()
        info.update(self._cache_info)
        return info

    @property
    def handle_info(self):
        return self._handle_info

    @property
    def table(self):
        return self._table


# @register_gym_env("FixedOpenCabinetDoorSensor-v0", max_episode_steps=500)
# class OpenCabinetDoorSensor(FixedXmate3RobotiqSensorEnv, OpenCabinetDoor):
#     pass


# @register_gym_env("FixedOpenCabinetDoorSensorLowRes-v0", max_episode_steps=500)
# class OpenCabinetDoorSensorLowRes(FixedXmate3RobotiqSensorLowResEnv, OpenCabinetDoor):
#     pass
