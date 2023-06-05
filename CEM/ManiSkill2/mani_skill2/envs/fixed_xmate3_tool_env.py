import copy
import os.path
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import attr
import numpy as np
import sapien.core as sapien
import yaml
from gym import spaces
from transforms3d.euler import euler2quat

from mani_skill2 import AGENT_CONFIG_DIR, DIGITAL_TWIN_DIR, ASSET_DIR
from mani_skill2.agents.camera import get_camera_images, get_camera_pcd, get_camera_rgb
from mani_skill2.agents.fixed_xmate3_robotiq_with_tool import FixedXmate3RobotiqTool
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import merge_dicts
from mani_skill2.utils.geometry import angle_distance, transform_points
from mani_skill2.utils.sapien_utils import (
    get_actor_state,
    get_articulation_state,
    get_entity_by_name,
    ignore_collision,
    set_actor_state,
    set_articulation_state,
)

@attr.s(auto_attribs=True, kw_only=True)
class DigitalTwinConfig:
    name: str
    urdf_path: str
    multiple_collisions: bool
    scale: float
    root_pos: np.ndarray
    root_ang: float
    target_joint_idx: int
    init_qpos: float
    target_qpos: float
    target_joint_axis: np.ndarray


class FixedXmate3RobotiqToolEnv(BaseEnv):
    SUPPORTED_OBS_MODES: Tuple[str] = ()
    SUPPORTED_REWARD_MODES: Tuple[str] = ()
    _agent: FixedXmate3RobotiqTool

    def __init__(
        self,
        articulation_config_path,
        obs_mode=None,
        reward_mode=None,
        sim_freq=500,
        control_freq=20,
    ):
        # articulation
        with open(articulation_config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            self._articulation_config = DigitalTwinConfig(**config_dict)
        self._max_dof = 1
        self.hide_agent_in_pcd = (
            False  # change it to hide agent in point cloud observation
        )

        self.step_in_ep = 0
        self._prev_actor_pose = sapien.Pose()
        self._cache_obs_state_dict: OrderedDict = (
            OrderedDict()
        )  # save obs state dict to save reward computation time
        self._cache_info = {}

        # for computing accumulated reward
        self.accumulated_contact_direction_err = 0.0

        super().__init__(obs_mode, reward_mode, sim_freq, control_freq)

    @property
    def agent(self):
        return self._agent

    @property
    def articulation(self):
        return self._articulation

    @property
    def target_link(self):
        return self._target_link

    @property
    def control_mode(self):
        return self._agent._control_mode

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    @property
    def obs_mode(self):
        return self._obs_mode

    def get_obs(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Agent
    # -------------------------------------------------------------------------- #

    def step_action(self, action):
        if action is None:
            pass
        elif isinstance(action, np.ndarray):
            self._agent.set_action(action)
        elif isinstance(action, dict):
            if action["control_mode"] != self._agent.control_mode:
                self._agent.set_control_mode(action["control_mode"])
            self._agent.set_action(action["action"])
        else:
            raise TypeError(type(action))
        for _ in range(self._sim_step_per_control):
            self._agent.simulation_step()
            self._scene.step()

            # compute contact reward
        #     contacts = self._scene.get_contacts()
        #     contact_force = []
        #     for contact in contacts:
        #         if (
        #             contact.actor0 in self.agent._robot.get_links()
        #             and contact.actor1 in self._articulation.get_links()
        #         ) or (
        #             contact.actor1 in self.agent._robot.get_links()
        #             and contact.actor0 in self._articulation.get_links()
        #         ):
        #             if (
        #                 (contact.actor0 == self.agent.finger1_link
        #                 and contact.actor1 == self.target_link)
        #                 or
        #                 (contact.actor0 == self.agent.finger2_link
        #                 and contact.actor1 == self.target_link)
        #                 or
        #                 (contact.actor1 == self.agent.finger1_link
        #                 and contact.actor0 == self.target_link)
        #                 or
        #                 (contact.actor1 == self.agent.finger2_link
        #                 and contact.actor0 == self.target_link)
        #             ):
        #                 for point in contact.points:
        #                     if contact.actor0 == self.target_link:
        #                         contact_force.append(point.impulse * self.sim_freq)
        #                     else:
        #                         contact_force.append(-point.impulse * self.sim_freq)

        # if len(contact_force):  # contact_force may be empty
        #     contact_force = np.vstack(contact_force)
        #     norm = np.linalg.norm(contact_force, axis=1)

        #     # deal with 0 contact force
        #     contact_force = contact_force[norm>0]
        #     norm = norm[norm>0]
        #     if len(contact_force):  # contact force may be empty again.
        #         norm_repeat = np.repeat(norm.T, 3).reshape(contact_force.shape)
        #         contact_force_normalized = contact_force / norm_repeat

        #         joint_axis = self._target_joint.get_global_pose().to_transformation_matrix()[:3, 0]
        #         dot_product = np.dot(contact_force_normalized, joint_axis)
        #         if self.target_joint_type == 'prismatic':
        #             contact_direction_err = np.sum(norm * (np.ones_like(dot_product) - np.abs(dot_product))) # sum of weighted contact direction err
        #         else: # revolute joint
        #             contact_direction_err = np.sum(norm * np.abs(dot_product))
        #         # print contact direction debug information
        #         if np.random.rand() < 0.001:
        #             print('joint_axis:', joint_axis)
        #             print('contact_force:', contact_force)
        #             print('norm:', norm)
        #             print('contact_force_normalized:', contact_force_normalized)
        #             print('dot_product:', dot_product)
        #             print('contact_direction_err:', contact_direction_err)
        #             print('\n')
        #         self.accumulated_contact_direction_err += contact_direction_err

        self._agent.update_generalized_external_forces()

    # -------------------------------------------------------------------------- #
    # Reward mode
    # -------------------------------------------------------------------------- #
    @property
    def reward_mode(self):
        return self._reward_mode

    @reward_mode.setter
    def reward_mode(self, mode: str):
        if mode not in self.SUPPORTED_REWARD_MODES:
            raise NotImplementedError("Unsupported reward mode: {}".format(mode))
        self._reward_mode = mode

    def get_reward(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def _clear(self):
        self._agent = None
        self._camera = None
        super()._clear()

    def _setup_physical_materials(self):
        self.add_physical_material("default", 1.0, 1.0, 0.0)
        self.add_physical_material("no_friction", 0.0, 0.0, 0.0)

    def _setup_render_materials(self):
        self.add_render_material(
            "ground",
            color=[0.5, 0.5, 0.5, 1],
            metallic=1.0,
            roughness=0.7,
            specular=0.04,
        )
        self.add_render_material(
            "default",
            color=[0.8, 0.8, 0.8, 1],
            metallic=0,
            roughness=0.9,
            specular=0.0,
        )

    def _load_articulation(self):
        loader = self._scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = (
            self._articulation_config.multiple_collisions
        )

        loader.scale = self._articulation_config.scale
        loader.fix_root_link = True

        config = {"material": self._physical_materials["no_friction"]}
        self._articulation = loader.load(
            os.path.join(DIGITAL_TWIN_DIR, self._articulation_config.urdf_path), config
        )
        self._articulation.set_name(self._articulation_config.name)

        self._target_joint = self._articulation.get_active_joints()[0]
        self._target_link = self._target_joint.get_child_link()

    def _load_agent(self):
        self._agent = FixedXmate3RobotiqTool.from_config_file(
            os.path.join(AGENT_CONFIG_DIR, "fixed_xmate3_robotiq_tool.yml"),
            self._scene,
            self._control_freq,
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self._agent._robot.get_links(), "grasp_convenient_link"
        )
        self._agent_actor_ids = []
        for actor in self._agent._robot.get_links():
            self._agent_actor_ids.append(actor.id)

        self._agent_actor_ids = np.array(self._agent_actor_ids).astype(int)

    def _initialize_articulation(self):
        raise NotImplementedError

    def _initialize_agent(self):
        raise NotImplementedError

    def _load_table(self):
        loader = self._scene.create_actor_builder()
        loader.add_visual_from_file(
            os.path.join(ASSET_DIR, "descriptions", "optical_table", "visual", "optical_table.dae")
        )
        loader.add_collision_from_file(
            os.path.join(ASSET_DIR, "descriptions", "optical_table", "visual", "optical_table.dae")
        )
        self._table = loader.build_static(name="table")
        # self._table.set_pose(sapien.Pose([0.0, 0.0, -0.04]))
        self._table.set_pose(sapien.Pose([0.6, -0.4, -0.065]))

    def reconfigure(self):
        self.step_in_ep = 0
        self._prev_actor_pose = sapien.Pose()
        self._clear()

        self._setup_scene()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_table()
        self._load_articulation()
        ignore_collision(self._articulation)
        self._load_agent()

        self._setup_lighting()
        self._setup_camera()
        if self._viewer is not None:
            self._setup_viewer()
        # cache actors and articulations for sim state
        self._actors = self.get_actors()
        self._articulations = self.get_articulations()
        # Cache initial simulation state
        self._initial_sim_state = self.get_sim_state()

    def reset(self, seed=None, reconfigure=False):
        if seed is None:
            self._episode_seed = self._main_rng.randint(2**32)
        else:
            self._episode_seed = seed
        self._episode_rng = np.random.RandomState(self._episode_seed)
        if reconfigure:
            self.reconfigure()
        else:
            self.set_sim_state(self._initial_sim_state)
        self._initialize_articulation()
        self._initialize_agent()
        self._cache_obs_state_dict.clear()
        self._cache_info.clear()
        return self.get_obs()

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #
    def check_success(self) -> bool:
        raise NotImplementedError

    def get_done(self):
        # return self.check_success()
        return False

    def get_info(self):
        return dict(success=self.check_success())

    def step(self, action: Union[None, np.ndarray, Dict]):
        self.step_in_ep += 1
        self.step_action(action)
        obs = self.get_obs()
        reward = self.get_reward()
        done = self.get_done()
        info = self.get_info()
        return obs, reward, done, info

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        self._scene.add_point_light([2, 2, 2], [1, 1, 1])
        self._scene.add_point_light([2, -2, 2], [1, 1, 1])
        self._scene.add_point_light([-2, 0, 2], [1, 1, 1])
        self._scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(1.0, 0.0, 1.2)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _setup_camera(self):
        self._camera = self._scene.add_camera("sideview", 1024, 1024, 1, 0.01, 10)
        self._camera.set_local_pose(
            sapien.Pose([0.6, -1.8, 1.2], euler2quat(0, 0.5, 1.57))
        )

    def render(self, mode="human"):
        if mode == "rgb_array":
            self._scene.update_render()
            self._camera.take_picture()
            rgb = self._camera.get_color_rgba()[..., :3]
            # rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            return rgb
        else:
            return super().render(mode=mode)

    # utilization
    def check_actor_static(self, actor, max_v=None, max_ang_v=None):
        if self.step_in_ep <= 1:
            flag_v = (max_v is None) or (np.linalg.norm(actor.get_velocity()) <= max_v)
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v
            )
        else:
            pose = actor.get_pose()
            t = 1.0 / self._control_freq
            flag_v = (max_v is None) or (
                np.linalg.norm(pose.p - self._prev_actor_pose.p) <= max_v * t
            )
            flag_ang_v = (max_ang_v is None) or (
                angle_distance(self._prev_actor_pose, pose) <= max_ang_v * t
            )
        self._prev_actor_pose = actor.get_pose()
        return flag_v and flag_ang_v

    # -------------------------------------------------------------------------- #
    # Simulation state (required for MPC)
    # -------------------------------------------------------------------------- #
    def get_actors(self):
        return [x for x in self._scene.get_all_actors()]

    def get_articulations(self):
        return [self._agent._robot, self._articulation]

    # def get_sim_state(self) -> np.ndarray:
    #     state = []
    #     # for actor in self._scene.get_all_actors():
    #     for actor in self._actors:
    #         state.append(get_actor_state(actor))
    #     # for articulation in self._scene.get_all_articulations():
    #     for articulation in self._articulations:
    #         state.append(get_articulation_state(articulation))
    #     return np.hstack(state)

    def set_sim_state(self, state: np.ndarray):
        KINEMANTIC_DIM = 13  # [pos, quat, lin_vel, ang_vel]
        start = 0
        for actor in self._actors:
            set_actor_state(actor, state[start : start + KINEMANTIC_DIM])
            start += KINEMANTIC_DIM
        for articulation in self._articulations:
            ndim = KINEMANTIC_DIM + 2 * articulation.dof
            set_articulation_state(articulation, state[start : start + ndim])
            start += ndim
        self.accumulated_contact_direction_err = state[-1]

    def get_sim_state(self) -> np.ndarray:
        """Get simulation state."""
        state = []
        for actor in self._actors:
            state.append(get_actor_state(actor))
        for articulation in self._articulations:
            state.append(get_articulation_state(articulation))
        state.append(self.accumulated_contact_direction_err)
        return np.hstack(state)

    def get_state(self):
        return self.get_sim_state()

    def set_state(self, state: np.ndarray):
        return self.set_sim_state(state)

    # def _get_obs_pointcloud(self):
    #     """get pointcloud from each camera, transform them to the *world* frame, and fuse together"""
    #     self.update_render()

    #     fused_pcd = self._agent.get_fused_pointcloud(actor_seg=self.hide_agent_in_pcd)

    #     pcds = []
    #     for _, camera in self._cameras.items():
    #         camera.take_picture()
    #         pcd = get_camera_pcd(camera, actor_seg=self.hide_agent_in_pcd)
    #         T = camera.get_model_matrix()
    #         pcd["xyz"] = transform_points(T, pcd["xyz"])
    #         pcds.append(pcd)

    #     if len(pcds) > 0:
    #         fused_pcd = merge_dicts([fused_pcd, merge_dicts(pcds, True)], True)
    #     if self.hide_agent_in_pcd:
    #         actor_seg = fused_pcd["actor_seg"][:, None]
    #         diff = actor_seg == self._agent_actor_ids[None, :]
    #         agent_mask = np.logical_or.reduce(diff, axis=1)
    #         for k, v in fused_pcd.items():
    #             fused_pcd[k] = v[np.logical_not(agent_mask)]

    #     return OrderedDict(
    #         pointcloud=fused_pcd,
    #         agent=self._get_proprioception(),
    #         extra=self._get_obs_extra(),
    #     )


# class FixedXmate3RobotiqSensorEnv(FixedXmate3RobotiqEnv):
#     def _load_agent(self):
#         self._agent = FixedXmate3Robotiq.from_config_file(
#             AGENT_CONFIG_DIR / "fixed_xmate3_robotiq_sensors.yml",
#             self._scene,
#             self._control_freq,
#         )
#         self.grasp_site: sapien.Link = get_entity_by_name(
#             self._agent._robot.get_links(), "grasp_convenient_link"
#         )
#         self._agent_actor_ids = []
#         for actor in self._agent._robot.get_links():
#             self._agent_actor_ids.append(actor.id)

#         self._agent_actor_ids = np.array(self._agent_actor_ids).astype(int)


# class FixedXmate3RobotiqSensorLowResEnv(FixedXmate3RobotiqEnv):
#     def _load_agent(self):
#         self._agent = FixedXmate3Robotiq.from_config_file(
#             AGENT_CONFIG_DIR / "fixed_xmate3_robotiq_sensors_low_res.yml",
#             self._scene,
#             self._control_freq,
#         )
#         self.grasp_site: sapien.Link = get_entity_by_name(
#             self._agent._robot.get_links(), "grasp_convenient_link"
#         )
#         self._agent_actor_ids = []
#         for actor in self._agent._robot.get_links():
#             self._agent_actor_ids.append(actor.id)

#         self._agent_actor_ids = np.array(self._agent_actor_ids).astype(int)
