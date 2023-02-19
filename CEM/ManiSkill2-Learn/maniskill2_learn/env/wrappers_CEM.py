from gym import spaces
from gym.spaces import Discrete
from gym.core import Wrapper, ObservationWrapper, ActionWrapper

from collections import deque
import numpy as np
import cv2
from typing import Dict, List

from pyrl.utils.meta import Registry, build_from_cfg
from pyrl.utils.data import deepcopy, DictArray, encode_np, to_array, GDict, is_num
from .observation_process import (
    process_mani_skill_base,
    process_mani_skill_target_object_only,
    process_mani_skill_uniform_downsample,
    process_mani_skill_voxel_downsample,
)

from mani_skill2.agents.fixed_xmate3_robotiq import FixedXmate3Robotiq
# from mani_skill2.agents.panda import Panda
from mani_skill2.utils.common import flatten_state_dict, convert_np_bool_to_float

WRAPPERS = Registry("wrapper of env")


class TrivialRolloutWrapper(Wrapper):
    """
    The same api as VectorEnv
    """

    def __init__(self, env, use_cost=False, reward_scale=1, extra_dim=False):
        super(TrivialRolloutWrapper, self).__init__(env)
        self.recent_obs = None
        self.iscost = -1 if use_cost else 1
        self.num_envs = 1
        self.reward_scale = reward_scale
        self.is_discrete = isinstance(env.action_space, Discrete)
        self.extra_dim = extra_dim
        self.episode_dones = False
        self.ready_idx = []
        self.running_idx = []
        self.idle_idx = 0

    @property
    def done_idx(self):
        if self.episode_dones:
            return [
                0,
            ]
        else:
            return []

    def act_process(self, action):
        if self.is_discrete:
            if is_num(action):
                action = int(action)
            else:
                action = action.reshape(-1)
                assert len(action) == 1, f"Dim of discrete action should be 1, but we get {len(action)}"
                action = int(action[0])
        return action

    # For original gym interface.
    def random_action(self):
        action = to_array(self.action_space.sample())
        if self.extra_dim:
            action = action[None]
        return action

    def reset(self, *args, idx=None, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        obs = GDict(obs).f64_to_f32(False)
        if self.extra_dim:
            obs = GDict(obs).unsqueeze(0, False)
        obs = deepcopy(obs)
        self.recent_obs = obs
        return obs

    # def step(self, *args, idx=None, sync=True, **kwargs):
    #     next_obs, rewards, episode_dones, infos = self.env.step(*args, **kwargs)
    #     rewards = rewards * self.iscost
    #     return next_obs, rewards, episode_dones, infos

    def reset_dict(self, idx=None, **kwargs):
        return {"obs": self.reset(**kwargs)}

    def render_dict(self, idx=None, **kwargs):
        return {"images": self.render(**kwargs)}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        else:
            return getattr(self.env, name)

    def step_dict(self, action, with_info=False, encode_info=True, **kwargs):
        """
        Change the output of step to a dict format !!
        """
        from .env_utils import true_done

        obs = self.recent_obs
        if self.extra_dim:
            obs = GDict(obs).take(0, 0, False)
            if action.ndim > 1:
                action = action[0]
        next_obs, rewards, episode_dones, infos = self.env.step(self.act_process(action))
        next_obs = GDict(next_obs).f64_to_f32(False)
        next_obs = deepcopy(next_obs)
        # episode_dones = episode_dones
        # rewards = deepcopy(rewards)
        self.episode_dones = episode_dones
        infos = deepcopy(infos)

        dones = true_done(episode_dones, infos)
        dones = np.array([dones], dtype=np.bool)
        rewards = np.array([rewards * self.reward_scale], dtype=np.float32)
        episode_dones = np.array([episode_dones], dtype=np.bool)
        ret = {
            "obs": obs,
            "actions": action,
            "next_obs": next_obs,
            "rewards": rewards,
            "dones": dones,
            "episode_dones": episode_dones,
        }
        if with_info:
            if "TimeLimit.truncated" not in infos:
                infos["TimeLimit.truncated"] = False
            ret["infos"] = np.array([encode_np(infos)], dtype=np.dtype("U10000")) if encode_info else infos  # info with at most 10KB
        if self.extra_dim:
            ret = GDict(ret).unsqueeze(0, False)
        # print(GDict(ret).shape)

        self.recent_obs = ret["next_obs"]
        return ret

    def step(self, action, with_info=True, to_list=True, encode_info=False, **kwargs):
        infos = self.step_dict(action, with_info, encode_info)
        return (infos["next_obs"], infos["rewards"][0], infos["episode_dones"][0], infos.get("infos", {})) if to_list else infos

    # The following three functions are availiable for VectorEnv too!
    def step_random_actions(self, num, with_info=False, encode_info=True):
        ret = []
        for i in range(num):
            obs = self.recent_obs
            item = self.step_dict(self.random_action(), encode_info=encode_info, with_info=with_info)
            item["obs"] = obs
            ret.append(item)
            if item["episode_dones"]:
                self.reset()
        if self.extra_dim:
            ret = DictArray.concat(ret, axis=0, wrapper=False)
        else:
            ret = DictArray.stack(ret, axis=0, wrapper=False)
        return ret

    def step_states_actions(self, states=None, actions=None):
        # for CEM only
        # states: length N
        # actions: [N, M, NA]
        assert actions.ndim == 3
        rewards = np.zeros_like(actions[..., :1], dtype=np.float32)
        for i in range(len(actions)):
            if hasattr(self, "set_state") and states is not None:
                self.set_state(states[i])
            for j in range(len(actions[i])):
                # print(self.env.step(actions[i, j])[1].shape, rewards[i, j].shape, actions[i, j].shape)
                rewards[i, j] = self.env.step(actions[i, j])[1] * self.iscost
        return rewards

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps


class MujocoWrapper(Wrapper):
    def get_state(self):
        if hasattr(self.env, "goal"):
            return np.concatenate([self.env.sim.get_state().flatten(), self.env.goal], axis=-1)
        else:
            return self.env.sim.get_state().flatten()

    def set_state(self, state):
        if hasattr(self.env, "goal"):
            sim_state_len = self.env.sim.get_state().flatten().shape[0]
            self.env.sim.set_state(self.env.sim.get_state().from_flattened(state[:sim_state_len], self.env.sim))
            self.env.goal = state[sim_state_len:]
        else:
            self.env.sim.set_state(self.env.sim.get_state().from_flattened(state, self.env.sim))

    def get_obs(self):
        return self.env.unwrapped._get_obs()


class PendulumWrapper(Wrapper):
    def get_state(self):
        return np.array(self.env.state)

    def set_state(self, state):
        self.env.state = deepcopy(state)

    def get_obs(self):
        return self.env.unwrapped._get_obs()


@WRAPPERS.register_module()
class ManiSkillOSCWrapper(ActionWrapper):
    def __init__(self, env, env_name):
        super(ManiSkillOSCWrapper, self).__init__(env)
        from mani_skill.utils.osc import OperationalSpaceControlInterface

        # print("Use operational space control")
        self.osc_interface = OperationalSpaceControlInterface(env_name)
        self._total_dim = self.osc_interface.right_arm_dim + self.osc_interface.left_arm_dim + self.osc_interface.osc_dim
        variance = 10
        self.action_space = spaces.Box(low=-variance, high=variance, shape=(self._total_dim,))

    def step(self, action):
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        return self.env.step(self.action(action))

    def action(self, action):
        qpos = self.env.agent.robot.get_qpos()
        assert len(action) == self._total_dim
        operational_space_action = action[: self.osc_interface.osc_dim]
        null_space_action = action[self.osc_interface.osc_dim :]
        q_action = self.osc_interface.operational_space_and_null_space_to_joint_space(qpos, operational_space_action, null_space_action)
        return q_action

    def reverse_action(self, action):
        qpos = self.env.agent.robot.get_qpos()
        osc_action = np.concatenate(self.osc_interface.joint_space_to_operational_space_and_null_space(qpos, action))
        return osc_action


@WRAPPERS.register_module()
class FixedInitWrapper(Wrapper):
    def __init__(self, env, init_state, level=None, *args, **kwargs):
        super(FixedInitWrapper, self).__init__(env)
        self.init_state = np.array(init_state)
        self.level = level

    def reset(self, *args, **kwargs):
        if self.level is not None:
            # For ManiSkill
            self.env.reset(level=self.level)
        else:
            self.env.reset()
        self.set_state(self.init_state)
        return self.env.get_obs()


class ManiSkillObsWrapper(ObservationWrapper):
    def __init__(self, env, stack_frame=1, remove_visual=False, process_mode="base", add_prev_action=False):
        """
        Stack k last frames for point clouds or rgbd and remap the rendering configs
        """
        super(ManiSkillObsWrapper, self).__init__(env)
        self.stack_frame = stack_frame
        self.remove_visual = remove_visual
        self.buffered_data = {}
        self.process_mode = process_mode

        self.PCD_OBS_MODE = ["pointcloud", "full_pcd", "no_robot", "target_object_only", "handle_only", "fused_pcd", "pointcloud_3d_ann"]

        if process_mode in ["target_object_only", "fuse_with_first_frame"]:
            assert self.obs_mode == "pointcloud"
        if self.obs_mode == "fused_pcd":
            self.process_mode = "uniform_downsample"
        self.first_pcd = None

        self.add_prev_action = add_prev_action
        self.prev_action = np.zeros(len(self.env.action_space.sample()))

    def get_state(self):
        return self.env.get_state(True)

    def _update_buffer(self, obs):
        for key in obs:
            if key not in self.buffered_data:
                self.buffered_data[key] = deque([obs[key]] * self.stack_frame, maxlen=self.stack_frame)
            else:
                self.buffered_data[key].append(obs[key])

    def _get_buffer_content(self):
        axis = 0 if self.obs_mode in self.PCD_OBS_MODE else -1
        return {key: np.concatenate(self.buffered_data[key], axis=axis) for key in self.buffered_data}

    def observation(self, observation):
        if self.obs_mode == "state":
            if self.add_prev_action:
                return np.concatenate([observation, self.prev_action])
            else:
                return np.concatenate([observation])

        # print(observation[self.obs_mode]['xyz'].min(), observation[self.obs_mode]['xyz'].max())

        if self.process_mode == "base":
            observation = process_mani_skill_base(observation, self.env)
        elif self.process_mode == "base_wrong":
            observation = process_mani_skill_base(observation, self.env, wrong=True)
        elif self.process_mode == "uniform_downsample":
            observation = process_mani_skill_uniform_downsample(observation, self.env)
        elif self.process_mode == "voxel_downsample":
            observation = process_mani_skill_voxel_downsample(observation, self.env)
        elif self.process_mode == "target_object_only":
            observation = process_mani_skill_target_object_only(observation, self.env)
        elif self.process_mode is not None:
            print(self.process_mode)
            raise ValueError

        visual_data = observation[self.obs_mode]

        # print(observation[self.obs_mode]['xyz'].min(), observation[self.obs_mode]['xyz'].max())
        # exit(0)

        self._update_buffer(visual_data)
        visual_data = self._get_buffer_content()
        if self.remove_visual:
            for key in visual_data:
                visual_data[key] = visual_data[key] * 0
        state = observation["state"] if "state" in observation else observation["agent"]
        if "target_info" in observation:
            target_info = observation.pop("target_info")
            state = np.concatenate([target_info, state])
        if self.add_prev_action:
            state = np.concatenate([state, self.prev_action])

        # Convert dict of array to list of array with sorted key
        ret = {}
        ret[self.obs_mode] = visual_data
        ret["state"] = state
        for key in observation:
            if key not in [self.obs_mode, "state", "agent"]:
                ret[key] = observation[key]
        return ret

    def step(self, action):
        self.prev_action[:] = action
        next_obs, reward, done, info = super(ManiSkillObsWrapper, self).step(action)
        return next_obs, reward, done, info

    def reset(self, level=None):
        # Change the ManiSkill level to seed as the standard interface in gym
        self.buffered_data = {}
        self.prev_action *= 0.0
        return self.observation(self.env.reset() if level is None else self.env.reset(level=level))

    def get_obs(self):
        return self.observation(self.env.get_obs())

    def set_state(self, *args, **kwargs):
        return self.observation(self.env.set_state(*args, **kwargs))

    def render(self, mode="human", *args, **kwargs):
        if mode == "human":
            self.env.render(mode, *args, **kwargs)
            return

        if mode in ["rgb_array", "color_image"]:
            img = self.env.render(mode="color_image", *args, **kwargs)
        else:
            img = self.env.render(mode=mode, *args, **kwargs)
        if isinstance(img, dict):
            if "world" in img:
                img = img["world"]
            elif "main" in img:
                img = img["main"]
            else:
                print(img.keys())
                exit(0)
        if isinstance(img, dict):
            img = img["rgb"]
        if img.ndim == 4:
            assert img.shape[0] == 1
            img = img[0]
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, a_min=0, a_max=1) * 255
        img = img[..., :3]
        img = img.astype(np.uint8)
        return img


@WRAPPERS.register_module()
class ManiSkill2_ObsWrapper(ObservationWrapper):
    def __init__(self, env, img_size=None, n_points=None, process_mode=None, process_pc_mode=None):
        super().__init__(env)
        if self.obs_mode in ["state", "state_dict"]:
            pass
        elif self.obs_mode == "rgbd":
            obs_space = env.observation_space
            h, w, _ = obs_space['hand_camera']['rgb'].shape
            new_shape = (h, w, 4*2) # assume there are 2 cameras
            low = np.full(new_shape, -float("inf"))
            high = np.full(new_shape, float("inf"))
            self.observation_space = spaces.Box(low, high, dtype=obs_space['hand_camera']['rgb'].dtype)

            self.img_size = img_size
        elif self.obs_mode == "pointcloud":
            self.process_mode = process_mode
            self.process_pc_mode = process_pc_mode
            self.n_points = n_points
        else:
            raise NotImplementedError()

    def observation(self, observation):
        from mani_skill2.utils.common import flatten_state_dict
        if self.obs_mode == "state":
            # key_list = ['qpos', 'qvel', 'cube_pose', 'gripper_site_pose']
            # return concat_vec_in_dict(observation, key_list)
            return observation
        elif self.obs_mode == "state_dict":
            obs = observation
            obs["extra"].pop("task", None)
            obs["extra"].pop("articulation", None)
            return flatten_state_dict(obs)
        elif self.obs_mode == 'rgbd':
            obs = observation
            rgb = np.concatenate([
                obs['hand_camera']['rgb'], obs['third_view']['rgb'],
            ], axis=2)
            depth = np.concatenate([
                obs['hand_camera']['depth'], obs['third_view']['depth'],
            ], axis=2)
            # s = np.concatenate([
            #     concat_vec_in_dict(obs['agent_state'], ['qpos', 'qvel']),
            #     obs['gripper_pose'],
            # ], axis=0)

            obs.pop('hand_camera', None)
            obs.pop('third_view', None)
            # obs['agent_state']['joint_external_torque'] = (np.abs(obs['agent_state']['joint_external_torque']) > 0.1 ) * 1.0
            s = flatten_state_dict(obs)

            if self.img_size is not None and self.img_size != (rgb.shape[0], rgb.shape[1]):
                rgb = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_LINEAR)

            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).astype(np.uint8)
            depth = depth.astype(np.float16)

            out_dict = {
                'rgb': rgb,
                'depth': depth,
                'state': s,
            }
            return out_dict
        elif self.obs_mode == 'depth':
            obs = observation
            if isinstance(self.env.agent, Panda):
                depth = np.concatenate([
                    obs['image']['hand_camera']['depth'], obs['image']['third_view_camera']['depth'],
                ], axis=2)
            elif isinstance(self.env.agent, FixedXmate3Robotiq):
                depth = np.concatenate([
                    obs['image']['hand_camera']['depth'], obs['image']['base_camera']['depth'],
                ], axis=2)

            obs.pop('image')
            # obs.pop('extra')  # the extra information should be inferred from RGBD observation
            s = flatten_state_dict(obs)

            if self.img_size != (depth.shape[0], depth.shape[1]):
                # Note: the order of dimension is permuted in cv2.resize()
                # rgb = cv2.resize(rgb, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

            # if rgb.dtype != np.uint8:
            #     rgb = (rgb * 255).astype(np.uint8)
            depth = depth.astype(np.float16)

            out_dict = {
                'depth': depth,
                'state': s,
            }
            return out_dict
        elif self.obs_mode == 'pointcloud':
            from pyrl.env.obs_process import pcd_base, pcd_uniform_downsample, pcd_voxel_downsample, pcd_target_object_only
            from mani_skill2.utils.contrib import apply_pose_to_points
            if self.process_pc_mode == "ms2_cam":
                # pcd = obs.pop("pointcloud")
                # resampled_pcd = pcd_ms2_base(pcd, self.env, self.n_points)
                observation = pcd_base(observation, self.env)
                # observation = pcd_uniform_downsample(observation, self.env)
                # observation = pcd_voxel_downsample(observation, self.env)
                # observation = pcd_target_object_only(observation, self.env)
                resampled_pcd = observation.pop('pointcloud')
                resampled_pcd['xyz'] = apply_pose_to_points(resampled_pcd['xyz'],
                                                        self.env.unwrapped.grasp_site.get_pose().inv())  ## gripper frame
                # resampled_pcd['xyz'] = apply_pose_to_points(resampled_pcd['xyz'],
                #                                             self.env.unwrapped._target_link.get_pose().inv())  ## target_link frame
            elif self.process_pc_mode == "ms2_gt":
                resampled_pcd = dict(observation.pop('pointcloud'))
                resampled_pcd.pop('rgb')
                # resampled_pcd.pop('seg')
            elif self.process_pc_mode == "ms2_real":
                resampled_pcd = dict(observation.pop('pointcloud'))
            elif self.process_pc_mode == "ms2_realfull":
                resampled_pcd = dict(observation.pop('pointcloud'))
            else:
                raise NotImplementedError

            # visual_state = flatten_state_dict(observation.pop('visual_state', None))
            # tcp_wrench_state = observation.pop("tcp_wrench", None)
            state = flatten_state_dict(observation)
            out_dict = {
                'pointcloud': resampled_pcd,
                # 'visual_state': visual_state,
                'state': state,
            }
            return out_dict

        else:
            raise NotImplementedError()

    @property
    def _max_episode_steps(self):
        return self.env.unwrapped._max_episode_steps

    def render(self, mode='human', *args, **kwargs):
        if mode == 'human':
            self.env.render(mode, *args, **kwargs)
            return

        if mode in ['rgb_array', 'color_image']:
            img = self.env.render(mode='rgb_array', *args, **kwargs)
        else:
            img = self.env.render(mode=mode, *args, **kwargs)
        if isinstance(img, dict):
            if 'world' in img:
                img = img['world']
            elif 'main' in img:
                img = img['main']
            else:
                print(img.keys())
                exit(0)
        if isinstance(img, dict):
            img = img['rgb']
        if img.ndim == 4:
            assert img.shape[0] == 1
            img = img[0]
        if img.dtype in [np.float32, np.float64]:
           img = np.clip(img, a_min=0, a_max=1) * 255
        img = img[..., :3]
        img = img.astype(np.uint8)
        return img


@WRAPPERS.register_module()
class ManiSkill2_VisualAugWrapper(ObservationWrapper):
    """ Data augmentation wrapper"""

    def __init__(self, env, visual_aug=False, view_independent=True, color_bright=0., color_contrast=0., color_saturation=0., color_hue=0.,
                 color_drop_prob=0., depth_noise_scale=0., depth_salt=0., pcd_noise_scale=0., pcd_drop_prob=0.):
        super().__init__(env)
        import torchvision
        self.visual_aug = visual_aug
        self.view_independent = view_independent  # whether augment the rgb data from different views separately
        self.color_aug = (color_bright > 0) or (color_contrast > 0) or (color_saturation > 0) or (color_hue > 0) or (
                color_drop_prob > 0)
        if self.color_aug:
            self.color_jitter = torchvision.transforms.ColorJitter(
                brightness=color_bright,
                contrast=color_contrast,
                saturation=color_saturation,
                hue=color_hue,
            )
            self.color_drop = torchvision.transforms.Grayscale(3)
            self.color_drop_prob = color_drop_prob
        self.depth_aug = (depth_noise_scale > 0) or (depth_salt > 0)
        self.depth_noise_scale = depth_noise_scale
        self.depth_salt = depth_salt

        self.pcd_aug = (pcd_noise_scale > 0) or (pcd_drop_prob > 0)
        self.pcd_noise_scale = pcd_noise_scale
        self.pcd_drop_prob = pcd_drop_prob

    def observation(self, observation):
        if self.obs_mode == "state":
            return observation
        if not self.visual_aug:
            return observation
        if self.obs_mode == "rgbd":
            from PIL import Image
            if self.color_aug:
                rgb = observation['rgb']
                num_views = rgb.shape[-1] // 3
                if self.view_independent:
                    rgbs = []
                    for view_id in range(num_views):
                        view_rgb = Image.fromarray(rgb[..., 3 * view_id:3 * view_id + 3])
                        if np.random.rand() < self.color_drop_prob:
                            view_rgb = self.color_drop(view_rgb)
                        else:
                            view_rgb = self.color_jitter(view_rgb)
                        rgbs.append(np.array(view_rgb))
                    rgb = np.concatenate(rgbs, axis=-1).astype(np.uint8)
                else:
                    h, w, c = rgb.shape
                    rgb = rgb.reshape(h, -1, 3)
                    rgb = Image.fromarray(rgb)
                    if np.random.rand() < self.color_drop_prob:
                        rgb = self.color_drop(rgb)
                    else:
                        rgb = self.color_jitter(rgb)
                    rgb = np.array(rgb).reshape(h, w, c).astype(np.uint8)
            else:
                rgb = observation['rgb']

            depth = observation['depth']
            if self.depth_aug:
                depth_noise = np.clip(np.random.randn(*depth.shape), -2, 2).astype(np.float16)
                depth_noise *= (depth ** 2)
                depth_salt = (np.random.rand(*depth.shape) > self.depth_salt).astype(np.float16)

                depth += depth_noise
                depth *= depth_salt

            out_dict = {
                'rgb': rgb,
                'depth': depth,
                'state': observation['state']
            }
            return out_dict
        elif self.obs_mode == "depth":
            depth = observation['depth']
            if self.depth_aug:
                depth_noise = np.clip(np.random.randn(*depth.shape), -2, 2).astype(np.float16)
                depth_noise *= (depth ** 2)
                depth_salt = (np.random.rand(*depth.shape) > self.depth_salt).astype(np.float16)

                depth += depth_noise
                depth *= depth_salt

            out_dict = {
                'depth': depth,
                'state': observation['state']
            }
            return out_dict
        elif self.obs_mode == "pointcloud":
            if "rgb" in observation["pointcloud"] and self.color_aug:
                from PIL import Image
                rgb = observation["pointcloud"]["rgb"].reshape(-1, 1, 3)
                rgb = Image.fromarray((rgb*255).astype(np.uint8))
                if np.random.rand() < self.color_drop_prob:
                    rgb = self.color_drop(rgb)
                else:
                    rgb = self.color_jitter(rgb)
                rgb = np.array(rgb).astype(np.float) / 255.
                observation["pointcloud"]["rgb"] = rgb

            if self.pcd_aug:
                xyz = observation["pointcloud"]["xyz"]
                if self.pcd_drop_prob > 0:
                    num_pts = xyz.shape[0]
                    num_chosen = int(num_pts*(1-self.pcd_drop_prob))
                    chosen_idx = np.random.choice(np.arange(num_pts), num_chosen, replace=False)
                    xyz[:num_chosen] = xyz[chosen_idx]
                    xyz[num_chosen:] = xyz[0]
                xyz_noise = np.clip(np.random.randn(*xyz.shape), -2, 2).astype(np.float16)
                xyz += xyz_noise
                observation["pointcloud"]["xyz"] = xyz

            return observation


class DSI_Wrapper(Wrapper):
    def __init__(self, env, demo_path, dsi_ratio=0.5):
        super().__init__(env)
        from pyrl.utils.file import load_hdf5
        raw_data = load_hdf5(demo_path)
        self.trajs = [
            raw_data[idx]['env_states'] for idx in raw_data
        ]
        self.dsi_ratio = dsi_ratio

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self.env.unwrapped._episode_rng.uniform() > self.dsi_ratio:
            return obs

        # Demo State Initialization
        i = self.env.unwrapped._episode_rng.randint(0, len(self.trajs))
        # l = len(self.trajs[i])
        l = 20
        j = self.env.unwrapped._episode_rng.randint(0, l)
        # print('DSI:', i, j)
        sampled_init_state = self.trajs[i][j]
        self.env.set_state(sampled_init_state)

        return self.env.get_obs()


def put_texts_on_image(image: np.ndarray, lines: List[str]):
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()

    font_size = 1.0
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (255, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image

def put_infos_on_image(image, info: Dict[str, np.ndarray], overlay=True):
    lines = [f"{k}: {v.round(3)}" for k, v in info.items()]
    return put_texts_on_image(image, lines)


class RenderInfoWrapper(Wrapper):

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if "TimeLimit.truncated" in info.keys():
            info["TimeLimit.truncated"] = convert_np_bool_to_float(info["TimeLimit.truncated"])
        self._info_for_render = info
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        # self._info_for_render = self.env.get_info()
        self._info_for_render = {}
        return obs

    def render(self, mode, **kwargs):
        if mode == 'rgb_array' or mode == 'cameras':
            img = super().render(mode=mode, **kwargs)
            img = (img * 255).astype(np.uint8)
            print(self._info_for_render)
            exit(0)
            return put_infos_on_image(img, self._info_for_render, overlay=True)
        else:
            return super().render(mode=mode, **kwargs)


def build_wrapper(cfg, default_args=None):
    return build_from_cfg(cfg, WRAPPERS, default_args)
