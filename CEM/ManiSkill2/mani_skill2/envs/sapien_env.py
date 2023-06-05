from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.camera import get_camera_images, get_camera_pcd, get_camera_rgb
from mani_skill2.utils.common import (
    convert_observation_to_space,
    flatten_state_dict,
    merge_dicts,
    warn,
)
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.sapien_utils import (
    get_actor_state,
    get_articulation_state,
    set_actor_state,
    set_articulation_state,
)
from mani_skill2.utils.trimesh_utils import (
    get_actor_meshes,
    get_articulation_meshes,
    merge_meshes,
)
from mani_skill2.utils.visualization.misc import observations_to_images, tile_images


class SapienEnv(gym.Env):
    """Superclass for SAPIEN environments.

    Args:
        sim_freq (int): simulation frequency (Hz)
        control_freq (int): control frequency (Hz).

    Notes:
        The environment should be reset before used.

    Example:
        sim_freq = 100, control_freq = 20
        Then _sim_step_per_control = 100 // 20 = 5
        SAPIEN scene physics simulation time step = 0.01 second
        Each action lasts for 0.05 second, which includes 5 physics simulation steps
    """

    _scene: sapien.Scene  # simulation scene instance

    def __init__(self, sim_freq, control_freq):
        self._engine = sapien.Engine()
        # self._engine.set_log_level("debug")
        self._engine.set_log_level("off")
        sapien.VulkanRenderer.set_log_level("off")
        self._renderer = sapien.VulkanRenderer(default_mipmap_levels=1)
        self._engine.set_renderer(self._renderer)
        self._viewer = None

        self._sim_freq = sim_freq
        self._control_freq = control_freq
        if sim_freq % control_freq != 0:
            warn(
                f"sim_freq({sim_freq}) is not divisible by control_freq({control_freq}).",
            )
        self._sim_step_per_control = sim_freq // control_freq

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._clear()

    @property
    def sim_freq(self):
        return self._sim_freq

    @property
    def control_freq(self):
        return self._control_freq

    @property
    def sim_time_step(self):
        return 1.0 / self._sim_freq

    @property
    def control_time_step(self):
        return 1.0 / self._control_freq

    # -------------------------------------------------------------------------- #
    # Utilities
    # -------------------------------------------------------------------------- #
    """Methods below are provided for convenience."""

    _physical_materials: Dict[str, sapien.PhysicalMaterial]
    _render_materials: Dict[str, sapien.RenderMaterial]

    def _setup_scene(self, scene_config: Optional[sapien.SceneConfig] = None):
        """Setup the simulation scene instance.
        The function should be called in reset(), and might be overrided.
        """
        if scene_config is None:
            scene_config = sapien.SceneConfig()
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(1.0 / self._sim_freq)

    def add_physical_material(self, name: str, *args, **kwargs):
        if name in self._physical_materials:
            warn(f"Overwrite physical material {name}")
        self._physical_materials[name] = self._scene.create_physical_material(
            *args, **kwargs
        )

    def add_render_material(self, name: str, **kwargs):
        if name in self._render_materials:
            warn(f"Overwrite render material {name}")
        m = self._renderer.create_material()
        for k, v in kwargs.items():
            if k == "color":
                m.set_base_color(v)
            else:
                getattr(m, f"set_{k}")(v)
        self._render_materials[name] = m

    def _clear(self):
        """Clear the simulation scene instance and other buffers.
        The function can be called in reset() before a new scene is created.
        """
        self._physical_materials = {}
        self._render_materials = {}
        self._scene = None

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    """Due to some known bugs of sapien.utils.Viewer, do not close it after created."""
    _viewer: Viewer

    def _setup_viewer(self):
        """Setup the interactive viewer.
        The function should be called in reset(), and overrided to adjust camera.
        """
        # CAUTION: call `set_scene` after assets are loaded.
        self._viewer.set_scene(self._scene)

    def update_render(self):
        """Update renderer(s). This function should be called before any rendering,
        to sync simulator and renderer."""
        self._scene.update_render()

    def render(self, mode="human"):
        if mode == "human":
            self.update_render()
            if self._viewer is None:
                self._viewer = Viewer(self._renderer)
                self._setup_viewer()
            self._viewer.render()
            return self._viewer
        else:
            raise NotImplementedError(f"Unsupported render mode {mode}.")


class BaseEnv(SapienEnv):
    """Superclass for ManiSkill environments.

    Args:
        obs_mode: observation mode registered in @SUPPORTED_OBS_MODES.
        reward_mode: reward mode registered in @SUPPORTED_REWARD_MODES.
        sim_freq: simulation frequency (Hz).
        control_freq: control frequency (Hz).
    """

    SUPPORTED_OBS_MODES: Tuple[str] = ()
    SUPPORTED_REWARD_MODES: Tuple[str] = ()
    _agent: BaseAgent
    _cameras: Dict[str, sapien.CameraEntity]

    def __init__(self, obs_mode=None, reward_mode=None, sim_freq=500, control_freq=20):
        super().__init__(sim_freq, control_freq)

        if obs_mode is None:
            obs_mode = self.SUPPORTED_OBS_MODES[0]
        if obs_mode not in self.SUPPORTED_OBS_MODES:
            raise NotImplementedError("Unsupported obs mode: {}".format(obs_mode))
        self._obs_mode = obs_mode

        if reward_mode is None:
            reward_mode = self.SUPPORTED_REWARD_MODES[0]
        if reward_mode not in self.SUPPORTED_REWARD_MODES:
            raise NotImplementedError("Unsupported reward mode: {}".format(reward_mode))
        self._reward_mode = reward_mode

        self.seed()

        obs = self.reset(reconfigure=True)
        self.observation_space = convert_observation_to_space(obs)
        self.action_space = self._agent.action_space

    def seed(self, seed=None):
        # For each episode, seed can be passed through `reset(seed=...)`,
        # or generated by `_main_rng`
        if seed is None:
            # Explicitly generate a seed for reproducibility
            seed = np.random.RandomState().randint(2**32)
        self._main_seed = seed
        self._main_rng = np.random.RandomState(self._main_seed)
        return [self._main_seed]

    @property
    def agent(self):
        return self._agent

    @property
    def control_mode(self):
        return self._agent.control_mode

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    @property
    def obs_mode(self):
        return self._obs_mode

    def get_obs(self):
        if self._obs_mode == "state":
            state_dict = self._get_obs_state_dict()
            return flatten_state_dict(state_dict)
        elif self._obs_mode == "state_dict":
            return self._get_obs_state_dict()
        elif self._obs_mode == "rgbd":
            return self._get_obs_rgbd()
        elif self._obs_mode == "depth":
            return self._get_obs_depth()
        elif self._obs_mode == "pointcloud":
            return self._get_obs_pointcloud()
        else:
            raise NotImplementedError(self._obs_mode)

    def _get_obs_state_dict(self) -> OrderedDict:
        """Get (GT) state-based observations."""
        return OrderedDict(
            agent=self._get_proprioception(),
            extra=self._get_obs_extra(),
        )

    def _get_proprioception(self) -> OrderedDict:
        """Get observations from proprioceptive sensors."""
        return self.agent.get_proprioception()

    def _get_obs_extra(self) -> OrderedDict:
        """Get task-relevant extra observations."""
        return OrderedDict()

    def _get_obs_rgbd(self) -> OrderedDict:
        return OrderedDict(
            image=self._get_images(),
            agent=self._get_proprioception(),
            extra=self._get_obs_extra(),
        )

    def _get_obs_depth(self) -> OrderedDict:
        return OrderedDict(
            image=self._get_images(rgb=False),
            agent=self._get_proprioception(),
            extra=self._get_obs_extra(),
        )

    def _get_images(self, rgb=True, depth=True, **kwargs) -> OrderedDict:
        """Get observations from cameras.
        The key is the camera name, and the value is an OrderedDict
        containing camera observations (rgb, depth, ...).
        """
        self.update_render()
        obs_dict = self.agent.get_images(rgb=rgb, depth=depth, **kwargs)
        for name, camera in self._cameras.items():
            obs_dict[name] = self._get_camera_images(
                camera, rgb=rgb, depth=depth, **kwargs
            )
        return obs_dict

    def _get_camera_images(self, camera: sapien.CameraEntity, **kwargs) -> OrderedDict:
        camera.take_picture()
        obs_dict = get_camera_images(camera, **kwargs)
        obs_dict.update(
            camera_intrinsic=camera.get_intrinsic_matrix(),
            camera_extrinsic_world_frame=camera.get_extrinsic_matrix(),
        )
        return obs_dict

    def _get_obs_pointcloud(self):
        """get pointcloud from each camera, transform them to the *world* frame, and fuse together"""
        self.update_render()

        fused_pcd = self._agent.get_fused_pointcloud()

        pcds = []
        for _, camera in self._cameras.items():
            camera.take_picture()
            pcd = get_camera_pcd(camera)
            T = camera.get_model_matrix()
            pcd["xyz"] = transform_points(T, pcd["xyz"])
            pcds.append(pcd)

        if len(pcds) > 0:
            fused_pcd = merge_dicts([fused_pcd, merge_dicts(pcds, True)], True)

        return OrderedDict(
            pointcloud=fused_pcd,
            agent=self._get_proprioception(),
            extra=self._get_obs_extra(),
        )

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
        if self._reward_mode == "sparse":
            return float(self.check_success())
        elif self._reward_mode == "dense":
            return self.compute_dense_reward()
        else:
            raise NotImplementedError(self._reward_mode)

    def compute_dense_reward(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Reconfigure
    # -------------------------------------------------------------------------- #
    def reconfigure(self):
        """Reconfigure the simulation scene instance.
        This function should clear the previous scene, and create a new one.
        """
        self._clear()

        self._setup_scene()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_actors()
        self._load_articulations()
        self._load_agent()
        self._setup_cameras()
        self._setup_lighting()

        if self._viewer is not None:
            self._setup_viewer()

        # Cache actors and articulations
        self._actors = self.get_actors()
        self._articulations = self.get_articulations()
        # Cache initial simulation state
        self._initial_sim_state = self.get_sim_state()

    def _clear(self):
        self._agent = None
        self._cameras = OrderedDict()
        super()._clear()

    def _setup_scene(self):
        scene_config = sapien.SceneConfig()
        scene_config.default_dynamic_friction = 1.0
        scene_config.default_static_friction = 1.0
        scene_config.default_restitution = 0.0
        scene_config.contact_offset = 0.02
        scene_config.solver_iterations = 25
        scene_config.solver_velocity_iterations = 0
        return super()._setup_scene(scene_config)

    def _setup_physical_materials(self):
        pass

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

    def _add_ground(self, altitude=0.0):
        return self._scene.add_ground(
            altitude=altitude, render_material=self._render_materials["ground"]
        )

    def _load_actors(self):
        pass

    def _load_articulations(self):
        pass

    def _load_agent(self):
        pass

    def _setup_cameras(self):
        self._cameras = OrderedDict()

    def _setup_lighting(self):
        pass

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def reset(self, seed=None, reconfigure=False):
        self.set_episode_rng(seed)

        if reconfigure:  # Reconfigure the scene if assets change
            self.reconfigure()
        else:
            self.set_sim_state(self._initial_sim_state)

        self.initialize_episode()

        return self.get_obs()

    def set_episode_rng(self, seed):
        """Set the random generator for current episode."""
        if seed is None:
            self._episode_seed = self._main_rng.randint(2**32)
        else:
            self._episode_seed = seed
        self._episode_rng = np.random.RandomState(self._episode_seed)

    def initialize_episode(self):
        """Initialize the episode, e.g., poses of actors and articulations,
        and robot configuration. No new assets are created."""
        self._initialize_actors()
        self._initialize_articulations()
        self._initialize_agent()

    def _initialize_actors(self):
        pass

    def _initialize_articulations(self):
        pass

    def _initialize_agent(self):
        pass

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #
    def step(self, action: Union[None, np.ndarray, Dict]):
        self.step_action(action)
        obs = self.get_obs()
        reward = self.get_reward()
        done = self.get_done()
        info = self.get_info()
        return obs, reward, done, info

    def step_action(self, action):
        if action is None:  # simulation without action
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

    def check_success(self) -> bool:
        raise NotImplementedError

    def get_done(self):
        # return self.check_success()
        return False

    def get_info(self):
        return dict(success=self.check_success())

    # -------------------------------------------------------------------------- #
    # Simulation state (required for MPC)
    # -------------------------------------------------------------------------- #
    def get_actors(self):
        return self._scene.get_all_actors()

    def get_articulations(self):
        # NOTE(jigu): There are dummy articulations in controllers.
        # return self._scene.get_all_articulations()
        return [self._agent._robot]

    def get_sim_state(self) -> np.ndarray:
        """Get simulation state."""
        state = []
        for actor in self._actors:
            state.append(get_actor_state(actor))
        for articulation in self._articulations:
            state.append(get_articulation_state(articulation))
        return np.hstack(state)

    def set_sim_state(self, state: np.ndarray):
        """Set simulation state."""
        KINEMANTIC_DIM = 13  # [pos, quat, lin_vel, ang_vel]
        start = 0
        for actor in self._actors:
            set_actor_state(actor, state[start : start + KINEMANTIC_DIM])
            start += KINEMANTIC_DIM
        for articulation in self._articulations:
            ndim = KINEMANTIC_DIM + 2 * articulation.dof
            set_articulation_state(articulation, state[start : start + ndim])
            start += ndim

    def get_state(self):
        """Get environment state. Override to include task information (e.g., goal)"""
        return self.get_sim_state()

    def set_state(self, state: np.ndarray):
        """Set environment state. Override to include task information (e.g., goal)"""
        return self.set_sim_state(state)

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    render_camera: sapien.CameraEntity  # should be created in `_setup_cameras`

    def render(self, mode="human"):
        if mode == "rgb_array":
            self.update_render()
            self.render_camera.take_picture()
            return get_camera_rgb(self.render_camera)
        elif mode == "cameras":
            images = [self.render("rgb_array")]
            self.update_render()
            for obs_dict in self._get_images().values():
                images.extend(observations_to_images(obs_dict))
            return tile_images(images)
        else:
            return super().render(mode=mode)

    def gen_scene_pcd(self, num_points: int = int(1e5)) -> np.ndarray:
        """generate scene point cloud for motion planning, excluding the robot"""
        meshes = []
        articulations = self._scene.get_all_articulations()
        if self._agent is not None:
            articulations.pop(articulations.index(self._agent._robot))
        for articulation in articulations:
            articulation_mesh = merge_meshes(get_articulation_meshes(articulation))
            if articulation_mesh:
                meshes.append(articulation_mesh)

        for actor in self._scene.get_all_actors():
            actor_mesh = merge_meshes(get_actor_meshes(actor))
            if actor_mesh:
                meshes.append(
                    actor_mesh.apply_transform(
                        actor.get_pose().to_transformation_matrix()
                    )
                )

        scene_mesh = merge_meshes(meshes)
        scene_pcd = scene_mesh.sample(num_points)
        return scene_pcd
