from collections import OrderedDict

import numpy as np
from gym import ActionWrapper, ObservationWrapper, spaces

from mani_skill2.utils.o3d_utils import pcd_voxel_down_sample_with_crop

from .common import clip_and_scale_action, normalize_action, normalize_action_space


class NormalizeActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = normalize_action_space(self.env.action_space)

    def action(self, action):
        return clip_and_scale_action(
            action, self.env.action_space.low, self.env.action_space.high
        )

    def reverse_action(self, action):
        return normalize_action(
            action, self.env.action_space.low, self.env.action_space.high
        )


class NormalizeActionDictWrapper(ActionWrapper):
    # Not completed
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.env.action_space, spaces.Dict), self.env.action_space
        new_action_space = OrderedDict()
        for action_name, action_space in self.env.action_space.spaces.items():
            if isinstance(action_space, spaces.Box):
                action_space = normalize_action_space(action_space)
            new_action_space[action_name] = action_space
        self.action_space = spaces.Dict(new_action_space)


class ManiSkillActionWrapper(ActionWrapper):
    def __init__(self, env, control_mode=None):
        super().__init__(env)
        if control_mode is None:
            control_mode = self.env.unwrapped.agent.control_mode
        else:
            self.env.unwrapped.agent.set_control_mode(control_mode)
        self._control_mode = control_mode
        self.env.action_space=self.env.unwrapped.agent.action_space
        # self.action_space = self.env.action_space[self._control_mode]
        self.action_space = self.env.action_space  # mlq

    def reset(self):
        ret = super().reset()
        self.env.unwrapped.agent.set_control_mode(self._control_mode)
        return ret

    def action(self, action):
        return {"control_mode": self._control_mode, "action": action}

    @property
    def control_mode(self):
        return self._control_mode


class PointCloudPreprocessObsWrapper(ObservationWrapper):
    """Preprocess point cloud, crop and voxel downsample"""

    _num_pts: int
    _vox_size: float

    def __init__(
        self,
        env,
        num_pts=2048,
        vox_size=0.003,
        min_bound=np.array([-1, -1, 1e-3]),
        max_bound=np.array([1, 1, 1]),
    ):
        super().__init__(env)
        self._num_pts = num_pts
        self._vox_size = vox_size
        self._min_bound = min_bound
        self._max_bound = max_bound

    def observation(self, observation):
        if not self.obs_mode == "pointcloud":
            return observation

        obs = observation
        pointcloud = obs["pointcloud"]
        xyz = pointcloud["xyz"]
        sample_indices = pcd_voxel_down_sample_with_crop(
            xyz, self._vox_size, self._min_bound, self._max_bound
        )
        if len(sample_indices) >= self._num_pts:
            sample_indices = np.random.choice(
                sample_indices, self._num_pts, replace=False
            )
        else:
            ext_sample_indices = np.random.choice(
                sample_indices, (self._num_pts - len(sample_indices))
            )
            sample_indices.extend(ext_sample_indices)
        for k, v in pointcloud.items():
            pointcloud[k] = v[sample_indices]
        obs["pointcloud"] = pointcloud
        return obs
