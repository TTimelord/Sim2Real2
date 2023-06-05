import copy
import os
from collections import OrderedDict
from typing import Dict

import numpy as np
import sapien.core as sapien

from sapien.core import Pose

from mani_skill2 import AGENT_CONFIG_DIR, DIGITAL_TWIN_CONFIG_DIR
from mani_skill2.agents.fixed_xmate3_robotiq import FixedXmate3Robotiq
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

engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)

digital_twin_config_path = os.path.join(DIGITAL_TWIN_CONFIG_DIR, 'faucet_video_2.yaml')

env = FixedXmate3RobotiqEnv(
        articulation_config_path=digital_twin_config_path,
        # obs_mode="state_dict",
        # reward_mode="dense",
        sim_freq=240,
        control_freq=20
)
viewer = env.render()

while True:
    env.render()
    env.step(None)
