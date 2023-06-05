import os

import numpy as np
import sapien
from sapien.utils import Viewer

from mani_skill2 import AGENT_CONFIG_DIR
from mani_skill2.agents.fixed_xmate3_robotiq import FixedXmate3Robotiq
from mani_skill2.utils.sapien_utils import get_entity_by_name

_sim_freq = 240
_control_freq = 20

scene_config = sapien.core.SceneConfig()
_engine = sapien.core.Engine()
_scene = _engine.create_scene(scene_config)
_scene.set_timestep(1.0 / _sim_freq)


_agent = FixedXmate3Robotiq.from_config_file(
        os.path.join(AGENT_CONFIG_DIR, "fixed_xmate3_robotiq.yml"),
        _scene,
        _control_freq
)
grasp_site: sapien.core.Link = get_entity_by_name(
        _agent._robot.get_links(), "grasp_convenient_link"
)
_agent_actor_ids = []
for actor in _agent._robot.get_links():
    _agent_actor_ids.append(actor.id)
_agent_actor_ids = np.array(_agent_actor_ids).astype(int)

_renderer = sapien.core.SapienRenderer()
_engine.set_renderer(_renderer)
_scene.set_ambient_light([0.5, 0.5, 0.5])
_scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
_viewer = Viewer(_renderer)
_viewer.set_scene(_scene)
_viewer.set_camera_xyz(x=-2, y=0, z=1)
_viewer.set_camera_rpy(r=0, p=-0.3, y=0)

while not _viewer.closed:
    _scene.step()
    _scene.update_render()
    _viewer.render()
