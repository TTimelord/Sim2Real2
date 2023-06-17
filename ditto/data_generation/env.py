"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig
from sapien.utils import Viewer
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from utils import process_angle_limit, get_random_number
import random

class ContactError(Exception):
    pass


class Env(object):
    
    def __init__(self, show_gui=True, render_rate=20, timestep=1/500, \
            object_position_offset=0.0, succ_ratio=0.1, zero_gravity=False):
        self.current_step = 0

        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        
        # render_config = OptifuserConfig()
        # render_config.shadow_map_size = 8192
        # render_config.shadow_frustum_size = 10
        # render_config.use_shadow = False
        # render_config.use_ao = True
        
        self.renderer = sapien.VulkanRenderer()
        # self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # set log level
        # self.engine.set_log_level("off")
        # sapien.VulkanRenderer.set_log_level("off")

        # GUI
        self.window = False
        if show_gui:
            self.viewer = Viewer(self.renderer)

        # scene
        scene_config = SceneConfig()
        if not zero_gravity:
            scene_config.gravity = [0, 0, -9.81]
        else:
            scene_config.gravity = [0, 0, 0]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0

        self.scene = self.engine.create_scene(config=scene_config)
        if show_gui:
            self.viewer.set_scene(self.scene)
            # self.viewer.set_camera_xyz(-3.0+object_position_offset, 0.0, 3.0)
            # self.viewer.set_camera_rpy(0.0, np.pi, np.pi/10)
            self.viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        # self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5]) # mlq: there is no shadow light 
        self.scene.add_point_light([1+object_position_offset, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1+object_position_offset, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1+object_position_offset, 0, 1], [1, 1, 1])

        # default Nones
        self.object = None
        self.object_target_joint = None

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.viewer.set_camera_xyz(x, y, z)
        self.viewer.set_camera_rpy(0.0, pitch, yaw)
        self.viewer.render()
    def take_rgb_picture(self, path):
        res = self.scene.get_cameras()
        print(len(res), " camera(s).")
        for cam in res:
            print(cam.pose.p, cam.pose.q)

    def load_object(self, urdf, material, scale=1.0, q=None):
        if not q:
            q = [1, 0, 0, 0]
        loader = self.scene.create_urdf_loader()
        # print('scale:', scale)
        loader.scale = scale
        self.object = loader.load(urdf, {"material": material})
        # random rotation around z axis
        theta_z_max = np.deg2rad(60)
        theta_z = (np.random.rand()*2-1) * theta_z_max + np.pi/12 # pi \pm theta_z_max + offset
        if 'Laptop' not in urdf:
            theta_z += np.pi
        # pose = Pose([self.object_position_offset, 0, 0], [np.cos(theta_z/2), 0, 0, np.sin(theta_z/2)])
        pose = Pose([self.object_position_offset, 0, 0], q)
        self.object.set_root_pose(pose)
        self.object_rotz = theta_z
        # compute link actor information
        self.active_joints = self.object.get_active_joints()
        while True:
            self.target_joint_id = random.choice(range(len(self.active_joints)))
            self.target_joint = self.active_joints[self.target_joint_id]
            if 'Drawer' in urdf:
                if self.target_joint.type == 'prismatic':
                    break
            else:
                break
        self.target_link = self.target_joint.get_child_link()

    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)

    def set_target_object_part_actor_id(self, actor_id):
        self.target_object_part_actor_id = actor_id
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id]))

        self.target_object_part_actor_link, self.target_object_part_joint_id = None, None
        # get the link handler
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_actor_link = j.get_child_link()
        
        # moniter the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_joint_id = idx
                idx += 1

        # if the target link is base link
        if self.target_object_part_actor_link is None:
            for j in self.object.get_links():
                if j.get_id() == actor_id:
                    self.target_object_part_actor_link = j
                    self.target_object_part_joint_id = -1  # set joint idx -1. This interaction is not VALID.

    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_target_part_qpos(self):
        qpos = self.object.get_qpos()
        return float(qpos[self.target_object_part_joint_id])
    
    def get_target_part_pose(self):
        return self.target_object_part_actor_link.get_pose()

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            # self.renderer_controller.show_window() # mlq:there is no function called show_window
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.viewer.render()
            return self.viewer

    def step(self):
        self.current_step += 1
        self.scene.step()

    def close_render(self):
        # if self.window:
        #     self.renderer_controller.hide_window() # mlq: there is no function called hide_window
        self.window = False
    
    def wait_to_start(self):
        print('press "e" to start\n')
        while not self.viewer.closed:
            self.scene.update_render()
            if self.show_gui:
                self.viewer.render()
            if self.viewer.window.key_down('e'):
                break

    def close(self):
        if self.show_gui:
            # self.viewer.set_scene(None)
            self.viewer.close()
        self.scene = None


