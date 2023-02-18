"""
    an RGB-D camera
"""
import numpy as np
from sapien.core import Pose
from env import Env
from active_light_sensor import ActiveLightSensor

class Sensor(object):

    def __init__(self, env: Env, phi=np.pi/5, theta=np.pi, random_position=False, fixed_position=False):
        builder = env.scene.create_actor_builder()
        camera_mount_actor = builder.build_kinematic()
        self.env = env

        self.sensor = ActiveLightSensor(
            name="sensor",
            scene=env.scene,
            mount=camera_mount_actor
        )

        # set camera extrinsics
        if random_position:
            # theta = np.random.random() * np.pi*2
            # phi = (np.random.random()+1) * np.pi/6
            theta = (np.random.random()*2-1) * np.pi/12 + np.pi/6 + np.pi # -pi/6 ~ pi/6 + pi/6
            phi = (np.random.random()*2+1) * np.pi/12 # pi/12 ~ pi/4
        if fixed_position:
            #theta = -np.pi/10
            #theta = -np.pi/8
            theta = np.pi
            phi = np.pi/10
        dist = 1.4
        pos = np.array([dist*np.cos(phi)*np.cos(theta), \
                dist*np.cos(phi)*np.sin(theta), \
                dist*np.sin(phi)])
        forward = -pos / np.linalg.norm(pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.vstack([forward, left, up]).T
        mat44[:3, 3] = pos      # mat44 is cam2world
        mat44[0, 3] += env.object_position_offset
        self.mat44 = mat44
        camera_mount_actor.set_pose(Pose.from_transformation_matrix(mat44))

        self.pos = pos
        self.dist = dist
        self.theta = theta
        self.phi = phi

    def get_observation(self):
        sensor_dict = self.sensor.get_image_dict(True, False, False)
        return sensor_dict["clean_depth"], sensor_dict["stereo_depth"], sensor_dict['rgb']

    def get_intrinsic(self):
        intrinsic = self.sensor._rgb_intrinsic
        return intrinsic
    
    def get_extrinsic(self):
        extrinsic = self.sensor._cam_rgb.get_extrinsic_matrix()
        return extrinsic
