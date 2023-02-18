"""
 @author Zhenyu Jiang
 @email stevetod98@gmail.com
 @date 2022-01-14
 @desc NViSII renderer
"""

from statistics import mode

import numpy as np
import nvisii

from utils3d.utils.transform import Transform


class NViSIIRenderer:
    def __init__(
        self,
        resolution=(640, 480),
        spp=256,
        camera_kwargs={"field_of_view": np.pi / 3.0},
        light_kwargs={"intensity": 3},
    ):
        """init function

        Args:
            resolution (tuple, optional): render resolution. Defaults to (640, 480).
            spp (int, optional): sample point per pixel. Defaults to 256.
            camera_kwargs (dict, optional): camera configs. Defaults to {'field_of_view': np.pi / 3.0}.
            light_kwargs (dict, optional): light configs. Defaults to {'intensity': 3}.
        """
        self.spp = spp
        self.width, self.height = resolution
        self.camera_kwargs = camera_kwargs
        self.light_kwargs = light_kwargs

    def reset(self, camera_pose, light_pose, dome_light_color=(1.0, 1.0, 1.0)):
        """clear scenen, reset camera and light

        Args:
            camera_pose (np.ndarray): camera pose, 4*4
            light_pose (np.ndarray): light pose 4*4
            dome_light_color (tuple, optional): dome light color. Defaults to (1.0, 1.0, 1.0)
        """
        nvisii.clear_all()
        # Camera
        self.camera = nvisii.entity.create(name="camera")
        self.camera.set_transform(nvisii.transform.create(name="camera_transform"))
        self.camera.set_camera(
            nvisii.camera.create_from_fov(
                name="camera_camera", aspect=self.width / float(self.height), **self.camera_kwargs
            )
        )
        nvisii.set_camera_entity(self.camera)
        self.set_camera(camera_pose)

        # Light
        self.light = nvisii.entity.create(
            name="light_0",
            mesh=nvisii.mesh.create_plane("light_0", flip_z=True),
            transform=nvisii.transform.create("light_1"),
            light=nvisii.light.create("light_1"),
        )
        self.set_light(light_pose, **self.light_kwargs)

        # Dome color
        nvisii.set_dome_light_color(dome_light_color)

        # Floor
        # self.floor = nvisii.entity.create(
        #     name='floor',
        #     mesh=nvisii.mesh.create_plane('mesh_floor'),
        #     transform=nvisii.transform.create('transform_floor'),
        #     material=nvisii.material.create('material_floor'))
        # self.set_floor(self.opt['floor']['texture'],
        #                self.opt['floor']['scale'],
        #                self.opt['floor']['position'])

        self.objects = {}

    def update_objects(self, mesh_pose_dict):
        """update objects in the scene

        Args:
            mesh_pose_dict (dict): dict of {'name': (mesh_path: str, scale: tuple of ints, transform: np.ndarray)}

        Returns:
            list: list of new object names
            list: list of removed object names
        """
        new_objects = []
        removed_objects = []
        for k, (path, scale, transform) in mesh_pose_dict.items():
            transform = Transform.from_matrix(transform)
            if k not in self.objects.keys():
                obj = nvisii.import_scene(path)
                obj.transforms[0].set_position(transform.translation)
                obj.transforms[0].set_rotation(transform.rotation.as_quat())
                obj.transforms[0].set_scale(scale)
                self.objects[k] = obj
                new_objects.append(k)
            else:
                obj = self.objects[k]
                obj.transforms[0].set_position(transform.translation)
                obj.transforms[0].set_rotation(transform.rotation.as_quat())
                obj.transforms[0].set_scale(scale)
        for k in self.objects.keys():
            if k not in mesh_pose_dict.keys():
                for obj in self.objects[k].entities:
                    obj.remove(obj.get_name())
                removed_objects.append(k)
        for k in removed_objects:
            self.objects.pop(k)

        return new_objects, removed_objects

    def render(self):
        """render current scene

        Returns:
            np.ndarray: rgb image of H*W*4
        """
        rgb = nvisii.render(width=self.width, height=self.height, samples_per_pixel=self.spp)
        rgb = np.asarray(rgb).reshape(self.height, self.width, 4)
        # flip over x axis
        rgb = np.flip(rgb, axis=0)
        return rgb

    def set_camera(self, pose):
        """set camera pose

        Args:
            pose (np.ndarray): camera pose
        """
        transform = Transform.from_matrix(pose)
        self.camera.get_transform().set_position(transform.translation)
        self.camera.get_transform().set_rotation(transform.rotation.as_quat())

    def set_light(self, pose, intensity, scale=(1.0, 1.0, 1.0)):
        """set light pose

        Args:
            pose (np.ndarray): light pose
            intensity (float): light intensity
            scale (tuple, optional): light scale. Defaults to (1.0, 1.0, 1.0).
        """
        self.light.get_light().set_intensity(intensity)
        self.light.get_transform().set_scale(scale)

        transform = Transform.from_matrix(pose)
        self.light.get_transform().set_position(transform.translation)
        self.light.get_transform().set_rotation(transform.rotation.as_quat())

    # def set_floor(self, texture_path, scale, position):
    #     if hasattr(self, 'floor_texture'):
    #         self.floor_texture.remove(self.floor_texture.get_name())
    #     self.floor_texture = nvisii.texture.create_from_file(
    #         name='floor_texture', path=texture_path)
    #     self.floor.get_material().set_base_color_texture(self.floor_texture)
    #     self.floor.get_material().set_roughness(0.4)
    #     self.floor.get_material().set_specular(0)

    #     self.floor.get_transform().set_scale(scale)
    #     self.floor.get_transform().set_position(position)

    @staticmethod
    def init():
        """init NViSII backend"""
        nvisii.initialize(headless=True, verbose=False)
        nvisii.enable_denoiser()

    @staticmethod
    def deinit():
        """shut down NViSII backend"""
        nvisii.deinitialize()
