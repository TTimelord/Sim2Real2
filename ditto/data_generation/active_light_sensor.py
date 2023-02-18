from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
from sapien.core import ActorBase, Pose, Scene
from sapien.sensor.depth_processor import calc_main_depth_from_left_right_ir

import os

def get_texture(camera: sapien.CameraEntity, name, dtype="float"):
    """Get texture from camera.
    Faster data communication are supported by cupy.

    Args:
        camera (sapien.CameraEntity): SAPIEN camera
        name (str): texture name
        dtype (str, optional): texture dtype, [float/uint32].
            Defaults to "float".

    Returns:
        np.ndarray: texture
    """
    if os.environ.get("WITH_CUPY", 0) == 1:
        # dtype info is included already in the dl_tensor
        dlpack = camera.get_dl_tensor(name)
        return cupy.asnumpy(cupy.from_dlpack(dlpack))
    else:
        if dtype == "float":
            return camera.get_float_texture(name)
        elif dtype == "uint32":
            return camera.get_uint32_texture(name)
        else:
            raise NotImplementedError(f"Unsupported texture type: {dtype}")

def get_camera_seg(camera: sapien.CameraEntity):
    seg = get_texture(camera, "Segmentation", "uint32")  # [H, W, 4]
    # channel 0 is visual id (mesh-level)
    # channel 1 is actor id (actor-level)
    return seg[..., :2]

class ActiveLightSensor:
    def __init__(
        self,
        name: str,
        scene: Scene,
        mount: ActorBase,
        rgb_resolution: Tuple[int, int] = (960, 540),
        ir_resolution: Tuple[int, int] = (960, 540),
        rgb_intrinsic: np.ndarray = np.array([ 1.3875e03/2, 0., 9.8259e02/2, 0., 1.3862e03/2, 5.6532e02/2, 0., 0., 1. ]).reshape((3, 3)),
        ir_intrinsic: np.ndarray = np.array([ 672, 0., 475.94, 0., 672, 270.71, 0., 0., 1. ]).reshape((3, 3)),
        trans_pose_l: Pose = Pose(p=[ -0.0008183810985, -0.0173809196, -0.002242552045], q=[9.99986449e-01, 5.69235052e-04, 1.23234267e-03, -5.02592655e-03]),
        trans_pose_r: Pose = Pose(p=[ -0.0008183810985, -0.07214373, -0.002242552045], q=[9.99986449e-01, 5.69235052e-04, 1.23234267e-03, -5.02592655e-03]),
        light_pattern: str = "data_generation/d415-pattern-sq.png",
        max_depth: float = 9.0,
        min_depth: float = 0.0,
        ir_ambient_strength: float = 0.002,
        ir_light_dim_factor: float = 0.05,
        ir_light_fov: float = 1.57,
        ir_intensity: float = 5.0,
    ):
        self.name = name
        self._scene = scene
        self._cam_mount = mount
        self._rgb_w, self._rgb_h = rgb_resolution
        self._ir_w, self._ir_h = ir_resolution
        self._rgb_intrinsic = rgb_intrinsic
        self._ir_intrinsic = ir_intrinsic
        self._trans_pose_l = trans_pose_l
        self._trans_pose_r = trans_pose_r
        self._light_pattern = light_pattern
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._ir_ambient_strength = ir_ambient_strength
        self._ir_light_dim_factor = ir_light_dim_factor
        self._ir_light_fov = ir_light_fov
        self._ir_intensity = ir_intensity

        self._create_cameras()
        self.alight = self._scene.add_active_light(
            pose=Pose([0, 0, 0]),
            color=[0, 0, 0],
            fov=self._ir_light_fov,
            tex_path=self._light_pattern,
        )

    def _create_cameras(self):
        tran_pose0 = sapien.Pose([0, 0, 0])
        camera0 = self._scene.add_mounted_camera(
            f"{self.name}",
            self._cam_mount,
            tran_pose0,
            self._rgb_w,
            self._rgb_h,
            fovy=0.0,
            near=0.001,
            far=100,
        )
        camera0.set_perspective_parameters(
            0.1,
            100.0,
            self._rgb_intrinsic[0, 0],
            self._rgb_intrinsic[1, 1],
            self._rgb_intrinsic[0, 2],
            self._rgb_intrinsic[1, 2],
            self._rgb_intrinsic[0, 1],
        )

        camera1 = self._scene.add_mounted_camera(
            f"{self.name}_ir_left",
            self._cam_mount,
            self._trans_pose_l,
            self._ir_w,
            self._ir_h,
            fovy=0.0,
            near=0.001,
            far=100,
        )
        camera1.set_perspective_parameters(
            0.1,
            100.0,
            self._ir_intrinsic[0, 0],
            self._ir_intrinsic[1, 1],
            self._ir_intrinsic[0, 2],
            self._ir_intrinsic[1, 2],
            self._ir_intrinsic[0, 1],
        )
        camera2 = self._scene.add_mounted_camera(
            f"{self.name}_ir_right",
            self._cam_mount,
            self._trans_pose_r,
            self._ir_w,
            self._ir_h,
            fovy=0.0,
            near=0.001,
            far=100,
        )
        camera2.set_perspective_parameters(
            0.1,
            100.0,
            self._ir_intrinsic[0, 0],
            self._ir_intrinsic[1, 1],
            self._ir_intrinsic[0, 2],
            self._ir_intrinsic[1, 2],
            self._ir_intrinsic[0, 1],
        )

        self._cam_rgb, self._cam_ir_l, self._cam_ir_r = camera0, camera1, camera2

    def get_image_dict(self, rgb=True, visual_seg=True, actor_seg=True) -> OrderedDict:
        apos = t3d.quaternions.mat2quat(
            self._cam_mount.get_pose().to_transformation_matrix()[:3, :3]
            @ t3d.quaternions.quat2mat((-0.5, 0.5, 0.5, -0.5))
        )
        self.alight.set_pose(Pose(self._cam_mount.get_pose().p, apos))
        self.alight.set_color([0, 0, 0])

        self._scene.update_render()
        self._cam_rgb.take_picture()

        self._ir_mode()
        self._scene.update_render()
        self._cam_ir_l.take_picture()
        self._cam_ir_r.take_picture()

        self._normal_mode()
        self._scene.update_render()

        ir_l = get_texture(self._cam_ir_l, "Color")[:, :, 0]
        ir_r = get_texture(self._cam_ir_r, "Color")[:, :, 0]
        ir_l = self._float2uint8(ir_l)
        ir_r = self._float2uint8(ir_r)

        clean_depth = -get_texture(self._cam_rgb, "Position")[:, :, [2]]  # unit: meter

        # stereo
        ex_l = self._pose2cv2ex(self._trans_pose_l)
        ex_r = self._pose2cv2ex(self._trans_pose_r)
        ex_main = self._pose2cv2ex(Pose())

        depth = calc_main_depth_from_left_right_ir(
            ir_l,
            ir_r,
            ex_l,
            ex_r,
            ex_main,
            self._ir_intrinsic,
            self._ir_intrinsic,
            self._rgb_intrinsic,
            lr_consistency=False,
            main_cam_size=(self._rgb_w, self._rgb_h),
            ndisp=128,
            use_census=True,
            register_depth=True,
            census_wsize=7,
            use_noise=True,
                scale = 0.0, 
                blur_ksize = 0, 
                blur_ksigma = 0.03, # 0.03
                speckle_shape = 398.12,  # 398.12
                speckle_scale = 2.54e-3, # 2.54e-3
                gaussian_mu =  -0.231, # -0.231, 
                gaussian_sigma = 0.83, # 0.83
                seed = 0
        )
        
        depth[depth > self._max_depth] = 0
        depth[depth < self._min_depth] = 0

        sensor_dict = OrderedDict()
        sensor_dict["ir_l"] = ir_l
        sensor_dict["ir_r"] = ir_r
        sensor_dict["clean_depth"] = clean_depth
        sensor_dict["stereo_depth"] = depth
        if rgb:
            rgb = get_texture(self._cam_rgb, "Color")[:, :, :3]
            rgb = self._float2uint8(rgb)
            sensor_dict["rgb"] = rgb
        if visual_seg or actor_seg:
            seg = get_camera_seg(self._cam_rgb)
            if visual_seg:
                sensor_dict["visual_seg"] = seg[..., 0]
            if actor_seg:
                sensor_dict["actor_seg"] = seg[..., 1]

        return sensor_dict

    def _ir_mode(self):
        self._light_d = {}
        for l in self._scene.get_all_lights():
            self._light_d[l] = l.color
            l.set_color(l.color * self._ir_light_dim_factor)

        self._light_a = self._scene.ambient_light
        self._scene.set_ambient_light([self._ir_ambient_strength, 0, 0])
        self.alight.set_color([self._ir_intensity, 0, 0])

    def _normal_mode(self):
        for l in self._scene.get_all_lights():
            l.set_color(self._light_d[l])
        self._scene.set_ambient_light(self._light_a)
        self.alight.set_color([0, 0, 0])

    @staticmethod
    def _float2uint8(x):
        return (x * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _pose2cv2ex(pose):
        ros2opencv = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return ros2opencv @ np.linalg.inv(pose.to_transformation_matrix())
