import numpy as np
import sapien.core as sapien

from mani_skill2.agents.controllers.base_controller import BaseController
from mani_skill2.utils.geometry import rotate_2d_vec_by_angle


class MobilePDJointVelDecoupledController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(MobilePDJointVelDecoupledController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = self.num_control_joints
        self.control_type = "vel"
        self.frame = controller_config["frame"]
        assert self.frame in ["ego", "world"], f"Unknown frame: {self.frame}"
        if self.frame == "ego":
            assert self.num_control_joints == 3
            assert self.control_joints[0].name == "root_x_axis_joint"
            assert self.control_joints[1].name == "root_y_axis_joint"
            assert self.control_joints[2].name == "root_z_rotation_joint"

        self.joint_damping = self.nums2array(
            controller_config["joint_damping"], self.num_control_joints
        )
        self.joint_friction = self.nums2array(
            controller_config["joint_friction"], self.num_control_joints
        )

        self.joint_vel_min = self.nums2array(
            controller_config["joint_vel_min"], self.num_control_joints
        )
        self.joint_vel_max = self.nums2array(
            controller_config["joint_vel_max"], self.num_control_joints
        )

        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()

    @property
    def action_range(self) -> np.ndarray:
        return np.stack([self.joint_vel_min, self.joint_vel_max], axis=1)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.curr_step = 0
        self.start_joint_vel = self._get_curr_joint_vel()
        if self.frame == "world":
            self.final_target_joint_vel = action
        elif self.frame == "ego":
            new_action = np.zeros(3)
            new_action[2] = action[2]
            new_action[:2] = rotate_2d_vec_by_angle(
                action[:2], self._get_curr_joint_pos()[2]
            )
            self.final_target_joint_vel = new_action

    def simulation_step(self):
        self.curr_step += 1
        if self.interpolate:
            curr_target_joint_vel = (
                self.final_target_joint_vel - self.start_joint_vel
            ) / self.control_step * self.curr_step + self.start_joint_vel
        else:
            curr_target_joint_vel = self.final_target_joint_vel

        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_velocity_target(curr_target_joint_vel[j_idx])

    def set_joint_drive_property(self):
        # set joint drive property. For velocity control, Stiffness = 0
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_property(0, self.joint_damping[j_idx])
            j.set_friction(0.0)

    def reset(self):
        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()
