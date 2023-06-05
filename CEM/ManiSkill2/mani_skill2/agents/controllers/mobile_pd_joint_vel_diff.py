import numpy as np
import sapien.core as sapien

from mani_skill2.agents.controllers.base_controller import BaseController


class MobilePDJointVelDiffController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(MobilePDJointVelDiffController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = 2  # forward velocity, angular velocity
        self.control_type = "vel"

        assert (
            self.num_control_joints == 3
        ), "Number of controlled joints should be 3. (X, Y, rotZ)"

        self.joint_damping = self.nums2array(
            controller_config["joint_damping"], self.num_control_joints
        )
        self.joint_friction = self.nums2array(
            controller_config["joint_friction"], self.num_control_joints
        )

        self.diff_vel_min = self.nums2array(controller_config["diff_vel_min"], 2)
        self.diff_vel_max = self.nums2array(controller_config["diff_vel_max"], 2)

        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()

    @property
    def action_range(self) -> np.ndarray:
        return np.stack([self.diff_vel_min, self.diff_vel_max], axis=1)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.curr_step = 0
        self.start_joint_vel = self._get_curr_joint_vel()
        curr_joint_pos = self._get_curr_joint_pos()
        self.final_target_joint_vel = np.array(
            [
                action[0] * np.cos(curr_joint_pos[2]),
                action[0] * np.sin(curr_joint_pos[2]),
                action[1],
            ]
        )

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
