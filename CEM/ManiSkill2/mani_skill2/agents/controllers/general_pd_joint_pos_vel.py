import numpy as np
import sapien.core as sapien

from mani_skill2.agents.controllers.base_controller import BaseController


class GeneralPDJointPosVelController(BaseController):
    """General PD joint position and velocity controller"""

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(GeneralPDJointPosVelController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = self.num_control_joints * 2
        self.control_type = "pos"
        self.use_delta = controller_config[
            "use_delta"
        ]  # interpret action as delta or absolute pose

        self.joint_stiffness = self.nums2array(
            controller_config["joint_stiffness"], self.num_control_joints
        )
        self.joint_damping = self.nums2array(
            controller_config["joint_damping"], self.num_control_joints
        )
        self.joint_friction = self.nums2array(
            controller_config["joint_friction"], self.num_control_joints
        )

        if self.use_delta:
            self.joint_delta_pos_min = self.nums2array(
                controller_config["joint_delta_pos_min"], self.num_control_joints
            )
            self.joint_delta_pos_max = self.nums2array(
                controller_config["joint_delta_pos_max"], self.num_control_joints
            )
        else:
            self.joint_pos_min = self.nums2array(
                controller_config["joint_pos_min"], self.num_control_joints
            )
            self.joint_pos_max = self.nums2array(
                controller_config["joint_pos_max"], self.num_control_joints
            )

        self.joint_vel_min = self.nums2array(
            controller_config["joint_vel_min"], self.num_control_joints
        )
        self.joint_vel_max = self.nums2array(
            controller_config["joint_vel_max"], self.num_control_joints
        )

        self.curr_joint_pos: np.ndarray = self._get_curr_joint_pos()
        self.start_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.final_target_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.concatenate(
                [
                    np.stack(
                        [self.joint_delta_pos_min, self.joint_delta_pos_max], axis=1
                    ),
                    np.stack([self.joint_vel_min, self.joint_vel_max], axis=1),
                ]
            )

        else:
            return np.concatenate(
                [
                    np.stack([self.joint_pos_min, self.joint_pos_max], axis=1),
                    np.stack([self.joint_vel_min, self.joint_vel_max], axis=1),
                ]
            )

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.curr_step = 0
        self.start_joint_pos = self._get_curr_joint_pos()
        self.start_joint_vel = self._get_curr_joint_vel()
        if self.use_delta:
            self.final_target_joint_pos = (
                action[: self.num_control_joints] + self.start_joint_pos
            )
        else:
            self.final_target_joint_pos = action[: self.num_control_joints]
        self.final_target_joint_vel = action[self.num_control_joints :]

    def simulation_step(self):
        self.curr_step += 1
        if self.interpolate:
            curr_target_joint_pos = (
                self.final_target_joint_pos - self.start_joint_pos
            ) / self.control_step * self.curr_step + self.start_joint_pos
            curr_target_joint_vel = (
                self.final_target_joint_vel - self.start_joint_vel
            ) / self.control_step * self.curr_step + self.start_joint_vel
        else:
            curr_target_joint_pos = self.final_target_joint_pos
            curr_target_joint_vel = self.final_target_joint_vel

        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_target(curr_target_joint_pos[j_idx])
            j.set_drive_velocity_target(curr_target_joint_vel[j_idx])

    def set_joint_drive_property(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_property(self.joint_stiffness[j_idx], self.joint_damping[j_idx])
            j.set_friction(self.joint_friction[j_idx])

    def reset(self):
        self.curr_joint_pos: np.ndarray = self._get_curr_joint_pos()
        self.start_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.final_target_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()
