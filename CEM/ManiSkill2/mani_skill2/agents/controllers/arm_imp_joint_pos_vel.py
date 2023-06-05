import numpy as np
import sapien.core as sapien

from mani_skill2.agents.controllers.base_controller import BaseController


class ArmImpJointPosVelController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpJointPosVelController, self).__init__(
            controller_config, robot, control_freq
        )
        self.control_type = "torque"
        self.use_delta = controller_config[
            "use_delta"
        ]  # interpret action as delta or absolute pose
        assert (
            not self.interpolate
        ), "Impedance controller have 1-order filter, and do not need interpolate anymore."

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

        self.joint_kp = None
        self.joint_kd = None

        self.joint_pos_limits = self.robot.get_qlimits()[self.control_joint_index]
        self.curr_joint_pos: np.ndarray = self._get_curr_joint_pos()
        self.start_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.final_target_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()

        self.filter_para = controller_config["filter_para"]
        self.target_joint_pos_ex: np.ndarray = self.curr_joint_pos.copy()
        self.target_joint_vel_ex: np.ndarray = self.curr_joint_vel.copy()

    def simulation_step(self):
        self._sync_articulation()
        curr_target_joint_pos = self.final_target_joint_pos
        curr_target_joint_vel = self.final_target_joint_vel

        target_joint_pos = (
            self.filter_para * curr_target_joint_pos
            + (1 - self.filter_para) * self.target_joint_pos_ex
        )
        self.target_joint_pos_ex = target_joint_pos.copy()
        target_joint_vel = (
            self.filter_para * curr_target_joint_vel
            + (1 - self.filter_para) * self.target_joint_vel_ex
        )
        self.target_joint_vel_ex = target_joint_vel.copy()
        kd = self.joint_kd

        # torques = pos_err * kp + vel_err * kd
        position_error = target_joint_pos - self._get_curr_joint_pos()
        vel_pos_error = target_joint_vel - self._get_curr_joint_vel()
        desired_acc = np.multiply(
            np.array(position_error), np.array(self.joint_kp)
        ) + np.multiply(vel_pos_error, kd)

        # Return desired torques. The passive force  compensation is handled by the combined controller.
        torques = np.dot(self._get_mass_matrix(), desired_acc)

        qf = np.zeros(len(self.robot.get_active_joints()))
        qf[self.control_joint_index] = torques
        return qf

    def _get_mass_matrix(self):
        return self.controller_articulation.compute_manipulator_inertia_matrix()

    def set_joint_drive_property(self):
        # clear joint drive property and use pure torque control
        for j in self.control_joints:
            j.set_drive_property(0.0, 0.0)
            j.set_friction(0.0)

    def reset(self):
        self.curr_joint_pos: np.ndarray = self._get_curr_joint_pos()
        self.start_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.final_target_joint_pos: np.ndarray = self.curr_joint_pos.copy()
        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.target_joint_pos_ex: np.ndarray = self.curr_joint_pos.copy()
        self.target_joint_vel_ex: np.ndarray = self.curr_joint_vel.copy()


class ArmImpJointPosVelConstController(ArmImpJointPosVelController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpJointPosVelConstController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = self.num_control_joints * 2

        self.joint_kp = self.nums2array(
            controller_config["joint_kp"], self.num_control_joints
        )
        self.joint_kd = self.nums2array(
            controller_config["joint_kd"], self.num_control_joints
        )

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
        self.start_joint_pos = self._get_curr_joint_pos()
        self.start_joint_vel = self._get_curr_joint_vel()
        if self.use_delta:
            self.final_target_joint_pos = np.clip(
                action[: self.num_control_joints] + self.start_joint_pos,
                self.joint_pos_limits[:, 0],
                self.joint_pos_limits[:, 1],
            )
        else:
            self.final_target_joint_pos = np.clip(
                action[: self.num_control_joints],
                self.joint_pos_limits[:, 0],
                self.joint_pos_limits[:, 1],
            )
        self.final_target_joint_vel = action[self.num_control_joints :]


class ArmImpJointPosVelKpController(ArmImpJointPosVelController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpJointPosVelKpController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = self.num_control_joints * 3

        self.joint_kp_min = self.nums2array(
            controller_config["joint_kp_min"], self.num_control_joints
        )
        self.joint_kp_max = self.nums2array(
            controller_config["joint_kp_max"], self.num_control_joints
        )
        self.joint_kp = self.target_joint_kp = self.joint_kp_min - 1
        self.joint_kd = self.nums2array(
            controller_config["joint_kd"], self.num_control_joints
        )

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.concatenate(
                [
                    np.stack(
                        [self.joint_delta_pos_min, self.joint_delta_pos_max], axis=1
                    ),
                    np.stack([self.joint_vel_min, self.joint_vel_max], axis=1),
                    np.stack([self.joint_kp_min, self.joint_kp_max], axis=1),
                ]
            )
        else:
            return np.concatenate(
                [
                    np.stack([self.joint_pos_min, self.joint_pos_max], axis=1),
                    np.stack([self.joint_vel_min, self.joint_vel_max], axis=1),
                    np.stack([self.joint_kp_min, self.joint_kp_max], axis=1),
                ]
            )

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.start_joint_pos = self._get_curr_joint_pos()
        self.start_joint_vel = self._get_curr_joint_vel()
        if self.use_delta:
            self.final_target_joint_pos = np.clip(
                action[: self.num_control_joints] + self.start_joint_pos,
                self.joint_pos_limits[:, 0],
                self.joint_pos_limits[:, 1],
            )
        else:
            self.final_target_joint_pos = np.clip(
                action[: self.num_control_joints],
                self.joint_pos_limits[:, 0],
                self.joint_pos_limits[:, 1],
            )
        self.final_target_joint_vel = action[
            self.num_control_joints : 2 * self.num_control_joints
        ]
        self.target_joint_kp = action[2 * self.num_control_joints :]

    def reset(self):
        super(ArmImpJointPosVelController, self).reset()
        self.joint_kp = self.target_joint_kp = self.joint_kp_min - 1

    def simulation_step(self):
        if np.all(self.joint_kp < self.joint_kp_min):
            self.joint_kp = self.target_joint_kp
        else:
            self.joint_kp = (
                1 - self.filter_para
            ) * self.joint_kp + self.filter_para * self.target_joint_kp
        return super(ArmImpJointPosVelController, self).simulation_step()


class ArmImpJointPosVelKpKdController(ArmImpJointPosVelController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpJointPosVelKpKdController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = self.num_control_joints * 4

        self.joint_kp_min = self.nums2array(
            controller_config["joint_kp_min"], self.num_control_joints
        )
        self.joint_kp_max = self.nums2array(
            controller_config["joint_kp_max"], self.num_control_joints
        )
        self.joint_kd_min = self.nums2array(
            controller_config["joint_kd_min"], self.num_control_joints
        )
        self.joint_kd_max = self.nums2array(
            controller_config["joint_kd_max"], self.num_control_joints
        )
        self.joint_kp = self.target_joint_kp = self.joint_kp_min - 1
        self.joint_kd = self.target_joint_kd = self.joint_kd_min - 1

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.concatenate(
                [
                    np.stack(
                        [self.joint_delta_pos_min, self.joint_delta_pos_max], axis=1
                    ),
                    np.stack([self.joint_vel_min, self.joint_vel_max], axis=1),
                    np.stack([self.joint_kp_min, self.joint_kp_max], axis=1),
                    np.stack([self.joint_kd_min, self.joint_kd_max], axis=1),
                ]
            )
        else:
            return np.concatenate(
                [
                    np.stack([self.joint_pos_min, self.joint_pos_max], axis=1),
                    np.stack([self.joint_vel_min, self.joint_vel_max], axis=1),
                    np.stack([self.joint_kp_min, self.joint_kp_max], axis=1),
                    np.stack([self.joint_kd_min, self.joint_kd_max], axis=1),
                ]
            )

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.start_joint_pos = self._get_curr_joint_pos()
        self.start_joint_vel = self._get_curr_joint_vel()
        if self.use_delta:
            self.final_target_joint_pos = np.clip(
                action[: self.num_control_joints] + self.start_joint_pos,
                self.joint_pos_limits[:, 0],
                self.joint_pos_limits[:, 1],
            )
        else:
            self.final_target_joint_pos = np.clip(
                action[: self.num_control_joints],
                self.joint_pos_limits[:, 0],
                self.joint_pos_limits[:, 1],
            )

        self.final_target_joint_vel = action[
            self.num_control_joints : 2 * self.num_control_joints
        ]
        self.target_joint_kp = action[
            2 * self.num_control_joints : 3 * self.num_control_joints
        ]
        self.target_joint_kd = action[3 * self.num_control_joints :]

    def reset(self):
        super(ArmImpJointPosVelKpKdController, self).reset()
        self.joint_kp = self.target_joint_kp = self.joint_kp_min - 1
        self.joint_kd = self.target_joint_kd = self.joint_kd_min - 1

    def simulation_step(self):
        if np.all(self.joint_kp < self.joint_kp_min):
            self.joint_kp = self.target_joint_kp
        else:
            self.joint_kp = (
                1 - self.filter_para
            ) * self.joint_kp + self.filter_para * self.target_joint_kp
        if np.all(self.joint_kd < self.joint_kd_min):
            self.joint_kd = self.target_joint_kd
        else:
            self.joint_kd = (
                1 - self.filter_para
            ) * self.joint_kd + self.filter_para * self.target_joint_kd
        return super(ArmImpJointPosVelKpKdController, self).simulation_step()
