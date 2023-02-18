import numpy as np
import quaternion
import sapien.core as sapien
import transforms3d as t3d

from mani_skill2.agents.control_utils import (
    inverse_se3_matrix,
    nullspace_torques,
    opspace_matrices,
    orientation_error,
)
from mani_skill2.agents.controllers.base_controller import BaseController


class ArmImpEEPosVelController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEPosVelController, self).__init__(
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
            self.ee_delta_pos_min = self.nums2array(
                controller_config["ee_delta_pos_min"], 6
            )
            self.ee_delta_pos_max = self.nums2array(
                controller_config["ee_delta_pos_max"], 6
            )
        else:
            self.ee_pos_min = self.nums2array(controller_config["ee_pos_min"], 6)
            self.ee_pos_max = self.nums2array(controller_config["ee_pos_max"], 6)

        self.ee_vel_min = self.nums2array(controller_config["ee_vel_min"], 6)
        self.ee_vel_max = self.nums2array(controller_config["ee_vel_max"], 6)

        self.filter_para = controller_config["filter_para"]
        self.ee_kp = None
        self.ee_kd = None

        curr_base_pose_RT = self.start_link.get_pose().to_transformation_matrix()
        curr_ee_pose_RT = self.end_link.get_pose().to_transformation_matrix()
        self.start_rel_pose_RT = inverse_se3_matrix(curr_base_pose_RT) @ curr_ee_pose_RT
        self.target_ee_pos_ex = self.final_ee_pos = self.start_rel_pose_RT[:3, 3]
        self.target_ee_quat_ex = self.final_ee_quat = quaternion.from_rotation_matrix(
            self.start_rel_pose_RT[:3, :3]
        )
        self.init_joint_pos = self._get_curr_joint_pos()  # used for nullspace stiffness
        self.nullspace_stiffness = np.array(
            [controller_config["nullspace_stiffness"]], dtype=np.float32
        )
        self.final_ee_vel_base = np.zeros(6)
        self.target_ee_vel_base_ex = np.zeros(6)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self._sync_articulation()
        curr_base_pose_RT = self.start_link.get_pose().to_transformation_matrix()
        curr_ee_pose_RT = self.end_link.get_pose().to_transformation_matrix()
        self.start_rel_pose_RT = inverse_se3_matrix(curr_base_pose_RT) @ curr_ee_pose_RT
        RT = np.eye(4)
        angle = np.linalg.norm(action[3:6])
        if angle < 1e-6:
            axis = (0, 0, 1)
        else:
            axis = action[3:6] / angle
        RT[:3, :3] = t3d.axangles.axangle2mat(axis, angle)
        RT[:3, 3] = action[:3]
        if self.use_delta:
            final_rel_pose_RT = self.start_rel_pose_RT @ RT
        else:
            final_rel_pose_RT = RT  #
        self.final_ee_pos = final_rel_pose_RT[:3, 3]
        self.final_ee_quat = quaternion.from_rotation_matrix(final_rel_pose_RT[:3, :3])
        self.final_ee_vel_base = action[6:12]

    def simulation_step(self):
        self._sync_articulation()
        curr_base_pose_RT = self.start_link.get_pose().to_transformation_matrix()
        curr_ee_pose_RT = self.end_link.get_pose().to_transformation_matrix()
        curr_rel_pose_RT = inverse_se3_matrix(curr_base_pose_RT) @ curr_ee_pose_RT

        # filter target pose
        curr_target_pos = self.final_ee_pos
        curr_target_quat = self.final_ee_quat
        curr_target_pos = (
            self.filter_para * curr_target_pos
            + (1 - self.filter_para) * self.target_ee_pos_ex
        )
        curr_target_quat = quaternion.slerp_evaluate(
            self.target_ee_quat_ex, curr_target_quat, self.filter_para
        )

        self.target_ee_pos_ex = curr_target_pos.copy()
        self.target_ee_quat_ex = curr_target_quat.copy()

        err_pos = curr_target_pos - curr_rel_pose_RT[:3, 3]
        err_ori = orientation_error(
            quaternion.as_rotation_matrix(curr_target_quat), curr_rel_pose_RT[:3, :3]
        )
        J_full = self.controller_articulation.compute_world_cartesian_jacobian()[-6:]

        J_pos, J_ori = J_full[:3, :], J_full[3:, :]
        curr_joint_vel = self._get_curr_joint_vel()
        curr_vel_base = J_full @ curr_joint_vel
        target_ee_vel_base = (
            self.filter_para * curr_vel_base
            + (1 - self.filter_para) * self.final_ee_vel_base
        )
        self.target_ee_vel_base_ex = target_ee_vel_base.copy()

        # Compute desired force and torque based on errors
        vel_pos_error = target_ee_vel_base[:3] - curr_vel_base[:3]

        # F_r = kp * pos_err + kd * vel_err
        desired_force = np.multiply(
            np.array(err_pos), np.array(self.ee_kp[0:3])
        ) + np.multiply(vel_pos_error, self.ee_kd[0:3])

        vel_ori_error = target_ee_vel_base[3:] - curr_vel_base[3:]

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = np.multiply(
            np.array(err_ori), np.array(self.ee_kp[3:6])
        ) + np.multiply(vel_ori_error, self.ee_kd[3:6])

        mass_matrix = self._get_mass_matrix()
        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            mass_matrix, J_full, J_pos, J_ori
        )

        desired_wrench = np.concatenate([desired_force, desired_torque])
        decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F. The passive force  compensation is handled by the combined controller.
        torques = np.dot(J_full.T, decoupled_wrench)

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        torques += nullspace_torques(
            mass_matrix,
            nullspace_matrix,
            self.init_joint_pos,
            self._get_curr_joint_pos(),
            curr_joint_vel,
            self.nullspace_stiffness,
        )

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
        self._sync_articulation()
        self.init_joint_pos = self._get_curr_joint_pos()
        curr_base_pose_RT = self.start_link.get_pose().to_transformation_matrix()
        curr_ee_pose_RT = self.end_link.get_pose().to_transformation_matrix()
        self.start_rel_pose_RT = inverse_se3_matrix(curr_base_pose_RT) @ curr_ee_pose_RT
        self.target_ee_pos_ex = self.final_ee_pos = self.start_rel_pose_RT[:3, 3]
        self.target_ee_quat_ex = self.final_ee_quat = quaternion.from_rotation_matrix(
            self.start_rel_pose_RT[:3, :3]
        )
        J_full = self.controller_articulation.compute_world_cartesian_jacobian()[-6:]
        curr_joint_vel = self._get_curr_joint_vel()
        self.final_ee_vel_base = J_full @ curr_joint_vel
        self.target_ee_vel_base_ex = self.final_ee_vel_base.copy()


class ArmImpEEPosVelConstController(ArmImpEEPosVelController):
    action_dimension = 12  # 3 for translation, 3 for rotation in axis-angle

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEPosVelConstController, self).__init__(
            controller_config, robot, control_freq
        )

        self.ee_kp = self.nums2array(controller_config["ee_kp"], 6)
        self.ee_kd = self.nums2array(controller_config["ee_kd"], 6)

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.concatenate(
                [
                    np.stack([self.ee_delta_pos_min, self.ee_delta_pos_max], axis=1),
                    np.stack([self.ee_vel_min, self.ee_vel_max], axis=1),
                ]
            )
        else:
            return np.concatenate(
                [
                    np.stack([self.ee_pos_min, self.ee_pos_max], axis=1),
                    np.stack([self.ee_vel_min, self.ee_vel_max], axis=1),
                ]
            )


class ArmImpEEPosVelKpController(ArmImpEEPosVelController):
    action_dimension = 18

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEPosVelKpController, self).__init__(
            controller_config, robot, control_freq
        )
        self.ee_kp_min = self.nums2array(controller_config["ee_kp_min"], 6)
        self.ee_kp_max = self.nums2array(controller_config["ee_kp_max"], 6)
        self.ee_kp = self.target_ee_kp = self.ee_kp_min - 1
        self.ee_kd = self.nums2array(controller_config["ee_kd"], 6)

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.concatenate(
                [
                    np.stack([self.ee_delta_pos_min, self.ee_delta_pos_max], axis=1),
                    np.stack([self.ee_vel_min, self.ee_vel_max], axis=1),
                    np.stack([self.ee_kp_min, self.ee_kp_max], axis=1),
                ]
            )
        else:
            return np.concatenate(
                [
                    np.stack([self.ee_pos_min, self.ee_pos_max], axis=1),
                    np.stack([self.ee_vel_min, self.ee_vel_max], axis=1),
                    np.stack([self.ee_kp_min, self.ee_kp_max], axis=1),
                ]
            )

    def set_action(self, action: np.ndarray):
        super(ArmImpEEPosVelKpController, self).set_action(action)
        self.target_ee_kp = action[12:18]

    def reset(self):
        super(ArmImpEEPosVelKpController, self).reset()
        self.ee_kp = self.target_ee_kp = self.ee_kp_min - 1

    def simulation_step(self):
        if np.all(self.ee_kp < self.ee_kp_min):
            self.ee_kp = self.target_ee_kp
        else:
            self.ee_kp = (
                1 - self.filter_para
            ) * self.ee_kp + self.filter_para * self.target_ee_kp
        return super(ArmImpEEPosVelKpController, self).simulation_step()


class ArmImpEEPosVelKpKdController(ArmImpEEPosVelController):
    action_dimension = 24

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEPosVelKpKdController, self).__init__(
            controller_config, robot, control_freq
        )
        self.ee_kp_min = self.nums2array(controller_config["ee_kp_min"], 6)
        self.ee_kp_max = self.nums2array(controller_config["ee_kp_max"], 6)
        self.ee_kp = self.target_ee_kp = self.ee_kp_min - 1
        self.ee_kd_min = self.nums2array(controller_config["ee_kd_min"], 6)
        self.ee_kd_max = self.nums2array(controller_config["ee_kd_max"], 6)
        self.ee_kd = self.target_ee_kd = self.ee_kd_min - 1

    @property
    def action_range(self) -> np.ndarray:
        if self.use_delta:
            return np.concatenate(
                [
                    np.stack([self.ee_delta_pos_min, self.ee_delta_pos_max], axis=1),
                    np.stack([self.ee_vel_min, self.ee_vel_max], axis=1),
                    np.stack([self.ee_kp_min, self.ee_kp_max], axis=1),
                    np.stack([self.ee_kd_min, self.ee_kd_max], axis=1),
                ]
            )
        else:
            return np.concatenate(
                [
                    np.stack([self.ee_pos_min, self.ee_pos_max], axis=1),
                    np.stack([self.ee_vel_min, self.ee_vel_max], axis=1),
                    np.stack([self.ee_kp_min, self.ee_kp_max], axis=1),
                    np.stack([self.ee_kd_min, self.ee_kd_max], axis=1),
                ]
            )

    def set_action(self, action: np.ndarray):
        super(ArmImpEEPosVelKpKdController, self).set_action(action)
        self.target_ee_kp = action[12:18]
        self.target_ee_kd = action[18:24]

    def reset(self):
        super(ArmImpEEPosVelKpKdController, self).reset()
        self.ee_kp = self.target_ee_kp = self.ee_kp_min - 1
        self.ee_kd = self.target_ee_kd = self.ee_kd_min - 1

    def simulation_step(self):
        if np.all(self.ee_kp < self.ee_kp_min):  # init
            self.ee_kp = self.target_ee_kp
        else:
            self.ee_kp = (
                1 - self.filter_para
            ) * self.ee_kp + self.filter_para * self.target_ee_kp
        if np.all(self.ee_kd < self.ee_kd_min):  # init
            self.ee_kd = self.target_ee_kd
        else:
            self.ee_kd = (
                1 - self.filter_para
            ) * self.ee_kd + self.filter_para * self.target_ee_kd
        return super(ArmImpEEPosVelKpKdController, self).simulation_step()
