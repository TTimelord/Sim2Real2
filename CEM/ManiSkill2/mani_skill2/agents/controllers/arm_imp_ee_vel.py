import numpy as np
import quaternion
import sapien.core as sapien
import transforms3d as t3d

from mani_skill2.agents.control_utils import (
    nullspace_torques,
    opspace_matrices,
    orientation_error,
)
from mani_skill2.agents.controllers.base_controller import BaseController


class ArmImpEEVelController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEVelController, self).__init__(
            controller_config, robot, control_freq
        )
        self.control_type = "torque"
        self.use_delta = controller_config[
            "use_delta"
        ]  # interpret action as delta or absolute pose
        assert (
            not self.interpolate
        ), "Impedance controller have 1-order filter, and do not need interpolate anymore."

        self.ee_vel_min = self.nums2array(controller_config["ee_vel_min"], 6)
        self.ee_vel_max = self.nums2array(controller_config["ee_vel_max"], 6)

        self.filter_para = controller_config["filter_para"]
        self.ee_kd = None

        self.init_joint_pos = self._get_curr_joint_pos()
        self.nullspace_stiffness = np.array(
            [controller_config["nullspace_stiffness"]], dtype=np.float32
        )
        self.final_ee_vel_base = np.zeros(6)
        self.target_ee_vel_base_ex = np.zeros(6)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.final_ee_vel_base = action[:6]

    def simulation_step(self):
        self._sync_articulation()
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
        desired_force = np.multiply(vel_pos_error, self.ee_kd[0:3])

        vel_ori_error = target_ee_vel_base[3:] - curr_vel_base[3:]

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = np.multiply(vel_ori_error, self.ee_kd[3:6])

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
        J_full = self.controller_articulation.compute_world_cartesian_jacobian()[-6:]
        curr_joint_vel = self._get_curr_joint_vel()
        self.final_ee_vel_base = J_full @ curr_joint_vel
        self.target_ee_vel_base_ex = self.final_ee_vel_base.copy()


class ArmImpEEVelConstController(ArmImpEEVelController):
    action_dimension = 6  # velocity in base frame

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEVelConstController, self).__init__(
            controller_config, robot, control_freq
        )

        self.ee_kd = self.nums2array(controller_config["ee_kd"], 6)

    @property
    def action_range(self) -> np.ndarray:
        return np.stack([self.ee_vel_min, self.ee_vel_max], axis=1)


class ArmImpEEVelKdController(ArmImpEEVelController):
    action_dimension = 12

    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(ArmImpEEVelKdController, self).__init__(
            controller_config, robot, control_freq
        )
        self.ee_kd_min = self.nums2array(controller_config["ee_kd_min"], 6)
        self.ee_kd_max = self.nums2array(controller_config["ee_kd_max"], 6)
        self.ee_kd = self.target_ee_kd = self.ee_kd_min - 1

    @property
    def action_range(self) -> np.ndarray:
        return np.concatenate(
            [
                np.stack([self.ee_vel_min, self.ee_vel_max], axis=1),
                np.stack([self.ee_kd_min, self.ee_kd_max], axis=1),
            ]
        )

    def set_action(self, action: np.ndarray):
        super(ArmImpEEVelKdController, self).set_action(action)
        self.target_ee_kd = action[6:]

    def reset(self):
        super(ArmImpEEVelKdController, self).reset()
        self.ee_kd = self.target_ee_kd = self.ee_kd_min - 1

    def simulation_step(self):
        if np.all(self.ee_kd < self.ee_kd_min):  # init
            self.ee_kd = self.target_ee_kd
        else:
            self.ee_kd = (
                1 - self.filter_para
            ) * self.ee_kd + self.filter_para * self.target_ee_kd
        return super(ArmImpEEVelKdController, self).simulation_step()
