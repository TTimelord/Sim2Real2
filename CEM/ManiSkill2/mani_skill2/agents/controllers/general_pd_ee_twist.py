import numpy as np
import sapien.core as sapien

from mani_skill2.agents.controllers.base_controller import BaseController


class GeneralPDEETwistController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(GeneralPDEETwistController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = 7  # 6D twist + 1 scale scalar
        self.control_type = "vel"

        self.joint_damping = self.nums2array(
            controller_config["joint_damping"], self.num_control_joints
        )

        self.joint_vel_min = self.nums2array(
            controller_config["joint_vel_min"], self.num_control_joints
        )
        self.joint_vel_max = self.nums2array(
            controller_config["joint_vel_max"], self.num_control_joints
        )

        self.ee_twist_min = self.nums2array(controller_config["ee_twist_min"], 7)
        self.ee_twist_max = self.nums2array(controller_config["ee_twist_max"], 7)

        self.regularization_weight = controller_config["regularization_weight"]

        self.pmodel = self.controller_articulation.create_pinocchio_model()

        self.init_joint_pos = self._get_curr_joint_pos()
        self.target_joint_vel = self._get_curr_joint_vel()

    @property
    def action_range(self) -> np.ndarray:
        return np.stack([self.ee_twist_min, self.ee_twist_max], axis=1)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        curr_qpos = self._get_curr_joint_pos()
        self.pmodel.compute_full_jacobian(curr_qpos)
        J = self.pmodel.get_link_jacobian(self.num_control_joints, local=True)
        delta_q = (
            np.array(self.init_joint_pos - curr_qpos).reshape((-1, 1))
            * self.regularization_weight
        )
        target_xi = action[:6].reshape((-1, 1))
        target_xi = target_xi / (np.linalg.norm(target_xi) + 1e-6) * (action[6] + 1)
        delta_xi = J @ delta_q
        q = np.linalg.pinv(J) @ (target_xi - delta_xi) + delta_q
        self.target_joint_vel = np.clip(q[:, 0], self.joint_vel_min, self.joint_vel_max)

    def simulation_step(self):
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_velocity_target(self.target_joint_vel[j_idx])

    def set_joint_drive_property(self):
        # set joint drive property. For velocity control, Stiffness = 0
        for j_idx, j in enumerate(self.control_joints):
            j.set_drive_property(0, self.joint_damping[j_idx])
            j.set_friction(0.0)

    def reset(self):
        self.init_joint_pos = self._get_curr_joint_pos()
        self.target_joint_vel = self._get_curr_joint_vel()
