import numpy as np
import sapien.core as sapien

from mani_skill2.agents.controllers.base_controller import BaseController


class GripperPDJointVelMimicController(BaseController):
    def __init__(
        self, controller_config: dict, robot: sapien.Articulation, control_freq: int
    ):
        super(GripperPDJointVelMimicController, self).__init__(
            controller_config, robot, control_freq
        )
        self.action_dimension = 1
        self.control_type = "vel"

        self.joint_damping = controller_config["joint_damping"]

        self.joint_vel_min = controller_config["joint_vel_min"]
        self.joint_vel_max = controller_config["joint_vel_max"]

        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()

    @property
    def action_range(self) -> np.ndarray:
        return np.array([self.joint_vel_min, self.joint_vel_max]).reshape(1, 2)

    def set_action(self, action: np.ndarray):
        assert action.shape[0] == self.action_dimension
        self.curr_step = 0
        self.start_joint_vel = self._get_curr_joint_vel()
        self.final_target_joint_vel = np.ones(self.num_control_joints) * action

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
            j.set_drive_property(0, self.joint_damping)
            j.set_friction(0.0)

    def reset(self):
        self.curr_joint_vel: np.ndarray = self._get_curr_joint_vel()
        self.start_joint_vel: np.ndarray = self.curr_joint_vel.copy()
        self.final_target_joint_vel: np.ndarray = self.curr_joint_vel.copy()
