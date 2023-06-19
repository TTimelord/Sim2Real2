from collections import OrderedDict
from typing import Tuple, List

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.sapien_utils import (
    check_joint_stuck,
    get_actor_by_name,
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class FixedXmate3RobotiqAllegro(BaseAgent):
    def __init__(self, *args, **kwargs):
        super(FixedXmate3RobotiqAllegro, self).__init__(*args, **kwargs)
        # self.finger1_link: sapien.LinkBase = get_actor_by_name(
        #     self._robot.get_links(), "left_inner_finger_pad"
        # )
        # self.finger2_link: sapien.LinkBase = get_actor_by_name(
        #     self._robot.get_links(), "right_inner_finger_pad"
        # )
        # self.finger_size = (0.03, 0.07, 0.0075)  # values from URDF
        # self.grasp_site: sapien.Link = get_entity_by_name(
        #     self._robot.get_links(), "grasp_convenient_link"
        # )
        # self.tool_link: sapien.Link = get_entity_by_name(
        #     self._robot.get_links(), "tool_2"
        # )
        links = self._robot.get_links()
        self.palm_link: sapien.LinkBase = get_actor_by_name(
            links, "palm_link"
        )
        self.palm_size = (0.0408, 0.1130, 0.095)
        self.link0_0: sapien.LinkBase = get_actor_by_name(
            links, "link_0.0"
        )
        self.link0_4_8_size = (0.0196, 0.0275, 0.0164)
        self.link1_0: sapien.LinkBase = get_actor_by_name(
            links, "link_1.0"
        )
        self.link1_5_9_size = (0.0196, 0.0275, 0.054)
        self.link2_0: sapien.LinkBase = get_actor_by_name(
            links, "link_2.0"
        )
        self.link2_6_10_size = (0.0196, 0.0275, 0.0384)
        self.link3_0: sapien.LinkBase = get_actor_by_name(
            links, "link_3.0"
        )
        self.link3_7_11_size = (0.0196, 0.0275, 0.0267)
        self.link3_0_tip: sapien.LinkBase = get_actor_by_name(
            links, "link_3.0_tip"
        )
        self.finger_tip_radius = 0.012
        self.link4_0: sapien.LinkBase = get_actor_by_name(
            links, "link_4.0"
        )
        self.link5_0: sapien.LinkBase = get_actor_by_name(
            links, "link_5.0"
        )
        self.link6_0: sapien.LinkBase = get_actor_by_name(
            links, "link_6.0"
        )
        self.link7_0: sapien.LinkBase = get_actor_by_name(
            links, "link_7.0"
        )
        self.link7_0_tip: sapien.LinkBase = get_actor_by_name(
            links, "link_7.0_tip"
        )
        self.link8_0: sapien.LinkBase = get_actor_by_name(
            links, "link_8.0"
        )
        self.link9_0: sapien.LinkBase = get_actor_by_name(
            links, "link_9.0"
        )
        self.link10_0: sapien.LinkBase = get_actor_by_name(
            links, "link_10.0"
        )
        self.link11_0: sapien.LinkBase = get_actor_by_name(
            links, "link_11.0"
        )
        self.link11_0_tip: sapien.LinkBase = get_actor_by_name(
            links, "link_11.0_tip"
        )
        self.link12_0: sapien.LinkBase = get_actor_by_name(
            links, "link_12.0"
        )
        self.link12_size = (0.0358, 0.034, 0.0455)
        self.link13_0: sapien.LinkBase = get_actor_by_name(
            links, "link_13.0"
        )
        self.link13_size = (0.0196, 0.0275, 0.0177)
        self.link14_0: sapien.LinkBase = get_actor_by_name(
            links, "link_14.0"
        )
        self.link14_size = (0.0196, 0.0275, 0.0514)
        self.link15_0: sapien.LinkBase = get_actor_by_name(
            links, "link_15.0"
        )
        self.link15_size = (0.0196, 0.0275, 0.0423)
        self.link15_0_tip: sapien.LinkBase = get_actor_by_name(
            links, "link_15.0_tip"
        )

    def get_proprioception(self):
        state_dict = OrderedDict()
        qpos = self._robot.get_qpos()
        qvel = self._robot.get_qvel()

        state_dict["qpos"] = qpos
        state_dict["qvel"] = qvel
        # state_dict["tcp_wrench"] = self.get_tcp_wrench()
        # state_dict["joint_external_torque"] = self.get_generalized_external_forces()
        # state_dict["gripper_grasp"] = np.array([self.check_gripper_grasp_real()]).astype(np.float)

        return state_dict

    # def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
    #     assert isinstance(actor, sapien.ActorBase), type(actor)
    #     contacts = self._scene.get_contacts()
    #
    #     limpulse = get_pairwise_contact_impulse(contacts, self.palm_link, actor)
    #     rimpulse = get_pairwise_contact_impulse(contacts, self.link3_0_tip, actor)
    #
    #     # direction to open the gripper
    #     ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 2]
    #     rdirection = self.finger2_link.pose.to_transformation_matrix()[:3, 2]
    #
    #     # angle between impulse and open direction
    #     langle = compute_angle_between(ldirection, limpulse)
    #     rangle = compute_angle_between(rdirection, rimpulse)
    #
    #     lflag = (
    #         np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
    #     )
    #     rflag = (
    #         np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
    #     )
    #
    #     return all([lflag, rflag])

    def sample_ee_coords(self, num_sample=10) -> np.ndarray:
        """Uniformly sample points on the two finger meshes. Used for dense reward computation
        return: ee_coords (2, num_sample, 3)"""
        def get_single_samples(original_size: Tuple, links: List):
            points = (
                np.arange(num_sample) / (num_sample - 1) - 0.5
            ) * original_size[1]
            points = np.stack(
                [np.zeros(num_sample), points, np.zeros(num_sample)], axis=1
            )  # (num_sample, 3)
            return [
                transform_points(link.get_pose().to_transformation_matrix(), points)
                for link in links
            ]

        def sample_semi_sphere_points(radius, links: List):
            samples = []
            for _ in range(num_sample):
                # phi is the angle off the z-axis, ranging from 0 to pi/2 (semi-sphere)
                phi = np.random.uniform(0, np.pi / 2)
                # theta is the azimuthal angle, ranging from 0 to 2pi
                theta = np.random.uniform(0, 2 * np.pi)

                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)

                samples.append((x, y, z))
            return [
                transform_points(link.get_pose().to_transformation_matrix(), np.array(samples))
                for link in links
            ]

        palm_points = get_single_samples(self.palm_size, [self.palm_link])[0]
        link0_points, link4_points, link8_points = get_single_samples(self.link0_4_8_size, [self.link0_0, self.link4_0, self.link8_0])
        link1_points, link5_points, link9_points = get_single_samples(self.link1_5_9_size, [self.link1_0, self.link5_0, self.link9_0])
        link2_points, link6_points, link10_points = get_single_samples(self.link2_6_10_size, [self.link2_0, self.link6_0, self.link10_0])
        link3_points, link7_points, link11_points = get_single_samples(self.link3_7_11_size, [self.link3_0, self.link7_0, self.link11_0])
        link12_points = get_single_samples(self.link12_size, [self.link12_0])
        link13_points = get_single_samples(self.link13_size, [self.link13_0])
        link14_points = get_single_samples(self.link14_size, [self.link14_0])
        link15_points = get_single_samples(self.link15_size, [self.link15_0])
        link3_tip_points, link7_tip_points, link11_tip_points, link15_tip_points = sample_semi_sphere_points(self.finger_tip_radius, [self.link3_0_tip, self.link7_0_tip, self.link11_0_tip, self.link15_0_tip])

        ee_coords = np.stack((palm_points, link0_points, link1_points, link2_points, link3_points, link3_tip_points,
                              link4_points, link5_points, link6_points, link7_points, link7_tip_points,
                              link8_points, link9_points, link10_points, link11_points, link11_tip_points,
                              link12_points, link13_points, link14_points, link15_points, link15_tip_points))

        return ee_coords

    # def get_tcp_wrench(self):
    #     joint_tau = self.get_generalized_external_forces()[:7]
    #     controller = self._combined_controllers[self._control_mode]._controllers[0]
    #     assert controller.control_joint_names == [
    #         "joint1",
    #         "joint2",
    #         "joint3",
    #         "joint4",
    #         "joint5",
    #         "joint6",
    #         "joint7",
    #     ]
    #     controller._sync_articulation()
    #     J_full = controller.controller_articulation.compute_world_cartesian_jacobian()[
    #         -6:
    #     ]
    #     J = np.linalg.pinv(J_full.T)
    #
    #     TCP_wrench = J @ joint_tau
    #     return TCP_wrench
    #
    # @staticmethod
    # def build_grasp_pose(forward, flat, center):
    #     extra = np.cross(flat, forward)
    #     ans = np.eye(4)
    #     ans[:3, :3] = np.array([forward, flat, -extra]).T
    #     ans[:3, 3] = center
    #     return Pose.from_transformation_matrix(ans)
    #
    # def check_gripper_grasp_real(self) -> bool:
    #     """check whether the gripper is grasping something by checking the joint position and velocity"""
    #     from agents.controllers import GripperPDJointPosMimicController
    #
    #     assert isinstance(
    #         self._combined_controllers[self._control_mode]._controllers[1],
    #         GripperPDJointPosMimicController,
    #     )
    #     for joint_idx, joint in enumerate(self._robot.get_active_joints()):
    #         if joint.name == "robotiq_2f_140_left_driver_joint":
    #             active_joint1_idx = joint_idx
    #         if joint.name == "robotiq_2f_140_right_driver_joint":
    #             active_joint2_idx = joint_idx
    #
    #     joint1_stuck = check_joint_stuck(self._robot, active_joint1_idx)
    #     joint2_stuck = check_joint_stuck(self._robot, active_joint2_idx)
    #
    #     return joint1_stuck or joint2_stuck
