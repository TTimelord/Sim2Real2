"""
    xmate3_arm with roboqiq grisper
"""

from __future__ import division
import numpy as np
import rospy
from impedance_control.msg import JointControlCommand, CartesianControlCommand, ImpedanceRobotState
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output, Robotiq2FGripper_robot_input
from tqdm import tqdm

# simulated robot in sapien
class Robot(object):
    def __init__(self, env, urdf, material, open_gripper=False, control_timestep=0.02):
        self.env = env
        self.timestep = env.scene.get_timestep()
        self.control_timestep = control_timestep
        # load robot
        loader = env.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf, {"material": material})
        #self.robot = loader.load(urdf, material)
        self.robot.name = "xmate3_robot"

        # define the distance from grasp point to the gripper root
        self.len_grasp_to_root = 0.225

        # hand (EE), two grippers, the rest arm joints (if any)
        self.end_effector_index, self.end_effector = \
            [(i, l) for i, l in enumerate(self.robot.get_links()) if l.name == 'grasp_convenient_link'][0]
        self.hand_actor_id, _ = \
            [(i, l) for i, l in enumerate(self.robot.get_links()) if l.name == 'robotiq_arg2f_base_link'][0]
        self.gripper_joints = [joint for joint in self.robot.get_active_joints() if 
                joint.get_name().endswith("driver_joint")]
        self.gripper_actor_ids = [joint.get_child_link().get_id() for joint in self.gripper_joints]
        self.arm_joints = [joint for joint in self.robot.get_active_joints() if
                joint.get_name().startswith("joint")]

        # set drive joint property
        for joint in self.arm_joints:
            joint.set_drive_property(1500, 400)
        for joint in self.gripper_joints:
            joint.set_drive_property(500, 120)

        # open/close the gripper at start
        if open_gripper:
            joint_angles = []
            for j in self.robot.get_joints():
                if j.get_dof() == 1:
                    if j.get_name().endswith("driver_joint"):
                        joint_angles.append(0.04)
                    else:
                        joint_angles.append(0)
            self.robot.set_qpos(joint_angles)

    def close_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.068)
        for i in range(int(1/self.timestep)):  # wait for 1s
            self.env.step()
            self.env.render()

    def open_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.0)
        for i in range(int(1/self.timestep)):  # wait for 1s
            self.env.step()
            self.env.render()

    def clear_velocity_command(self):
        for joint in self.arm_joints:
            joint.set_drive_velocity_target(0)
    
    def follow_path(self, result):
        n_step = result['position'].shape[0]
        for i in range(n_step):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(7):
                self.arm_joints[j].set_drive_target(result['position'][i][j])
                self.arm_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            for i in range(int(self.control_timestep/self.timestep)):
                self.env.step()
                self.env.render()

    def wait_n_steps(self, n: int):
        self.clear_velocity_command()
        for i in range(n):
            self.balance_passive_force()
            self.env.step()
            self.env.render()
        self.robot.set_qf([0] * self.robot.dof)

    def balance_passive_force(self, including_object=True):
        passive_force_robot = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force_robot)
        if including_object:
            passive_force_object = self.env.object.compute_passive_force()
            self.env.object.set_qf(passive_force_object)

    def get_extrinsic_matrix(self):
        for l in self.robot.get_links():
            if l.get_name() == 'camera_base_color_optical_frame':
                return np.linalg.inv(l.get_pose().to_transformation_matrix())


######## real robot cotroller

# gripper controller
class Robotiq2FGripper_Listener:
    def __init__(self) -> None:
        self.gripper_is_ready = True

    def listen(self, input):
        if input.gACT == 1 and input.gOBJ > 0:
            self.gripper_is_ready = True
        else:
            self.gripper_is_ready = False

# arm controller
class RealRobotController(object):
    def __init__(self, ROS_rate=10, control_mode="imp_joint_pos", joint_stiffness=[300.0]*7, joint_damping=[20.0]*7) -> None:
        rospy.init_node('where2act_xmate3_robot_control_node')
        self.gripper_listener = Robotiq2FGripper_Listener()
        self.rate = rospy.Rate(ROS_rate) # 10hz
        self.control_mode = control_mode
        self.joint_stiffness = joint_stiffness
        self.joint_damping = joint_damping
        self._initialize_robotcontrol()
        self._initialize_gripper()

    def _initialize_robotcontrol(self):
        
        if self.control_mode == "imp_joint_pos":
            self.command_pub = rospy.Publisher('joint_control_command', JointControlCommand, queue_size=1)
        elif self.control_mode == "imp_ee_pose":
            self.command_pub = rospy.Publisher('cartesian_control_command', CartesianControlCommand, queue_size=1)
        elif self.control_mode == "pd_joint_delta_pos":
            self.command_pub = rospy.Publisher('joint_control_command', JointControlCommand, queue_size=1)
        else:
            raise NotImplementedError
    
    def _initialize_gripper(self):
        rospy.Subscriber("Robotiq2FGripperRobotInput", Robotiq2FGripper_robot_input, self.gripper_listener.listen)

        self.gripper_input = Robotiq2FGripper_robot_input()
        self.gripper_output = Robotiq2FGripper_robot_output()
        self.gripper_pub = rospy.Publisher('Robotiq2FGripperRobotOutput', Robotiq2FGripper_robot_output, queue_size=1)
        self.gripper_output.rACT = 1         #1: Active, 0: Not active
        self.gripper_output.rGTO = 1
        self.gripper_output.rATR = 0
        self.gripper_output.rPR = 150        #0~255: Placement
        self.gripper_output.rFR = 50         #0~255: Force
        self.gripper_output.rSP = 100        #0~255: Speed

        while self.gripper_listener.gripper_is_ready: self.gripper_pub.publish(self.gripper_output)
        self.gripper_output.rPR = 0        #Open Gripper
        while not self.gripper_listener.gripper_is_ready: pass
        while self.gripper_listener.gripper_is_ready: self.gripper_pub.publish(self.gripper_output)
    
    def get_realstate(self):
        self.real_state: ImpedanceRobotState = rospy.wait_for_message("impedance_robot_state", ImpedanceRobotState, timeout=10000)
        assert self.real_state is not None, 'no real robot state received in 10000s'
        return self.real_state
    
    def exec_gripper(self, gripper_rPR):
        self.gripper_output.rPR = gripper_rPR
        while not self.gripper_listener.gripper_is_ready: pass
        # while self.gripper_listener.gripper_is_ready: self.gripper_pub.publish(self.gripper_output)
        self.gripper_pub.publish(self.gripper_output)
        while not self.gripper_listener.gripper_is_ready: pass

    def exec_action(self, target, stiffness = None, damping = None): 
        '''
        target: (joint_pos): qpos[7], (ee_pose): mat(16)
        stiffness: (joint_): 7, (ee_): 6
        damping: (joint_): 7, (ee_): 6
        '''
        self.get_realstate()
        assert (stiffness is not None and damping is not None), "joint/ee stiffness and damping must be set!"
        if self.control_mode == "imp_ee_pose":
            # env.agent._robot.jacobian
            msg = CartesianControlCommand(target, stiffness, damping, False)
        elif self.control_mode == "imp_joint_pos": 
            msg = JointControlCommand(target, stiffness, damping, False)
            # print("++++ get robot arm control msg ++++")
        else:
            raise NotImplementedError
        self.command_pub.publish(msg)
        # print("++++ robot arm executation finished ++++")
        self.rate.sleep()

    def exec_trajs(self, plan_result): 
        '''
        plan_result: mplib planner result
        stiffness: (joint_): 7, (ee_): 6
        damping: (joint_): 7, (ee_): 6
        '''
        trajs = plan_result["position"]
        self.get_realstate()
        for i in tqdm(range(len(trajs))):
            if self.control_mode == "imp_joint_pos": 
                msg = JointControlCommand(trajs[i], self.joint_stiffness, self.joint_damping, False)
            else:
                raise NotImplementedError
            self.command_pub.publish(msg)
            self.rate.sleep()