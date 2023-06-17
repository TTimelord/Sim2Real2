import sapien.core as sapien
import mplib
import numpy as np
from sapien.utils.viewer import Viewer

class PlanningDemo():
    def __init__(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 240.0)
        self.scene.add_ground(0)
        physical_material = self.scene.create_physical_material(1, 1, 0.0)
        self.scene.default_physical_material = physical_material


        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
        self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)

        # Robot
        # Load URDF
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.Articulation = loader.load("/data/jtr/where2act_allegro_sim/arm_hand.urdf")
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

        # Set initial joint positions
        arm_zero_qpos = [0,0,0,0,0,0,0]
        gripper_init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        init_qpos = arm_zero_qpos + gripper_init_qpos
        self.robot.set_qpos(init_qpos)

        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)
 #       print(len(self.active_joints))
 

        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.object: sapien.Articulation = loader.load("/data/jtr/simulation/44817/drawer1.urdf")
        self.object.set_root_pose(sapien.Pose([1, 0, 0.5], [1, 0, 0, 0]))
#        print(object.get_qpos())
        init_qpos_1 = [0.1, 0.1, 0.1, 0.1]
        self.object.set_qpos(init_qpos_1)
        self.active_joints_1 = self.object.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=200)

        self.setup_planner()
    
    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.planner = mplib.Planner(
            urdf="/data/jtr/where2act_allegro_sim/arm_hand.urdf",
            srdf="/data/jtr/where2act_allegro_sim/arm_hand.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="palm_link",
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7))

    def follow_path(self, result):
        n_step = result['position'].shape[0]
        for i in range(n_step):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            for j in range(7):
                self.active_joints[j].set_drive_target(result['position'][i][j])
                self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def open_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0.4)
        for i in range(100): 
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def close_gripper(self):
        for joint in self.active_joints[-2:]:
            joint.set_drive_target(0)
        for i in range(100):  
            qf = self.robot.compute_passive_force(
                gravity=True, 
                coriolis_and_centrifugal=True)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                self.viewer.render()

    def move_to_pose_with_RRTConnect(self, pose):
        result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1/250)
        if result['status'] != "Success":
            print(result['status'])
            return -1
        self.follow_path(result)
        return 0

    def move_to_pose_with_screw(self, pose):
        result = self.planner.plan_screw(pose, self.robot.get_qpos(), time_step=1/250)
        if result['status'] != "Success":
            result = self.planner.plan(pose, self.robot.get_qpos(), time_step=1/250)
            if result['status'] != "Success":
                print(result['status'])
                return -1 
        self.follow_path(result)
#        print(result)

        return 0
    
    def move_to_pose(self, pose, with_screw):
        if with_screw:
            return self.move_to_pose_with_screw(pose)
        else:
            return self.move_to_pose_with_RRT(pose)

    def demo(self, with_screw = True):
        pose = [0.36,0,0.65,1,0,0,0]
        self.move_to_pose(pose, with_screw)
        pose[0] +=0.1
        self.move_to_pose(pose, with_screw)

if __name__ == '__main__':
    demo = PlanningDemo()
    demo.demo()
