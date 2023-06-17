import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
from PIL import Image



engine = sapien.Engine()  #create a physical simulation engine
renderer = sapien.SapienRenderer() # create a renderer
engine.set_renderer(renderer)

balance_passive_force = True
fix_root_link = True
scene_config = sapien.SceneConfig()
scene = engine.create_scene(scene_config)
scene = engine.create_scene()
scene.set_timestep(1/240.0)
scene.add_ground(0)

scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

viewer = Viewer(renderer)
viewer.set_scene(scene)
viewer.set_camera_xyz(x=-0.75, y=0.5, z=1)
viewer.set_camera_rpy(r=0, p=-0.3, y=0)

#load robot

loader: sapien.URDFLoader = scene.create_urdf_loader()
loader.fix_root_link = fix_root_link
robot: sapien.Articulation = loader.load("/data/jtr/simulation/arm_hand.urdf")
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
#print(robot.get_qpos())
#init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#set joint positions for PID
arm_zero_qpos = [0,0,0,0,0,0,0]
gripper_init_qpos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
init_qpos = arm_zero_qpos + gripper_init_qpos
robot.set_qpos(init_qpos)

arm_target_qpos = [2.71, 0.84, 0.0, 0.75, 2.62, 2.28, 2.88]
target_qpos = arm_target_qpos + gripper_init_qpos

#def your own PID
class SimplePID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.p = kp
        self.i = ki
        self.d = kd

        self._cp = 0
        self._ci = 0
        self._cd = 0

        self._last_error = 0

    def compute(self, current_error, dt):
        self._cp = current_error
        self._ci += current_error * dt
        self._cd = (current_error - self._last_error) / dt
        self._last_error = current_error
        signal = (self.p * self._cp) + \
            (self.i * self._ci) + (self.d * self._cd)
        return signal
def pid_forward(pids: list,
                target_pos: np.ndarray,
                current_pos: np.ndarray,
                dt: float) -> np.ndarray:
    errors = target_pos - current_pos
    qf = [pid.compute(error, dt) for pid, error in zip(pids, errors)]
    return np.array(qf)

#action controller
#how_to_control = 0,use internal drive; =1,use external drive
how_to_control = 0;
if how_to_control == 0:
    active_joints = robot.get_active_joints()
    for joint_idx, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=20, damping=5)
        joint.set_drive_target(target_qpos[joint_idx])
if how_to_control == 1:
    active_joints = robot.get_active_joints()
    pids = []
    pid_parameters = [
        (40, 5, 2), (40, 5, 2), (40, 5, 2), (20, 5.0, 2),
        (5, 0.8, 2), (5, 0.8, 2), (5, 0.8, 0.4),
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
        (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
        (0, 0, 0)
    ]
    for i, joint in enumerate(active_joints):
        pids.append(SimplePID(*pid_parameters[i]))


#load drawer

loader: sapien.URDFLoader = scene.create_urdf_loader()
loader.fix_root_link = fix_root_link
object: sapien.Articulation = loader.load("/data/jtr/simulation/44817/drawer1.urdf")
object.set_root_pose(sapien.Pose([1, 0, 0.5], [1, 0, 0, 0]))
print(object.get_qpos())
init_qpos_1 = [0, 0, 0, 0]
object.set_qpos(init_qpos_1)

#set camera
#random_position = False
#fixed_position = True

near, far = 0.1, 100
width = height = 448
camera_mount_actor = scene.create_actor_builder().build_kinematic()
camera = scene.add_mounted_camera(

    name = "camera1",
    width = width,
    height = height,
    fovy = np.deg2rad(100),
    fovx = np.deg2rad(100),
    near = near,
    far = far,
    actor = camera_mount_actor,
    pose = sapien.Pose(),
)
#fixed position:

            #theta = -np.pi/10
            #theta = -np.pi/8
theta = np.pi
phi = np.pi/10
'''
pos = np.array([dist*np.cos(phi)*np.cos(theta), \
                dist*np.cos(phi)*np.sin(theta), \
                dist*np.sin(phi)])
#print(np.linalg.norm(pos))
forward = -pos / np.linalg.norm(pos)
left = np.cross([0, 0, 1], forward)
left = left / np.linalg.norm(left)
up = np.cross(forward, left)
mat44 = np.eye(4)
mat44[:3, :3] = np.vstack([forward, left, up]).T
mat44[:3, 3] = pos      # mat44 is cam2world
mat44[0, 3] += 0.0
camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
#set the position based on eye_hand
'''

r_x = -0.0374
b_y = 0.9776
a_z = -1.2197

pos = [0.2192, -0.43277, 0.7225]
left = np.array([-np.cos(a_z)*np.cos(r_x)-np.sin(a_z)*np.sin(b_y)*np.sin(r_x), \
                 -np.sin(a_z)*np.cos(r_x)+np.cos(a_z)*np.sin(b_y)*np.sin(r_x), \
                 np.sin(r_x)*np.cos(b_y)])
forward = np.array([-np.sin(a_z)*np.cos(b_y), \
                    +np.cos(a_z)*np.cos(b_y), \
                    -np.sin(b_y)])
up = np.cross(forward, left)

mat44 = np.eye(4)
mat44[:3, :3] = np.vstack([forward, left, up]).T
mat44[:3, 3] = pos      # mat44 is cam2world
'''
mat44 = [[0.19226438, 0.928, 0.320, -0.4327], [-0.5249, 0.3728, -0.7652, -0.2192],
         [-0.8292, -0.0209, 0.5586, 0.7225], [0, 0, 0, 1]]
'''
camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
print(mat44)

#return rgb
scene.step()
scene.update_render()
camera.take_picture()
rgba = camera.get_float_texture('Color')
rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
rgba_pil = Image.fromarray(rgba_img)
rgba_pil.save('color1.png')

while not viewer.closed:
    for _ in range(4):  # render every 4 steps
        if balance_passive_force:
            qf = robot.compute_passive_force(
                gravity=False,
                coriolis_and_centrifugal=True,
            )
        if how_to_control == 1:
            pid_qf = pid_forward(
                pids,
                target_qpos,
                robot.get_qpos(),
                scene.get_timestep()
            )
            qf += pid_qf
        robot.set_qf(qf)


# robot.get_qf()
        scene.step()
    scene.update_render()
    viewer.render()

#print(active_joints)
#print(len(active_joints))