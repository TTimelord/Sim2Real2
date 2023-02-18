import math
import os
from posixpath import join

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
from sapien.core.pysapien import Pose
from sapien.utils import Viewer

from mani_skill2.utils.urdf.urdf import URDF
from mani_skill2.utils.urdf.urdfpy_utils import *


def filePath(dirpath, originPath):
    return dirpath + "/" + originPath


def origin2pose(origin):
    pose = sapien.Pose.from_transformation_matrix(origin)
    return pose


def origin2pose_inertia(origin, inertia, scale):
    inertial2link = sapien.Pose.from_transformation_matrix(origin)
    inertial2link.set_p(inertial2link.p)
    p, q = np.linalg.eig(inertia)
    if np.linalg.det(q) < 0:
        q[:, 2] = -1 * q[:, 2]
    principle_moments_inertia = np.diagonal(
        np.matmul(np.matmul(np.linalg.inv(q), inertia), q)
    )
    print(principle_moments_inertia)
    inertia2inertial = sapien.Pose()
    # quad = np.array(list(Quaternion(matrix = q)))
    quad = t3d.quaternions.mat2quat(q)
    quad = quad / np.linalg.norm(quad)
    inertia2inertial.set_q(quad)
    inertia2link = inertial2link * inertia2inertial
    return inertia2link, principle_moments_inertia


def x_as_axis(joint_origin, joint_axis):
    # urdfpy gives joint_axis as unit vector
    if (
        math.isclose(joint_axis[0], 0.0)
        and math.isclose(joint_axis[1], 0.0)
        and math.isclose(joint_axis[2], 0.0)
    ):
        pose_in_parent = sapien.Pose()
        pose_in_child = sapien.Pose()
    elif math.isclose(joint_axis[0], 1.0):
        pose_in_parent = sapien.Pose.from_transformation_matrix(joint_origin)
        pose_in_child = sapien.Pose()
    else:
        i1 = joint_axis[0]
        i2 = joint_axis[1]
        i3 = joint_axis[2]
        if math.isclose(i3, 0.0):
            j1 = -i2
            j2 = i1
            j3 = 0.0
        else:
            j1 = 0.0
            j2 = 1 / math.sqrt(1 + (i2 / i3) ** 2)
            j3 = -i2 / i3 * j2
        i = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        j = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        k = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        i_prime = np.array([i1, i2, i3], dtype=np.float32)
        j_prime = np.array([j1, j2, j3], dtype=np.float32)
        k_prime = np.cross(i_prime, j_prime)
        rotate_trans = np.array(
            [
                [np.dot(i, i_prime), np.dot(i, j_prime), np.dot(i, k_prime), 0.0],
                [np.dot(j, i_prime), np.dot(j, j_prime), np.dot(j, k_prime), 0.0],
                [np.dot(k, i_prime), np.dot(k, j_prime), np.dot(k, k_prime), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pose_in_parent = sapien.Pose.from_transformation_matrix(
            np.matmul(joint_origin, rotate_trans)
        )
        pose_in_child = sapien.Pose.from_transformation_matrix(rotate_trans)
    return pose_in_parent, pose_in_child


def load_urdf_into_scene(urdf_file, scene):

    robot = URDF.load(urdf_file)  # Get robot model from URDF file
    dirpath = os.path.dirname(urdf_file)

    links = robot.links  # All robot links
    joints = robot.joints  # All robot joints

    name2link = dict(
        zip([link.name for link in links], links)
    )  # Dictionary from all link names to links

    link2child_joints = {}  # Dictionary from any link to its child joints
    for joint in joints:
        if joint.parent not in link2child_joints:
            link2child_joints[joint.parent] = [joint]
        else:
            link2child_joints[joint.parent].append(joint)

    link2parent_joint = {}  # Dictionary from any link to its parent joint
    for joint in joints:
        link2parent_joint[joint.child] = joint

    articulation_roots = [robot.base_link.name]  # Roots for all articulation builders
    for joint in joints:
        if joint.joint_type == "floating":
            articulation_roots.append(joint.child)

    for root in articulation_roots:
        if root in link2parent_joint:
            temp_joint = link2parent_joint[root]
            bias = temp_joint.origin
            temp_link = temp_joint.parent
            while temp_link in link2parent_joint:
                temp_joint = link2parent_joint[temp_link]
                bias = np.matmul(temp_joint.origin, bias)
                temp_link = temp_joint.parent

            for visual in name2link[
                root
            ].visuals:  # Articulation object roots now have different origins since it is detached from the former urdf links
                visual.origin = np.matmul(bias, visual.origin)
            for collision in name2link[root].collisions:
                collision.origin = np.matmul(bias, collision.origin)
            if root in link2child_joints:
                for joint in link2child_joints[root]:
                    if joint.joint_type != "floating":
                        joint.origin = np.matmul(bias, joint.origin)
                        # joint.axis = np.matmul(bias[0:3, 0:3], joint.axis)

        links = []  # All robot links in this articulation object
        joints = []  # All robot joints in this articulation object

        temp_links = [root]
        while not temp_links == []:
            new_links = []
            for link in temp_links:
                links.append(link)
                if link in link2child_joints:
                    child_joints = link2child_joints[link]
                    for joint in child_joints:
                        if not joint.child in articulation_roots:
                            joints.append(joint)
                            new_links.append(joint.child)
            temp_links = new_links

        if len(links) == 1:
            builder = scene.create_actor_builder()
            link_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            for visual in name2link[root].visuals:
                if visual.geometry is not None:
                    if visual.geometry.mesh is not None:
                        if visual.geometry.mesh.scale is not None:
                            link_scale = visual.geometry.mesh.scale
                        builder.add_visual_from_file(
                            filename=filePath(dirpath, visual.geometry.mesh.filename),
                            pose=origin2pose(visual.origin),
                            scale=link_scale,
                        )
            link_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            for collision in name2link[root].collisions:
                if collision.geometry is not None:
                    if collision.geometry.mesh is not None:
                        if collision.geometry.mesh.scale is not None:
                            link_scale = collision.geometry.mesh.scale
                        builder.add_collision_from_file(
                            filename=filePath(
                                dirpath, collision.geometry.mesh.filename
                            ),
                            pose=origin2pose(collision.origin),
                            scale=link_scale,
                        )
            inertial = name2link[root].inertial
            if inertial is not None:
                inertia2link, principle_moments_inertia = origin2pose_inertia(
                    inertial.origin, inertial.inertia, link_scale
                )
                builder.set_mass_and_inertia(
                    inertial.mass, inertia2link, principle_moments_inertia
                )
            actor = builder.build(name=root)
            actor.lock_motion(True, True, True, True, True, True)
        else:
            builder = scene.create_articulation_builder()
            sapien_links = []  # All links have been added to the builder
            link_builders = []  # All sapien link builders
            name2builder = (
                {}
            )  # Dictionary from link names to their link builders' indexes in the list link_builders

            while len(sapien_links) < len(links):
                for joint in joints:
                    if joint.parent == root and joint.parent not in sapien_links:
                        # Must be the root link for all links
                        link_name = joint.parent
                        link_builders.append(builder.create_link_builder())
                        last = len(link_builders) - 1
                        link_builders[last].set_name(link_name)
                        link_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                        for visual in name2link[link_name].visuals:
                            if visual.geometry is not None:
                                if visual.geometry.mesh is not None:
                                    if visual.geometry.mesh.scale is not None:
                                        link_scale = visual.geometry.mesh.scale
                                    link_builders[last].add_visual_from_file(
                                        filename=filePath(
                                            dirpath, visual.geometry.mesh.filename
                                        ),
                                        pose=origin2pose(visual.origin),
                                        scale=link_scale,
                                    )
                        for collision in name2link[link_name].collisions:
                            if collision.geometry is not None:
                                if collision.geometry.mesh is not None:
                                    if collision.geometry.mesh.scale is not None:
                                        link_scale = collision.geometry.mesh.scale
                                    link_builders[last].add_collision_from_file(
                                        filename=filePath(
                                            dirpath, collision.geometry.mesh.filename
                                        ),
                                        pose=origin2pose(collision.origin),
                                        scale=link_scale,
                                    )
                        inertial = name2link[link_name].inertial
                        if inertial is not None:
                            print(link_name)
                            print(inertial.origin)
                            (
                                inertia2link,
                                principle_moments_inertia,
                            ) = origin2pose_inertia(
                                inertial.origin, inertial.inertia, link_scale
                            )
                            link_builders[last].set_mass_and_inertia(
                                inertial.mass, inertia2link, principle_moments_inertia
                            )
                        sapien_links.append(link_name)
                        name2builder[link_name] = len(sapien_links) - 1
                    elif (
                        joint.child not in sapien_links and joint.parent in sapien_links
                    ):
                        # Joint that links a novel link to a link already in builders
                        link_name = joint.child
                        joint_name = joint.name
                        link_builders.append(
                            builder.create_link_builder(
                                link_builders[name2builder[joint.parent]]
                            )
                        )
                        last = len(link_builders) - 1
                        link_builders[last].set_name(link_name)
                        link_builders[last].set_joint_name(joint_name)
                        link_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                        for visual in name2link[link_name].visuals:
                            if visual.geometry is not None:
                                if visual.geometry.mesh is not None:
                                    if visual.geometry.mesh.scale is not None:
                                        link_scale = visual.geometry.mesh.scale
                                    link_builders[last].add_visual_from_file(
                                        filename=filePath(
                                            dirpath, visual.geometry.mesh.filename
                                        ),
                                        pose=origin2pose(visual.origin),
                                        scale=link_scale,
                                    )
                        for collision in name2link[link_name].collisions:
                            if collision.geometry is not None:
                                if collision.geometry.mesh is not None:
                                    if collision.geometry.mesh.scale is not None:
                                        link_scale = collision.geometry.mesh.scale
                                    link_builders[last].add_collision_from_file(
                                        filename=filePath(
                                            dirpath, collision.geometry.mesh.filename
                                        ),
                                        pose=origin2pose(collision.origin),
                                        scale=link_scale,
                                    )
                        inertial = name2link[link_name].inertial
                        if inertial is not None:
                            print(link_name)
                            print(inertial.origin)
                            (
                                inertia2link,
                                principle_moments_inertia,
                            ) = origin2pose_inertia(
                                inertial.origin, inertial.inertia, link_scale
                            )
                            link_builders[last].set_mass_and_inertia(
                                inertial.mass, inertia2link, principle_moments_inertia
                            )
                        pip, pic = x_as_axis(joint.origin, joint.axis)
                        if joint.limit is not None:
                            joint_limit = [[joint.limit.lower, joint.limit.upper]]
                        else:
                            joint_limit = []
                        link_builders[last].set_joint_properties(
                            joint_type=joint.joint_type,
                            limits=joint_limit,
                            pose_in_parent=pip,
                            pose_in_child=pic,
                        )
                        sapien_links.append(link_name)
                        name2builder[link_name] = len(sapien_links) - 1

            sapienRobot = builder.build(fix_root_link=True)
            sapienRobot.set_name(root)
        if root in link2parent_joint:
            bias = np.linalg.inv(
                bias
            )  # These origin changes are only valid in the building process of this articulation object
            for visual in name2link[root].visuals:
                visual.origin = np.matmul(bias, visual.origin)
            for collision in name2link[root].collisions:
                collision.origin = np.matmul(bias, collision.origin)
            if root in link2child_joints:
                for joint in link2child_joints[root]:
                    if joint.joint_type != "floating":
                        joint.origin = np.matmul(bias, joint.origin)
                        joint.axis = np.matmul(bias[0:3, 0:3], joint.axis)
