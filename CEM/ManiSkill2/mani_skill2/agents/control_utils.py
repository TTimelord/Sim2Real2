import numpy as np

# from numba import jit


# @jit(nopython=True)
def nullspace_torques(
    mass_matrix, nullspace_matrix, initial_joint, joint_pos, joint_vel, joint_kp
):
    """
    For a robot with redundant DOF(s), a nullspace exists which is orthogonal to the remainder of the controllable
    subspace of the robot's joints. Therefore, an additional secondary objective that does not impact the original
    controller objective may attempt to be maintained using these nullspace torques.

    This utility function specifically calculates nullspace torques that attempt to maintain a given robot joint
    positions @initial_joint with zero velocity using proportinal gain @joint_kp

    :Note: @mass_matrix, @nullspace_matrix, @joint_pos, and @joint_vel should reflect the robot's state at the current
    timestep

    Args:
        mass_matrix (np.array): 2d array representing the mass matrix of the robot
        nullspace_matrix (np.array): 2d array representing the nullspace matrix of the robot
        initial_joint (np.array): Joint configuration to be used for calculating nullspace torques
        joint_pos (np.array): Current joint positions
        joint_vel (np.array): Current joint velocities
        joint_kp (np.array): Proportional control gain when calculating nullspace torques

    Returns:
          np.array: nullspace torques
    """

    # kv calculated below corresponds to critical damping
    joint_kv = np.sqrt(joint_kp) * 2

    # calculate desired torques based on gains and error
    pose_torques = np.dot(
        mass_matrix, (joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel)
    )

    # map desired torques to null subspace within joint torque actuator space
    nullspace_torques = np.dot(nullspace_matrix.transpose(), pose_torques)
    return nullspace_torques


# @jit(nopython=True)
def opspace_matrices(mass_matrix, J_full, J_pos, J_ori):
    """
    Calculates the relevant matrices used in the operational space control algorithm

    Args:
        mass_matrix (np.array): 2d array representing the mass matrix of the robot
        J_full (np.array): 2d array representing the full Jacobian matrix of the robot
        J_pos (np.array): 2d array representing the position components of the Jacobian matrix of the robot
        J_ori (np.array): 2d array representing the orientation components of the Jacobian matrix of the robot

    Returns:
        4-tuple:

            - (np.array): full lambda matrix (as 2d array)
            - (np.array): position components of lambda matrix (as 2d array)
            - (np.array): orientation components of lambda matrix (as 2d array)
            - (np.array): nullspace matrix (as 2d array)
    """
    mass_matrix_inv = np.linalg.inv(mass_matrix)

    # J M^-1 J^T
    lambda_full_inv = np.dot(np.dot(J_full, mass_matrix_inv), J_full.transpose())

    # Jx M^-1 Jx^T
    lambda_pos_inv = np.dot(np.dot(J_pos, mass_matrix_inv), J_pos.transpose())

    # Jr M^-1 Jr^T
    lambda_ori_inv = np.dot(np.dot(J_ori, mass_matrix_inv), J_ori.transpose())

    # take the inverses, but zero out small singular values for stability
    lambda_full = np.linalg.pinv(lambda_full_inv)
    lambda_pos = np.linalg.pinv(lambda_pos_inv)
    lambda_ori = np.linalg.pinv(lambda_ori_inv)

    # nullspace
    Jbar = np.dot(mass_matrix_inv, J_full.transpose()).dot(lambda_full)
    nullspace_matrix = np.eye(
        J_full.shape[-1], J_full.shape[-1], dtype=J_full.dtype
    ) - np.dot(Jbar, J_full)

    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


# @jit(nopython=True)
def orientation_error(desired, current):
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error


# @jit(nopython=True)
def inverse_se3_matrix(H: np.ndarray):
    H_inv = np.eye(4, dtype=H.dtype)
    H_inv[:3, :3] = H[:3, :3].T
    H_inv[:3, 3] = -H[:3, :3].T @ H[:3, 3]
    return H_inv
