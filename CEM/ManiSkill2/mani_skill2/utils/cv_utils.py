import numpy as np


def depth2pts_np(
    depth_map: np.ndarray,
    cam_intrinsic: np.ndarray,
    cam_extrinsic: np.ndarray = np.eye(4),
) -> np.ndarray:
    assert (len(depth_map.shape) == 2) or (
        len(depth_map.shape) == 3 and depth_map.shape[2] == 1
    )
    assert cam_intrinsic.shape == (3, 3)
    assert cam_extrinsic.shape == (4, 4)
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))  # (3, N)

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()  # (N, 3)
    return world_points


def get_pixel_grids_np(height: int, width: int):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(np.float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid
