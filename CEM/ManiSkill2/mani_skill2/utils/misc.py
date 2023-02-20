from collections import OrderedDict
from typing import Dict

import numpy as np
import torch


def sample_from_tuple_or_scalar(rng, x):
    if isinstance(x, tuple):
        return rng.uniform(low=x[0], high=x[1])
    else:
        return x


import pathlib

import yaml


def get_model_ids_from_yaml(yaml_file_path):
    path = pathlib.Path(yaml_file_path).resolve()
    with path.open("r") as f:
        raw_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    return list(raw_yaml.keys())


def get_raw_yaml(yaml_file_path):
    path = pathlib.Path(yaml_file_path).resolve()
    with path.open("r") as f:
        raw_yaml = yaml.load(f, Loader=yaml.SafeLoader)
    return raw_yaml


def get_actor_state(actor):
    """
    returns actor state with shape (13, )
    actor_state[:3] = pose p
    actor_state[3:7] = pose q
    actor_state[7:10] = velocity
    actor_state[10:13] = angular velocity
    """
    pose = actor.get_pose()

    p = pose.p  # (3, )
    q = pose.q  # (4, )
    vel = actor.get_velocity()  # (3, )
    ang_vel = actor.get_angular_velocity()  # (3, )

    return np.concatenate([p, q, vel, ang_vel], axis=0)


def get_articulation_state(art):
    root_link = art.get_links()[0]
    base_pose = root_link.get_pose()
    base_vel = root_link.get_velocity()
    base_ang_vel = root_link.get_angular_velocity()
    qpos = art.get_qpos()
    qvel = art.get_qvel()
    return base_pose.p, base_pose.q, base_vel, base_ang_vel, qpos, qvel


def get_pad_articulation_state(art, max_dof):
    base_pos, base_quat, base_vel, base_ang_vel, qpos, qvel = get_articulation_state(
        art
    )
    k = len(qpos)
    pad_obj_internal_state = np.zeros(2 * max_dof)
    pad_obj_internal_state[:k] = qpos
    pad_obj_internal_state[max_dof : max_dof + k] = qvel
    return np.concatenate(
        [base_pos, base_quat, base_vel, base_ang_vel, pad_obj_internal_state]
    )


def print_dict(d: Dict):
    if not isinstance(d, Dict):
        if isinstance(d, np.ndarray):
            return f"{d.shape} {np.array2string(d, precision=3)}"
        return str(d)
    s = ""
    for k, v in d.items():
        if isinstance(v, dict) or isinstance(v, OrderedDict):
            s += f"[{k}: "
            s += "{"
            s += print_dict(v)
            s += "}]"
        elif isinstance(v, np.ndarray):
            if len(v.shape) > 1:
                s += f"{k}: {v.shape}; "
            else:
                s += f"{k}: {v.shape} {np.array2string(v, precision=3)}; "
        elif isinstance(v, torch.Tensor):
            if len(v.shape) > 1:
                s += f"{k}: {v.shape}; "
            else:
                s += f"{k}: {v.shape} {np.array2string(v.cpu().numpy(), precision=3)}; "
        else:
            s += f"{k}: {v}; "
    return s
