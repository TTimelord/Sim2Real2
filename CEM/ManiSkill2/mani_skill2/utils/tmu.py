import os
import os.path as osp

import gym
import numpy as np

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.io import load_json

OCRTOC_DIR = os.path.join(ASSET_DIR, "ocrtoc")
DEFAULT_MODEL_JSON = os.path.join(OCRTOC_DIR, "models.json")


def register_gym_env_for_OCRTOC_models(name, **kwargs):
    def _register(cls):
        entry_point = "{}:{}".format(cls.__module__, cls.__name__)
        all_valid_models = list(load_json(DEFAULT_MODEL_JSON).keys())
        for model_id in all_valid_models:
            gym.register(
                id=name.replace("@", model_id),
                entry_point=entry_point,
                kwargs={
                    "model_ids": [model_id],
                },
                **kwargs,
            )
        return cls

    return _register


def register_gym_env_for_tmu(name, max_episode_steps=200, **kwargs):
    def _register(cls):
        entry_point = "{}:{}".format(cls.__module__, cls.__name__)
        for obs_mode in ["state", "rgbd", "depth", "pointcloud"]:
            tag = obs_mode
            gym.register(
                id=name.replace("@", tag),
                entry_point=entry_point,
                kwargs={
                    "obs_mode": obs_mode,
                    "reward_mode": "dense",
                },
                max_episode_steps=max_episode_steps,
                **kwargs,
            )
        return cls

    return _register


import cv2


def show_image_by_opencv(rgb_img):
    cv2.imshow("image", rgb_img[:, :, ::-1])
    cv2.waitKey(1)


def show_images_by_opencv(imgs):
    large_img = np.concatenate(imgs, axis=1)
    show_image_by_opencv(large_img)


def show_images_from_obs_dict_by_opencv(obs):
    show_images_by_opencv([obs[x]["rgb"] for x in obs if "rgb" in obs[x]])


from mani_skill2.utils.visualization.misc import normalize_depth


def save_images_from_obs_dict(obs, dir):
    os.makedirs(dir, exist_ok=True)
    for camear_name in obs:
        if "rgb" in obs[camear_name]:
            rgb = obs[camear_name]["rgb"][:, :, ::-1]
            assert rgb.dtype in [np.uint8, np.float32]
            if rgb.dtype == np.float32:
                rgb = (rgb * 255).astype(np.uint8)
            cv2.imwrite(osp.join(dir, camear_name + "_rgb.png"), rgb)

            depth = obs[camear_name]["depth"]

            depth = normalize_depth(depth, max_depth=10)
            depth = np.clip(depth * 255, 0, 255).astype(np.uint8)
            depth = np.repeat(depth, 3, axis=-1)

            cv2.imwrite(osp.join(dir, camear_name + "_depth.png"), depth)
