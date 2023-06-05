import argparse
import io
import os
import zipfile

import requests

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.io import load_json


def download_partnet_mobility(model_id, directory=None):
    url = "https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/{}.zip".format(
        model_id
    )
    if not directory:
        directory = os.environ.get("PARTNET_MOBILITY_DATASET")
        if not directory:
            directory = str(ASSET_DIR / "partnet_mobility" / "dataset")

    urdf_file = os.path.join(directory, str(model_id), "mobility.urdf")

    print(f"{model_id} already exists.")
    # return if file exists
    if os.path.exists(urdf_file):
        return urdf_file

    print(f"Downloading {model_id} to {directory}...")
    # download file
    r = requests.get(url, stream=True)
    if not r.ok:
        raise Exception(
            "Download PartNet-Mobility failed. "
            "Please check your token and IP address."
            "Also make sure sure the model id is valid"
        )

    z = zipfile.ZipFile(io.BytesIO(r.content))

    os.makedirs(directory, exist_ok=True)
    z.extractall(directory)
    return urdf_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name

    if name == "faucet_v0":
        model_db = load_json(ASSET_DIR / "partnet_mobility/meta/info_faucet_v0.json")
        for model_id in model_db.keys():
            download_partnet_mobility(model_id)
    else:
        raise NotImplementedError(name)


if __name__ == "__main__":
    main()
