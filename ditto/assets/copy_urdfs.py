import os
import sys
from argparse import ArgumentParser


base_path = "/home/guest2/Documents/Sim2Real2/ditto/data/urdfs/Shape2Motion"
target_base_path = "/home/guest2/Documents/Sim2Real2/ditto/assets/urdf"


def createSynlink(src_cat, dst_cat, shape_id):
    target_path = os.path.join(target_base_path, dst_cat, shape_id)
    if os.path.exists(target_path):
        return
    src_dir_name = os.path.join(base_path, src_cat)
    for task in os.listdir(src_dir_name):
        src_task_dir_name = os.path.join(src_dir_name, task)
        for id in os.listdir(src_task_dir_name):
            if id == shape_id:
                src_task_id_dir_name = os.path.join(src_task_dir_name, id)
                os.symlink(src_task_id_dir_name, target_path)
                return

    raise ValueError(f"Invalid shape_id {shape_id} in category {src_cat}")


if __name__ == '__main__':
    data_fns = ["/home/guest2/Documents/Sim2Real2/ditto/assets/stats/drawer_train.txt",
                "/home/guest2/Documents/Sim2Real2/ditto/assets/stats/drawer_val.txt",
                "/home/guest2/Documents/Sim2Real2/ditto/assets/stats/faucet_train.txt",
                "/home/guest2/Documents/Sim2Real2/ditto/assets/stats/faucet_val.txt"]

    for fn in data_fns:
        file = open(fn)
        shape_ids = [line.split(" ")[0] for line in file.readlines()]
        if fn.count("drawer") > 0:
            for shape_id in shape_ids:
                createSynlink("cabinet", "Drawer", shape_id)
        else:
            for shape_id in shape_ids:
                createSynlink("faucet", "Faucet", shape_id)
        file.close()
