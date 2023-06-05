import os
import sys
from argparse import ArgumentParser


base_path = "/data/laptop"
target_base_path = "/home/guest2/Documents/Sim2Real2/ditto/assets/urdf"


def createSynlink(src_cat, dst_cat, shape_id):
    target_path = os.path.join(target_base_path, dst_cat, shape_id)
    if os.path.exists(target_path):
        return
    src_dir_name = os.path.join(base_path, shape_id)
    os.symlink(src_dir_name, target_path)


if __name__ == '__main__':
    data_fns = ["/home/guest2/Documents/Sim2Real2/ditto/assets/stats/laptop_train.txt",
                "/home/guest2/Documents/Sim2Real2/ditto/assets/stats/laptop_val.txt"]

    for fn in data_fns:
        file = open(fn)
        shape_ids = [line.split(" ")[0] for line in file.readlines()]
        for shape_id in shape_ids:
            createSynlink("laptop", "Laptop", shape_id)
        file.close()

# import sapien
# import requests
#
# target_url = "https://sapien.ucsd.edu/user/refresh-token"
# headers = {
#     'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
#     'cookie': 'authentication=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IjIwMjMxMTY1QGJ1YWEuZWR1LmNuIiwidmVyc2lvbiI6MCwiaWF0IjoxNjgzNDYwODY1fQ.k0du1uD-6ihR-gvb9rvv_5Kcbd8Uc5HIpNZYSxCGCP8; authorization=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6IjIwMjMxMTY1QGJ1YWEuZWR1LmNuIiwiaXAiOiIxNzIuMjAuMC4xIiwicHJpdmlsZWdlIjowLCJpYXQiOjE2ODM0NjEwMzIsImV4cCI6MTY4MzU0NzQzMn0.nSF9gnIS4cz8-Kfbpk3RsHr0XJqM81gkejoqzYu-Fcs'
# }
# response = requests.get(target_url, headers=headers).json()
# token = response['token']
# print(token)
#
# for id in [9748, 10211, 10213, 10305, 10626]:
#     urdf_file = sapien.asset.download_partnet_mobility(id, token)
