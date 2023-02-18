import os
import subprocess
from tqdm import tqdm
from glob import glob

#---------------------需要改动的参数为name/classes/sub_dataset----------------------
data_path = '/home/rvsa/where2act_ws/Ditto/assets/urdf/Laptop'

def fixmesh(input_path,output_path):
    try:
        completed = subprocess.run(["/home/rvsa/where2act_ws/ManifoldPlus/build/manifold", "--input", input_path, "--output", output_path, "--depth", '7'], timeout=60, check=True, capture_output=True)
    except:
        print("{} failed to run",format(input_path))
        return False
    return True

def fix_mesh_in_folder(data_path):
    for shape_id in tqdm(os.listdir(data_path)):
        ori_mesh_path = os.path.join(data_path, shape_id, 'part_mesh')
        out_mesh_path = os.path.join(data_path, shape_id, 'part_mesh_fixed')
        if not os.path.exists(out_mesh_path):
            os.makedirs(out_mesh_path)
        for mesh in os.listdir(ori_mesh_path):
            if mesh[-4:] == '.obj':
                input_path = os.path.join(ori_mesh_path,mesh)
                output_path = os.path.join(out_mesh_path,mesh)
                fixmesh(input_path,output_path)
                # print(mesh)

fix_mesh_in_folder(data_path)
