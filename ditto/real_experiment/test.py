from markupsafe import re
import numpy as np
import os
# x1 = [-0.2724604,0.00562957,0.00150904]

# x2 = [-0.7755829,-0.6301434,-0.00089072]
# y = np.dot(x1,x2)
# z = np.arccos(y)
# print(y)
# print(z)
root_path = os.getcwd()
data_model = 'faucet'
realdata_path = os.path.join(root_path,'notebooks/real_datasets',data_model)
axis_ori_path = os.path.join(realdata_path,'axis_ori_err.txt')
config_err_path = os.path.join(realdata_path,'config_err.txt')
type_val_path = os.path.join(realdata_path,'type_val.txt') 
axis_ori_err = np.loadtxt(axis_ori_path,dtype=np.float32, delimiter=',')
config_err = np.loadtxt(config_err_path,dtype=np.float32, delimiter=',')
type_val = np.loadtxt(type_val_path,dtype=np.float32, delimiter=',')

result_dict = {}

axis_ori_err_average = np.mean(axis_ori_err,axis = 0)
config_err_average = np.mean(config_err, axis = 0)
if data_model=='drawer':
    type_val_average = np.mean(type_val, axis = 0)
else:
    type_val_average = np.mean(type_val+1, axis = 0)

print(axis_ori_err_average)
print(config_err_average)
print(type_val_average)
result_dict["articulation_all"] = {
    "prismatic": {
        "axis_orientation_average": axis_ori_err_average,
        "config_err_average": config_err_average,
    },
    "revolute": None,
    "joint_type_average": {"accuracy": type_val_average},
}

result_showmodel_path = f"{realdata_path}/result_showmodel.yaml"
f = open(result_showmodel_path, 'a')
print(result_dict,file=f)
# print('**********************************************************************************',file=f)
f.close()