from lib2to3.pgen2.token import RPAR
import numpy as np

contact_force = []
joint_axis = np.array([1, 0, 0])
for i in range(5):
    contact_force.append(np.random.rand(3,))

contact_force.append(np.array([0, 0, 0]))

# print(contact_force)
contact_force = np.vstack(contact_force)
# print(contact_force)

norm = np.linalg.norm(contact_force, axis=1)
print(contact_force)
print(norm)
contact_force = contact_force[norm>0]
norm = norm[norm>0]
norm_repeat = np.repeat(norm.T, 3).reshape(contact_force.shape)
# print(norm)
contact_force_norm = contact_force / norm_repeat

# print(contact_force_norm)
dot_product = np.dot(contact_force_norm, joint_axis) # arccos rad
print(dot_product)
contact_direction_err = np.sum(norm * (np.ones_like(dot_product) - np.abs(dot_product)))
print(contact_direction_err)
