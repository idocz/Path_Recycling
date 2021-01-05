import numpy as np
from scipy.io import loadmat, savemat
from os.path import join

field_path = join("data","large_rico.mat")
output_path = join("data","rico2.mat")

large_field = loadmat(field_path)["vol"]

x_range = [20, 200]
y_range = [800, 1000]


# x_range = [800, 1020]
# y_range = [150, 400]


cloud = large_field[y_range[0]:y_range[1], x_range[0]:x_range[1], :]
cloud_dict = {"vol":cloud}
savemat(output_path, cloud_dict)
# np.save(output_path, cloud)