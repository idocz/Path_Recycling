import numpy as np
from scipy.io import loadmat
from os.path import join
from scipy.ndimage import zoom


beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
new_array = zoom(beta_cloud, (0.5,0.5,0.5))
print(beta_cloud.shape)
print(new_array.shape)
new_array = zoom(new_array, (2,37/18,2))
print(new_array.shape)