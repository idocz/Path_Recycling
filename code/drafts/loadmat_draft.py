from scipy.io import loadmat
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from utils import *

name = "clouds_dist.mat"
beta_cloud = loadmat(join("code","data",name))["beta"]
beta_cloud = remove_zero_planes(beta_cloud)
beta_cloud = resize_to_cubic_shape(beta_cloud)
beta_cloud = downsample_3D(beta_cloud,(16,16,16))
plt.hist(beta_cloud[beta_cloud>0.01].reshape(-1))
plt.show()
exit()
shape = beta_cloud.shape
print(shape)
fig = plt.figure()
ax = plt.axes(projection='3d')
if beta_cloud.max() != beta_cloud.min():
    beta_norm = (beta_cloud - beta_cloud.min()) / (beta_cloud.max() - beta_cloud.min())
else:
    beta_norm = 0.5 * np.ones_like(beta_cloud)
beta_colors = 1 - beta_norm
# set the colors of each object
x = np.linspace(0, 1, shape[0] + 1)
y = np.linspace(0, 1, shape[1] + 1)
z = np.linspace(0, 1, shape[2] + 1)
xx, yy, zz = np.meshgrid(x, y, z)
colors = cm.gray(beta_colors+0.5)
ax.voxels(xx, yy, zz, beta_cloud > 0.1, alpha=0.7, facecolors=colors, edgecolors='gray')
plt.show()