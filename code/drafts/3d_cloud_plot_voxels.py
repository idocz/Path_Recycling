import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from utils import construct_beta

shape = 3
beta = construct_beta(shape)
beta_norm = (beta - beta.min())/(beta.max() - beta.min())
beta_colors = 1 - beta_norm

x = np.linspace(0, 1, shape+1)
y = np.linspace(0, 1, shape+1)
z = np.linspace(0, 1,shape+1)
xx, yy, zz = np.meshgrid(x, y, z)
# set the colors of each object

colors = cm.gray(beta_colors)

# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(xx, yy, zz, beta>0, alpha=0.7, facecolors=colors)

plt.show()