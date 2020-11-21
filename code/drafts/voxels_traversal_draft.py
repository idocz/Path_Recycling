import numpy as np
from classes.grid import *
from classes.volume import *
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

ticks = np.linspace(0, 1, 5+1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.grid()


bbox = np.array([[0,1],
                 [0,1],
                 [0,1]])

shape = np.array((5,5,5))
grid = Grid(bbox, shape)
beta_cloud = np.ones(shape.tolist())
beta_air = 1
volume = Volume(grid, beta_cloud, beta_air)

start = np.array([0.47, 0.22, 0.12])
current_voxel = grid.get_voxel_of_point(start)
theta = np.pi/4
phi = np.pi/4
direction = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
tau_rand = -np.log(1-np.random.rand())
tau_rand = 5


path = [start]
current_point, in_medium, seg_voxels, seg_lengths, seg_size, beta = volume.voxel_traversal_algorithm(start, current_voxel, direction, tau_rand)
# current_point = start
# for i in range(5):
#     t, current_voxel, current_point, min_ind = grid.travel_to_voxels_border(current_point, direction, current_voxel)
#     print(t, current_voxel, current_point, min_ind)
#     path.append(current_point)

path.append(current_point)
path = np.vstack(path)
path_T = path.T
ax.scatter(*path_T)
ax.plot(*path_T)
ax.quiver(*start, *direction, length=0.3)

print(seg_voxels)
print(seg_lengths)
print(in_medium)
print(path)

# 2d planes
fig = plt.figure()
for i in range(2):
    ax_2d = plt.subplot(1,2,i+1)
    if i == 0:
        inds = [0, 1]
        x_lable = "x"
        y_label = "y"
    elif i == 1:
        inds = [0, 2]
        x_lable = "x"
        y_label = "z"
    else:
        inds = [1, 2]
        x_lable = "y"
        y_label = "z"
    ax_2d.set_xticks(ticks)
    ax_2d.set_yticks(ticks)
    ax_2d.set_xlim(0,1)
    ax_2d.set_ylim(0,1)
    ax_2d.scatter(*path_T[inds])
    ax_2d.plot(*path_T[inds])
    ax_2d.grid()
    ax_2d.set_xlabel(x_lable)
    ax_2d.set_ylabel(y_label)

plt.show()
