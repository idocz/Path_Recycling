import numpy as np
from classes.grid import *
from classes.volume import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
from os.path import join
def get_voxel_of_point(point, grid_shape, bbox, bbox_size, res):
    if point[0] == bbox[0, 1]:
        res[0] = grid_shape[0] - 1
    else:
        res[0] = int(((point[0] - bbox[0, 0]) / bbox_size[0]) * grid_shape[0])

    if point[1] == bbox[1, 1]:
        res[1] = grid_shape[1] - 1
    else:
        res[1] = int(((point[1] - bbox[1, 0]) / bbox_size[1]) * grid_shape[1])

    if point[2] == bbox[2, 1]:
        res[2] = grid_shape[2] - 1
    else:
        res[2] = int(((point[2] - bbox[2, 0]) / bbox_size[2]) * grid_shape[2])

def assign_3d(a, b):
    a[0] = b[0]
    a[1] = b[1]
    a[2] = b[2]


def min_3d(x, y, z):
    if x <= y:
        if x <= z:
            return x
        else:
            return z
    else:
        if y <= z:
            return y
        else:
            return z

def sign(a):
    if a >= 0:
        return 1
    else:
        return -1

def travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel):
    inc_x = sign(direction[0])
    inc_y = sign(direction[1])
    inc_z = sign(direction[2])
    voxel_fix_x = (inc_x - 1) / 2
    voxel_fix_y = (inc_y - 1) / 2
    voxel_fix_z = (inc_z - 1) / 2
    t_x = 2
    t_y = 2
    t_z = 2
    if direction[0] != 0:
        t_x = (((current_voxel[0] + 1 + voxel_fix_x) * voxel_size[0]) - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (((current_voxel[1] + 1 + voxel_fix_y) * voxel_size[1]) - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (((current_voxel[2] + 1 + voxel_fix_z) * voxel_size[2]) - current_point[2]) / direction[2]
    t_min = min_3d(t_x, t_y, t_z)
    assign_3d(next_voxel, current_voxel)
    if t_min == t_x:
        next_voxel[0] += inc_x
    if t_min == t_y:
        next_voxel[1] += inc_y
    if t_min == t_z:
        next_voxel[2] += inc_z
    # if t_min < 0:
    #     print("bug:",t_min, current_voxel[0],current_voxel[1],current_voxel[2], next_voxel[0], next_voxel[1], next_voxel[2],"inc:", inc_x, inc_y, inc_z)
    #     print2_3d(current_point, direction)
    current_point[0] = current_point[0] + t_min*direction[0]
    current_point[1] = current_point[1] + t_min*direction[1]
    current_point[2] = current_point[2] + t_min*direction[2]
    return abs(t_min)


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


edge_x = 0.64
edge_y = 0.74
edge_z = 1.04
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]])

beta_cloud = loadmat(join("data", "clouds_dist.mat"))["beta"]
grid_shape = beta_cloud.shape

bbox_size = bbox[:,1] - bbox[:,0]
voxel_size = bbox_size / np.array(grid_shape)
beta_air = 1

start = np.array([ 0.437918, 0.401586, 0.225365 ])
current_voxel = np.empty(3, dtype=np.uint8)
get_voxel_of_point(start, grid_shape, bbox, bbox_size, current_voxel)
print(current_voxel)
exit(0)
# current_voxel = np.array([77, 88, 43])
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
