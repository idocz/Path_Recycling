import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene_rr import *
# from classes.scene_rr_norecyling import *
from camera import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.optimizer import *
from os.path import join
from tqdm import tqdm
cuda.select_device(1)


########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
beta_cloud = np.load(join("data","jpl_ext.npy"))
beta_cloud = beta_cloud.astype(float_reg)
# Grid parameters #
# bounding box
voxel_size_x = 0.02
voxel_size_y = 0.02
voxel_size_z = 0.04
edge_x = voxel_size_x * beta_cloud.shape[0]
edge_y = voxel_size_y * beta_cloud.shape[1]
edge_z = voxel_size_z * beta_cloud.shape[2]
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]])


print(beta_cloud.shape)
print(bbox)

beta_air = 0.004
# w0_air = 1.0 #0.912
w0_air = 0.912
# w0_cloud = 0.8 #0.9
w0_cloud = 0.99
g_cloud = 0.85

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################
height_factor = 2

focal_length = 50e-3
sensor_size = np.array((50e-3, 50e-3)) / height_factor
ps_max = 76

pixels = np.array((ps_max, ps_max))

N_cams = 9
cameras = []
# volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.7
volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.7
R = height_factor * edge_z

cam_deg = 360 // (N_cams-1)
for cam_ind in range(N_cams-1):
    theta = 29
    theta_rad = theta * (np.pi/180)
    phi = (-(N_cams//2) + cam_ind) * cam_deg
    phi_rad = phi * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi_rad) + volume_center
    euler_angles = np.array((180-theta, 0, phi-90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
t = R * theta_phi_to_direction(0,0) + volume_center
euler_angles = np.array((180, 0, -90))
cameras.append(Camera(t, euler_angles, cameras[0].focal_length, cameras[0].sensor_size, cameras[0].pixels))

# mask parameters
image_threshold = 0.15
hit_threshold = 0.9
spp = 100000

# Simulation parameters
Np_gt = int(5e7)

rr_depth = 20
rr_stop_prob = 0.05


scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)

I_gt_list = []
N_exp = 200
for i in tqdm(range(N_exp)):
    cuda_paths = scene_rr.build_paths_list(Np_gt)
    I_gt = scene_rr.render(cuda_paths).reshape(1,-1)
    I_gt_list.append(I_gt)

I_gt_list = np.concatenate(I_gt_list, axis=0)
I_mean = np.mean(I_gt_list, axis=0)
I_std = np.std(I_gt_list, axis=0)

plt.figure()
N = int(0.08*I_mean.shape[0])
rand_inds = np.random.randint(0,I_mean.shape[0],N)
plt.scatter(I_mean[rand_inds], I_std[rand_inds])
plt.xlabel("mean")
plt.ylabel("std")
plt.show()