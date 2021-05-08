import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene import *
from classes.scene_lowmem_gpu import *
from classes.camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from classes.checkpoint_wrapper import CheckpointWrapper
from time import time
from classes.optimizer import *
from os.path import join
from tqdm import tqdm
cuda.select_device(0)


########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
beta_cloud = loadmat(join("data", "smoke.mat"))["data"] * 10
# beta_cloud = beta_cloud.T
# beta_cloud = loadmat(join("data", "rico2.mat"))["vol"]
beta_cloud = beta_cloud.astype(float_reg)
# beta_cloud *= (127/beta_cloud.max())
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
w0_cloud = 0.9
g_cloud = 0.5

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################


focal_length = 60e-3
sensor_size = np.array((56e-3, 56e-3))


ps_gt = 160
pixels = np.array([ps_gt, ps_gt])
N_cams = 9
cameras = []
volume_center = (bbox[:,1] - bbox[:,0]) / 1.6
R = 1.5 * edge_z
#
for cam_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams//2) + cam_ind) * 40
    theta_rad = theta * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi) + volume_center
    euler_angles = np.array((180, theta, 0))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
#
# for cam_ind in range(N_cams):
#     theta = np.pi/2
#     phi = (-(N_cams//2) + cam_ind) * 40
#     phi_rad = phi * (np.pi/180)
#     t = R * theta_phi_to_direction(theta,phi_rad) + volume_center
#     print(cam_ind, t-volume_center)
#     euler_angles = np.array((90, 0, phi-90))
#     camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
#     cameras.append(camera)



# Simulation parameters
Np_gt = int(5e7)
Np_max = int(5e7)
Np = int(5e5)
resample_freq = 10
step_size = 1e9
# Ns = 15
Ns = 5
iterations = 10000000
to_mask = True
tensorboard = True
tensorboard_freq = 15
beta_max = beta_cloud.max()
win_size = 100


scene_lowmem = SceneLowMemGPU(volume, cameras, sun_angles, g_cloud, Ns)
scene_lowmem.set_cloud_mask(volume.cloud_mask)

visual = Visual_wrapper(scene_lowmem)
visual.create_grid()
visual.plot_cameras()
# visual.plot_medium()
plt.show()
cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
I_gt, grad = scene_lowmem.render(cuda_paths, 0)
scene_lowmem.upscale_cameras(5)
cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
I_gt_downscale, grad_downscale = scene_lowmem.render(cuda_paths, 0)
print(np.mean(I_gt)/np.mean(I_gt_downscale))
print(np.mean(grad_downscale)/np.mean(grad))
print(np.sum(I_gt_downscale**2)/np.sum(I_gt**2))