import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene_rr_noNEgrad import *
from classes.scene_multi_eff import *
from classes.scene_seed_noNEgrad import *
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
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

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
# beta_cloud = loadmat(join("data","small_cloud_field.mat"))["beta_smallcf"]
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
ps= 76

pixels = np.array((ps, ps))

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

rr_depth = 20
rr_stop_prob = 0.05

scene_rr = SceneMultiEff(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
scene_seed = SceneSeed(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
scene_rr.set_cloud_mask(beta_cloud>0)
scene_seed.set_cloud_mask(beta_cloud>0)
Np = int(5e7)
print("Init cuda param")
scene_rr.init_cuda_param(int(Np), init=True)
scene_seed.init_cuda_param(int(Np), init=True)
Idiff = np.zeros((N_cams,*camera.pixels), dtype=float_reg)
for _ in range(2):

    cuda_paths = scene_rr.build_paths_list(Np)
    # I_gt,_ = scene_rr.render(cuda_paths,0)
    I_gt,_ = scene_rr.render(cuda_paths,Idiff)

    scene_seed.build_paths_list(Np)
    _,_ = scene_seed.render(I_gt)

# Traditional
start = time()
scene_seed.build_paths_list(Np, to_sort=False)
T_sample_traditional = time() - start

start = time()
_, _ = scene_seed.render(I_gt)
T_render_traditional = time() - start

traditional_baseline = T_sample_traditional + T_render_traditional

# sorting
start = time()
scene_seed.build_paths_list(Np, to_sort=True)
T_sample_sorting = time() - start

start = time()
_, _ = scene_seed.render(I_gt)
T_render_sorting = time() - start

# storing
start = time()
cuda_paths = scene_rr.build_paths_list(Np, to_sort=False)
T_sample_storing = time() - start

start = time()
I_gt, _ = scene_rr.render(cuda_paths, Idiff)
T_render_storing = time() - start

# storing and sorting
start = time()
cuda_paths = scene_rr.build_paths_list(Np, to_sort=True)
T_sample_storing_sorting = time() - start

start = time()
I_gt, _ = scene_rr.render(cuda_paths, Idiff)
T_render_storing_sorting = time() - start

speedup = []
speedup_sorting = []
speedup_storing = []
speedup_storing_sorting = []
max_Nr= 30
Nrs = np.arange(1,max_Nr)
for Nr in Nrs:
    speedup.append(traditional_baseline / ((T_sample_traditional / Nr) + T_render_traditional))
    speedup_sorting.append(traditional_baseline / ((T_sample_sorting / Nr) + T_render_sorting))
    speedup_storing.append(traditional_baseline / ((T_sample_storing / Nr) + T_render_storing))
    speedup_storing_sorting.append(traditional_baseline / ((T_sample_storing_sorting / Nr) + T_render_storing_sorting))



# plt.figure()
text_size = 22
tick_size = 17
plt.figure(figsize=(8,2.5))
output_dir = join("experiments","plots")
plt.plot(Nrs, speedup, label="no sorting no storing")
plt.plot(Nrs, speedup_sorting, label="sorting")
plt.plot(Nrs, speedup_storing, label="storing")
plt.plot(Nrs, speedup_storing_sorting, label="storing & sorting")
plt.xlim(1,max_Nr)
# plt.xlabel(r"$N_r$",fontsize=text_size)
# plt.ylabel("speed up",fontsize=text_size)
xticks = np.arange(0,max_Nr+1,5)
xticks[0]+=1
plt.xticks(xticks, fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.legend(fontsize=tick_size)
plt.grid()
plt.tight_layout()
# plt.savefig(join(output_dir,f"speedup_vs_Nr.pdf"), bbox_inches='tight')
plt.show()