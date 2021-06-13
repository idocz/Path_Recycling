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
from scipy.ndimage.morphology import binary_dilation
from scipy import ndimage

cuda.select_device(0)
###################
# Grid parameters #
###################
# bounding box
edge_x = 0.64
edge_y = 0.74
edge_z = 1.04
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]])

########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
beta_cloud = loadmat(join("data", "smoke.mat"))["data"] * 10
beta_cloud = beta_cloud.astype(float_reg)
shape = beta_cloud.shape
# beta_cloud *= (127/beta_cloud.max())


print(beta_cloud)

beta_air = 0.1
w0_air = 1.0
w0_cloud = 0.8
g_cloud = 0.5
g_air = 0.5

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################


focal_length = 60e-3
height_factor = 2.5
sensor_size = np.array((56e-3, 56e-3)) / height_factor
ps = 80
pixels = np.array((ps, ps))

N_cams = 9
volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.6


R = height_factor * edge_z
cameras = []
for cam_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams//2) + cam_ind) * 40
    theta_rad = theta * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi) + volume_center
    euler_angles = np.array((180, theta, 0))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
#
# R = 2.5 * edge_z
# t = R * theta_phi_to_direction(0, 0) + volume_center
# euler_angles = np.array((180, 0, 0))
# cameras.append(Camera(t, euler_angles, focal_length, sensor_size, pixels))
# phis = np.linspace(0, 360, N_cams - 1)
# phis_rad = phis * (np.pi / 180)
# theta = 50
# theta_rad = theta * (np.pi / 180)
# for k in range(N_cams - 1):
#     t = R * theta_phi_to_direction(theta_rad, phis_rad[k]) + volume_center
#     euler_angles = np.array((180, theta, phis[k]))
#     camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
#     cameras.append(camera)

# Simulation parameters
Np_gt = int(1e7)
Ns = 15
# grads_window = np.zeros((win_size, *beta_cloud.shape), dtype=float_reg)

seed = None
# Cloud mask (GT for now)
cloud_mask_real = beta_cloud > 0
# cloud_mask = beta_cloud >= 0
volume.set_mask(cloud_mask_real)

scene_gpu = SceneLowMemGPU(volume, cameras, sun_angles, g_cloud, Ns)

visual = Visual_wrapper(scene_gpu)
scene_gpu.init_cuda_param(Np_gt, init=True)
cuda_paths = scene_gpu.build_paths_list(Np_gt, Ns)
print(scene_gpu.Np_nonan)
I_gt = scene_gpu.render(cuda_paths)
del (cuda_paths)
cuda_paths = None
max_val = np.max(I_gt, axis=(1, 2))
visual.plot_images(I_gt, "GT")
# plt.show()

plt.hist(I_gt.reshape(-1))
# plt.show()
img_mask = np.zeros(I_gt.shape, dtype=np.bool)
img_mask[I_gt>3e-6] = True

for k in range(N_cams):
    ax = plt.subplot(N_cams,2,1 + 2*k)
    ax.imshow(img_mask[k], cmap="gray")
    ax = plt.subplot(N_cams,2,2 + 2*k)
    ax.imshow(I_gt[k], cmap="gray")
plt.show()


cloud_mask = np.zeros(shape, dtype=np.bool)
point = np.zeros(3, dtype=np.float)
N_sample = 1
counters = np.zeros(cloud_mask.shape)
for cam_ind in tqdm(range(N_cams)):
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for exp in range(N_sample):
                    point[0] = np.random.uniform(grid.voxel_size[0] * i, (i+1) * grid.voxel_size[0])
                    point[1] = np.random.uniform(grid.voxel_size[1] * j, (j+1) * grid.voxel_size[1])
                    point[2] = np.random.uniform(grid.voxel_size[2] * k, (k+1) * grid.voxel_size[2])
                    pixel = cameras[cam_ind].project_point(point)
                    if img_mask[cam_ind, pixel[0], pixel[1]]:
                        counters[i,j,k] += 1
                        break

cloud_mask = counters >= 7


for i in range(2):
    print(f"accuracy:", np.mean(cloud_mask == cloud_mask_real))
    print(f"fp:", np.mean((cloud_mask == 1)*(cloud_mask_real==0)))
    fn = (cloud_mask == 0)*(cloud_mask_real==1)
    print(f"fn:", np.mean(fn))
    fn_exp = (fn * beta_cloud).reshape(-1)
    print(f"fn_exp mean:", np.mean(fn_exp))
    print(f"fn_exp max:", np.max(fn_exp))
    print(f"fn_exp min:", np.min(fn_exp[fn_exp!=0]))
    plt.hist(fn_exp[fn_exp!=0])
    print("missed beta:",np.sum(fn_exp)/np.sum(beta_cloud))
    plt.show()
    print("DIALATING")
    # kernel = ndimage.generate_binary_structure(2, )
    # cloud_mask = binary_dilation(cloud_mask, structure=kernel)
    cloud_mask = binary_dilation(cloud_mask)

