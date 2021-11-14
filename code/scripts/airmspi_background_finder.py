import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
# from classes.scene_airmspi_backward import *
from classes.scene_airmspi_backward_recycling import *
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

beta_cloud = loadmat(join("data", "FINAL_3D_extinction.mat"))["estimated_extinction"][:,:,:,0]
a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
airmspi_data = pickle.load(a_file)

zenith = np.deg2rad(airmspi_data["sun_zenith"])
azimuth = np.deg2rad(airmspi_data["sun_azimuth"])
dir_x = np.sin(zenith)*np.cos(azimuth)
dir_y = np.sin(zenith)*np.sin(azimuth)
dir_z = np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
downscale = 4

# sun_intensity = 1e1/7
sun_intensity = -1/np.cos(zenith)
TOA = 20
#####################
# grid parameters #
#####################
# grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
grid = Grid(airmspi_data["bbox"], np.array([50, 50, 50]))
# grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
print(grid.bbox)




#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004
# beta_air = 0.00001


cloud_mask = airmspi_data["cloud_mask"]


w0_air = 0.912
w0_cloud = 0.99
# Declerations
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
g_cloud = 0.85
#######################
# Cameras declaration #
#######################
# exclude_index = 7
cam_inds = np.arange(9)#.tolist()
# cam_inds.remove(exclude_index)
# cam_inds = np.array(cam_inds)
print(cam_inds)
# exit()
N_cams = len(cam_inds)

resolutions = airmspi_data["resolution"]#[:3]
I_gt = airmspi_data["images"]#[:3]

camera_array_list = airmspi_data["camera_array_list"]#[:1]
total_num_cam = len(camera_array_list)
camera_array_list = [np.ascontiguousarray(camera_array[::downscale, ::downscale, :6]) for camera_array in camera_array_list]
# camera_array_list = [np.ascontiguousarray(camera_array[:, :, :6]) for camera_array in camera_array_list]

# Simulation parameters
logNp = 8
Np = int(1 * 10 **(logNp))
rr_depth = 20
rr_stop_prob = 0.1

ocean_albedo = 0.001965592532275277
background = 0
scene_airmspi = SceneAirMSPI(volume, camera_array_list, sun_direction, sun_intensity, TOA, background, g_cloud, rr_depth, rr_stop_prob)
pad_shape = scene_airmspi.pixels_shape

scene_airmspi.init_cuda_param(Np, init=True)
scene_airmspi.cam_inds = cam_inds
# I_background = scene_airmspi.render_background(Np, ocean_albedo, False)
# np.save(join("data",f"background_{str(ocean_albedo).split('.')[-1]}_{downscale}_{logNp}.npy"), I_background)


# logNp = 9
# I_background = np.load(join("data",f"background_{str(ocean_albedo).split('.')[-1]}_{downscale}_{logNp}.npy"))

image_threshold = np.ones(total_num_cam, dtype=float_reg) * 0.25
image_threshold[0] = 0.35
image_threshold[1] = 0.25
image_threshold[2] = 0.2
image_threshold[3] = 0.18
image_threshold[4] = 0.2
image_threshold[5] = 0.2
image_threshold[6] = 0.23
image_threshold[7] = 0.25
image_threshold[8] = 0.32
hit_threshold = 0.9

I_gt_pad = np.zeros((total_num_cam, *scene_airmspi.pixels_shape), dtype=float_reg)
for cam_ind in range(total_num_cam):
    pix_shape = scene_airmspi.pixels_shapes[cam_ind]
    I_gt_pad[cam_ind, :pix_shape[0], :pix_shape[1]] = I_gt[cam_ind][::downscale, ::downscale]

I_mask = np.zeros(I_gt_pad.shape, dtype=bool)
I_total_norm = (I_gt_pad - I_gt_pad.min()) / (I_gt_pad.max() - I_gt_pad.min())
for k in range(N_cams):
    I_mask[k][I_total_norm[k] > image_threshold[k]] = True
I_mask = np.logical_not(I_mask)

spp_map = np.copy(I_gt_pad)
spp_map = (Np * spp_map / np.sum(spp_map)).astype(np.uint32)
scene_airmspi.create_job_list(spp_map)
scene_airmspi.build_path_list(Np, cam_inds, False)
I_opt = scene_airmspi.render()
I_background = scene_airmspi.render_background(Np, 0, False)
plt.figure()
I_grid = imgs2grid(I_opt+I_background)
plt.imshow(I_grid, cmap="gray")
plt.show()
N = 20
scalars = np.logspace(-2,-1,N)
# scalars = [0]
res = []
I_deltas = []
for albedo in tqdm(scalars):
    I_delta = np.zeros_like(I_gt_pad)
    # I_delta[I_mask] = np.abs(I_gt_pad[I_mask]-scalar*I_background[I_mask])
    I_background = scene_airmspi.render_background(Np, albedo, False)
    I_background += I_opt
    I_delta[I_mask] = np.abs(I_gt_pad[I_mask]-I_background[I_mask])
    I_delta_grid = imgs2grid(I_delta)
    plt.figure()
    plt.imshow(I_delta_grid,cmap="gray")
    plt.title(f"scalar:{albedo}, error:{np.linalg.norm(I_delta)}")
    plt.show()
    res.append(np.linalg.norm(I_delta))
    I_deltas.append(I_delta)
plt.figure()
plt.semilogx(scalars, res)
plt.show()
min_ind = np.argmin(res)
best_scalar = scalars[min_ind]
I_delta = I_deltas[min_ind]
# I_delta[I_mask] = np.abs(I_gt_pad[I_mask] - best_scalar * I_background[I_mask])
print("chosen albedo is:", scalars[min_ind])
I_gt_mean = I_gt_pad[I_mask].mean()
I_delta_mean = I_delta[I_mask].mean()
print(f"I_gt_mean={I_gt_mean}, I_delta_mean={I_delta_mean}")
I_delta_grid = imgs2grid(I_delta)
plt.figure()
plt.imshow(I_delta_grid,cmap="gray")
plt.title(f"best albedo:{best_scalar}, error:{res[min_ind]}")
plt.show()

# exit()
# spp_map = np.copy(I_gt_pad)
# spp_map = (Np * spp_map / np.sum(spp_map)).astype(np.uint32)
#
# scene_airmspi.create_job_list(spp_map)
# scene_airmspi.build_path_list(Np, cam_inds, False)
# I_opt = scene_airmspi.render()
# I_opt += background
#
# for im in I_opt:
#     plt.figure()
#     plt.imshow(im, cmap="gray")
#     plt.show()