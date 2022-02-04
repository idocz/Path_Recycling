import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
# from classes.scene_airmspi_backward import *
from classes.scene_airmspi_backward_recycling import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
import pickle
from classes.optimizer import *
from os.path import join

cuda.select_device(1)


a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
beta_shdom = loadmat(join("data","shdom_exp","FINAL_3D_extinction.mat"))['estimated_extinction'][:,:,:,0]

airmspi_data = pickle.load(a_file)

bbox = airmspi_data["bbox"]
zenith = np.deg2rad(airmspi_data["sun_zenith"])
azimuth = np.deg2rad(airmspi_data["sun_azimuth"])
dir_x = (np.sin(zenith)*np.cos(azimuth))
dir_y = (np.sin(zenith)*np.sin(azimuth))
dir_z = np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
print("sun direction:", sun_direction)
downscale = 1
# sun_intensity = 1e1/7
sun_intensity = -1/np.cos(zenith)
print("sun intensity:",sun_intensity)
# sun_intensity = 1
TOA = 20
ocean_albedo = 0.05
exclude_index = 7
background = np.load(join("data",f"background_{str(ocean_albedo).split('.')[-1]}_{downscale}_9.npy"))
# background[exlude_index,:,:] = 0
# background = 0
#####################
# grid parameters #
#####################
# grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
voxel_size = np.array([0.02,0.02,0.02])
grid_shape = ((bbox[:,1]-bbox[:,0])//voxel_size).astype(np.uint16)
grid = Grid(bbox, grid_shape)
beta_shdom = np.ones(grid.shape, dtype=float_reg)
# grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
print("bbox", grid.bbox)
print("shape", grid.shape)
print("voxel_size", grid.voxel_size)




#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004


cloud_mask = airmspi_data["cloud_mask"]
beta_init = np.zeros(grid.shape, dtype=float_reg)


w0_air = 0.912
w0_cloud = 0.99
# Declerations
volume = Volume(grid, beta_init, beta_air, w0_cloud, w0_air)
g_cloud = 0.85
#######################
# Cameras declaration #
#######################

cam_inds = np.arange(9).tolist()
cam_inds.remove(exclude_index)
cam_inds = np.array(cam_inds)
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
Np_max = int(7e8)
Np = int(5e7)
resample_freq = 10
step_size = 3e12
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.1
iterations = 10000000
to_mask = True
beta_max = 20
win_size = 20
max_grad_norm = 30
beta_init_scalar = 7.5

tensorboard = True
tensorboard_freq = 5
#optimizer parameters
alpha = 0.9
beta1 = 0.9
beta2 = 0.999
start_iter = iterations
scaling_factor = 1.5
max_update = 1

# loop parameters
find_initialization = False
compute_spp = False
covergence_goal = 5
convergence_counter = 0

scene_airmspi = SceneAirMSPI(volume, camera_array_list, sun_direction, sun_intensity, TOA, background, g_cloud, rr_depth, rr_stop_prob)
pad_shape = scene_airmspi.pixels_shape




I_gt_pad = np.zeros((total_num_cam, *scene_airmspi.pixels_shape), dtype=float_reg)
for cam_ind in range(total_num_cam):
    pix_shape = scene_airmspi.pixels_shapes[cam_ind]
    I_gt_pad[cam_ind,:pix_shape[0],:pix_shape[1]] = I_gt[cam_ind][::downscale, ::downscale]


image_threshold = np.ones(total_num_cam, dtype=float_reg) * 0.25
image_threshold[0] = 0.35
image_threshold[1] = 0.25
image_threshold[2] = 0.2
image_threshold[3] = 0.18
image_threshold[4] = 0.22
image_threshold[5] = 0.23
image_threshold[6] = 0.23
image_threshold[7] = 0.35
image_threshold[8] = 0.4
hit_threshold = 0.9
spp = 1000

print("Calculating Cloud Mask")

cloud_mask = scene_airmspi.space_curving(I_gt_pad, image_threshold=image_threshold, hit_threshold=hit_threshold,
                                         spp=spp)

for axis in [0,1,2]:
    axis1 = (axis+1)%3
    axis2 = (axis+2)%3
    middle_voxel1 = grid.shape[axis1] // 2
    middle_voxel2 = grid.shape[axis2] // 2
    # while True:





# plot GT
visual = Visual_wrapper(grid)
# visual.plot_images_airmspi(I_gt_pad, resolutions, "GT", downscale)
plt.show()

