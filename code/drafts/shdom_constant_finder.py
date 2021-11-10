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
from time import time
from classes.optimizer import *
from os.path import join
import pickle
cuda.select_device(1)


a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
airmspi_data = pickle.load(a_file)
beta_cloud = loadmat(join("data","shdom_exp","FINAL_3D_extinction.mat"))['estimated_extinction'][:,:,:,0]

zenith = airmspi_data["sun_zenith"]
azimuth = airmspi_data["sun_azimuth"]
dir_x = -(np.sin(zenith)*np.cos(azimuth))
dir_y = -(np.sin(zenith)*np.sin(azimuth))
dir_z = -np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
downscale = 4
sun_intensity = 1
TOA = 20
grid = Grid(airmspi_data["bbox"], np.array([50, 50, 50]))


#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004


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
Np = int(1e8)
rr_depth = 20
rr_stop_prob = 0.1
# background = 0
ocean_albedo = 0.01
background = np.load(join("data",f"background_{str(ocean_albedo).split('.')[-1]}_{downscale}_9.npy"))

scene_airmspi = SceneAirMSPI(volume, camera_array_list, sun_direction, sun_intensity, TOA, background, g_cloud, rr_depth, rr_stop_prob)
pad_shapes = scene_airmspi.pixels_shapes
pad_shape = scene_airmspi.pixels_shape

azimuth = 0
zenith_list = [91, 100, 120, 140, 160, 180]
img_list = [loadmat(join("data","shdom_exp",f"sum_azimuth_{azimuth}_sum_zenith_{zenith}"))['images'] for zenith in zenith_list]
N_exp = len(zenith_list)
visual = Visual_wrapper(grid)
for exp in range(N_exp):
    print(f"zenith: {zenith_list[exp]}")
    SUN_THETA = np.deg2rad(zenith_list[exp])
    SUN_PHI = np.deg2rad(azimuth)
    sun_x = np.sin(SUN_THETA)*np.cos(SUN_PHI)
    sun_y = np.sin(SUN_THETA)*np.sin(SUN_PHI)
    sun_z = np.cos(SUN_THETA)
    sun_direction = np.array([sun_x, sun_y, sun_z])
    scene_airmspi.dsun_direction.copy_to_device(sun_direction)
    shdom_imgs = [img for img in img_list[exp][0]]

    I_shdom_pad = np.zeros((N_cams, *pad_shape), dtype=float_reg)
    for cam_ind in range(N_cams):
        shape = pad_shapes[cam_ind]
        I_shdom_pad[cam_ind, :shape[0], :shape[1]] = shdom_imgs[cam_ind][::downscale, ::downscale]
    spp_map = np.copy(I_shdom_pad)
    spp_map = (Np * spp_map / np.sum(spp_map)).astype(np.uint32)

    scene_airmspi.init_cuda_param(Np, init=True)
    scene_airmspi.create_job_list(spp_map)
    scene_airmspi.build_path_list(Np, cam_inds, False)
    I_opt = scene_airmspi.render()
    visual.plot_images(I_shdom_pad, f"zenith: {zenith_list[exp]} (shdom)")
    plt.show()
    visual.plot_images(I_opt, f"zenith: {zenith_list[exp]} (mine)")
    plt.show()
    print(f"ratio:{I_shdom_pad.mean() / I_opt.mean()}\n")



