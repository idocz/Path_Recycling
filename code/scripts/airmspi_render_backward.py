import sys
from os.path import join
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
from classes.scene_airmspi_backward_recycling import *
from utils import *
from cuda_utils import *
import pickle
cuda.select_device(0)

sun_intensity = 1
downscale = 8
a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
airmspi_data = pickle.load(a_file)
beta_cloud = loadmat(join("data", "shdom_exp", "FINAL_3D_extinction.mat"))["estimated_extinction"][:,:,:,0]
cloud_mask = airmspi_data["cloud_mask"]
camera_array_list = airmspi_data["camera_array_list"]
camera_array_list = [np.ascontiguousarray(camera_array[::downscale, ::downscale, :6]) for camera_array in camera_array_list]
resolutions = airmspi_data["resolution"]
# beta_cloud = np.zeros(airmspi_data["grid_shape"], dtype=float_reg) * 2
# beta_cloud[cloud_mask] = 2


zenith = airmspi_data["sun_zenith"]
azimuth = airmspi_data["sun_azimuth"]
dir_x = -(np.sin(zenith)*np.cos(azimuth))
dir_y = -(np.sin(zenith)*np.sin(azimuth))
dir_z = -np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
#####################
# grid parameters #
#####################
grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
print(grid.bbox)




#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004

print(beta_cloud.shape)
w0_air = 0.912
w0_cloud = 0.99


# Declerations
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
g_cloud = 0.85
#######################
# Cameras declaration #
#######################


Np = int(1e8)
rr_depth = 20
rr_stop_prob = 0.05

# volume.set_mask(cloud_mask)
TOA = 20
background = 0

scene_airmspi = SceneAirMSPI(volume, camera_array_list, sun_direction, sun_intensity, TOA, background, g_cloud, rr_depth, rr_stop_prob)
pad_shape = scene_airmspi.pixels_shape
N_cams = 9
I_gt = airmspi_data["images"]
I_gt_pad = np.ones((N_cams, *scene_airmspi.pixels_shape), dtype=float_reg) * np.min(I_gt[0])
for cam_ind in range(N_cams):
    pix_shape = scene_airmspi.pixels_shapes[cam_ind]
    I_gt_pad[cam_ind,:pix_shape[0],:pix_shape[1]] = I_gt[cam_ind][::downscale, ::downscale]


spp_map = (Np*I_gt_pad/np.sum(I_gt_pad)).astype(np.uint32)
scene_airmspi.create_job_list(spp_map)
scene_airmspi.init_cuda_param(Np, init=True)
scene_airmspi.build_path_list(Np, np.arange(9), init_cuda=True, sort=True, compute_spp_map=False)
I_opt, total_grad = scene_airmspi.render(I_gt_pad)
print(total_grad.mean())

