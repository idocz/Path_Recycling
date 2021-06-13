import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene_lowmem_gpu import *
from classes.camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.optimizer import *
cuda.select_device(0)


########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

beta_cloud = loadmat(join("data", "smoke.mat"))["data"] * 10
# beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
beta_cloud = beta_cloud.astype(float_reg)
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
ps = 254
pixels = np.array((ps, ps))

N_cams = 9
cameras = []
volume_center = (bbox[:,1] - bbox[:,0]) / 1.6
R = 1.5 * edge_z
#
for cam_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams//2) + cam_ind) * (360/N_cams)
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
Ns = 15


seed = None
cloud_mask_real = beta_cloud > 0.1

scene_lowmem = SceneLowMemGPU(volume, cameras, sun_angles, g_cloud, Ns)

visual = Visual_wrapper(scene_lowmem)
cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
I_gt = scene_lowmem.render(cuda_paths)
del(cuda_paths)
visual.plot_images(I_gt, "GT")
plt.show()
cloud_mask = scene_lowmem.space_curving(I_gt, image_threshold=0.7, hit_threshold=0.9, spp=1000)
print(f"accuracy:", np.mean(cloud_mask == cloud_mask_real))
print(f"fp:", np.mean((cloud_mask == 1)*(cloud_mask_real==0)))
fn = (cloud_mask == 0)*(cloud_mask_real==1)
print(f"fn:", np.mean(fn[cloud_mask_real==1]))
fn_exp = (fn * beta_cloud).reshape(-1)
print(f"fn_exp mean:", np.mean(fn_exp[fn_exp!=0]))
print(f"fn_exp max:", np.max(fn_exp))
plt.hist(fn_exp[fn_exp!=0])
print("missed beta:",np.sum(fn_exp)/np.sum(beta_cloud))
beta_masked = beta_cloud * cloud_mask
print("rel_dit:",relative_distance(beta_cloud, beta_masked))

volume.beta_cloud = beta_masked
cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
I_gt = scene_lowmem.render(cuda_paths)
visual.plot_images(I_gt, "GT")
plt.show()