import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate
from os.path import join
from classes.scene_lowmem_gpu import *
from classes.camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
from classes.optimizer import *
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
cloud_gt = loadmat(join("data", "smoke.mat"))["data"] * 10
cloud_mask = cloud_gt > 0.1
cloud_gt = cloud_gt.astype(float_reg)
# bounding box
voxel_size_x = 0.02
voxel_size_y = 0.02
voxel_size_z = 0.04
edge_x = voxel_size_x * cloud_gt.shape[0]
edge_y = voxel_size_y * cloud_gt.shape[1]
edge_z = voxel_size_z * cloud_gt.shape[2]
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]])


print(cloud_gt.shape)
print(bbox)

beta_air = 0.004
w0_air = 0.912
w0_cloud = 0.9
g_cloud = 0.5

# Declerations
grid = Grid(bbox, cloud_gt.shape)
volume = Volume(grid, cloud_gt, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(cloud_gt)
#######################
# Cameras declaration #
#######################


focal_length = 60e-3
sensor_size = np.array((56e-3, 56e-3))
ps = 120
pixels = np.array((ps, ps))

N_cams = 1
cameras = []
volume_center = (bbox[:,1] - bbox[:,0]) / 1.6
R = 1.5 * edge_z
delta_theta = 360/N_cams

for theta_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams//2) + theta_ind) * delta_theta
    theta_rad = theta * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi) + volume_center
    euler_angles = np.array((180, theta, 0))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)


# Simulation parameters
Np_gt = int(5e8)
Ns = 15
frames = 150
cloud_hom = np.zeros_like(cloud_gt)
cloud_hom[cloud_mask] = np.mean(cloud_gt[cloud_mask])
cloud_step = (cloud_gt - cloud_hom)/frames



scene_lowmem = SceneLowMemGPU(volume, cameras, sun_angles, g_cloud, Ns)
volume.beta_cloud = cloud_gt
cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
I_gt = scene_lowmem.render(cuda_paths)
max_val = I_gt.max()

img_list = []
for frame in tqdm(range(frames)):
    volume.beta_cloud = cloud_hom + frame * cloud_step
    if frame == 0:
        cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
    I_opt = scene_lowmem.render(cuda_paths)
    I_opt[I_opt>max_val] = max_val
    img_concat = [np.hstack([np.rot90(img1),np.rot90(img2)]) for img1, img2 in zip(I_opt,I_gt)]
    img_list.extend(img_concat)

print("saving gif...")
output_name = join("experiments","gifs","recycling_illustration.mp4")
# output_name = None
animate(img_list, interval=60, repeat_delay=2000, output_name=output_name)


