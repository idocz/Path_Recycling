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

exp_name = "0505-1441-10"
cloud_opts = []
iteration = 6300
exp_cloud_name = join("checkpoints",exp_name,"data", f"opt_{iteration}.npz")
cloud_opt = np.load(exp_cloud_name)["betas"]
cloud_opt[cloud_opt<=0.1] = 0
########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
cloud_gt = loadmat(join("data", "smoke.mat"))["data"] * 10
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
ps = 254
pixels = np.array((ps, ps))

N_cams_batch = 9
N_cams = 9
cameras = []
all_cameras = []
volume_center = (bbox[:,1] - bbox[:,0]) / 1.6
R = 1.5 * edge_z
delta_theta = 360/(N_cams*N_cams_batch)
for batch in range(N_cams_batch):
    cameras = []
    for cam_ind in range(N_cams):
        theta_ind = N_cams*batch + cam_ind
        phi = 0
        theta = (-(N_cams//2) + theta_ind) * delta_theta
        theta_rad = theta * (np.pi/180)
        t = R * theta_phi_to_direction(theta_rad,phi) + volume_center
        euler_angles = np.array((180, theta, 0))
        camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
        cameras.append(camera)
    all_cameras.append(cameras)


# Simulation parameters
Np_gt = int(5e8)
Ns = 15

scene_lowmem = SceneLowMemGPU(volume, cameras, sun_angles, g_cloud, Ns)
img_list = []
for batch in tqdm(range(N_cams_batch)):
    scene_lowmem.set_cameras(all_cameras[batch])
    # opt
    # exp_cloud_name = join("checkpoints", exp_name, "data", f"opt_{iterations[batch]}.npz")
    # cloud_opt = np.load(exp_cloud_name)["betas"]
    volume.beta_cloud = cloud_opt
    cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
    I_opt = scene_lowmem.render(cuda_paths)
    # gt
    volume.beta_cloud = cloud_gt
    cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
    I_gt = scene_lowmem.render(cuda_paths)
    img_concat = [np.hstack([np.rot90(img1),np.rot90(img2)]) for img1, img2 in zip(I_opt,I_gt)]
    img_list.extend(img_concat)

print("saving gif...")
output_name = join("experiments","gifs","rotation.gif")
# output_name = None
animate(img_list, interval=60, output_name=output_name, repeat_delay=0)


