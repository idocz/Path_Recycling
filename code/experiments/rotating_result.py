import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate
from os.path import join
from classes.scene_hybrid_gpu import *
from classes.camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
from classes.optimizer import *
from tqdm import tqdm
cuda.select_device(0)

exp_name = "0705-2057-19"
iterations =[0, 1700, 6500, 10900, 16300]
mins = [0, 1, 5, 15, 60]
########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
cloud_gt = loadmat(join("data", "smoke.mat"))["data"] * 10
cloud_gt = np.ascontiguousarray(np.rot90(cloud_gt, axes=(2,1)))
cloud_gt =np.roll(cloud_gt, axis=0, shift=-15)

cloud_gt = cloud_gt.astype(float_reg)
# bounding box
voxel_size_x = 0.02
voxel_size_y = 0.02
voxel_size_z = 0.02
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


ps = 254
pixels = np.array((ps, ps))

N_cams_batch = 9
N_cams = 9

delta_theta = 360/(N_cams*N_cams_batch)
N_cams = 9
height_factor = 1.5
focal_length = 50e-3
sensor_size = np.array((47e-3, 47e-3)) / height_factor
volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.65
R = height_factor * edge_z
cameras = []
all_cameras = []
delta_phi = 360 / (N_cams * N_cams_batch)

for batch in range(N_cams_batch):
    cameras = []
    for cam_ind in range(N_cams):
        theta = np.pi / 2
        phi_ind = N_cams * batch + cam_ind
        phi = (-(N_cams // 2) + phi_ind) * delta_phi
        phi_rad = phi * (np.pi / 180)
        t = R * theta_phi_to_direction(theta, phi_rad) + volume_center
        t[2] -= 0.55
        euler_angles = np.array((90, 0, phi - 90))
        camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
        cameras.append(camera)
    all_cameras.append(cameras)

# Simulation parameters
Np_gt = int(3e8)
Ns = 15

scene_hybrid = SceneHybridGpu(volume, cameras, sun_angles, g_cloud, Ns)

# creating exp dir in checkpoints
output_dir = join("checkpoints", exp_name, "experiments", "gifs")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for iteration, min in zip(iterations, mins):
    exp_cloud_name = join("checkpoints", exp_name, "data", f"opt_{iteration}.npz")
    cloud_opt = np.load(exp_cloud_name)["betas"]
    cloud_opt[cloud_opt <= 0.1] = 0
    img_list = []
    for batch in tqdm(range(N_cams_batch)):
        scene_hybrid.set_cameras(all_cameras[batch])
        # opt
        # exp_cloud_name = join("checkpoints", exp_name, "data", f"opt_{iterations[batch]}.npz")
        # cloud_opt = np.load(exp_cloud_name)["betas"]
        volume.beta_cloud = cloud_opt
        cuda_paths = scene_hybrid.build_paths_list(Np_gt, Ns)
        I_opt = scene_hybrid.render(cuda_paths)
        # gt
        volume.beta_cloud = cloud_gt
        cuda_paths = scene_hybrid.build_paths_list(Np_gt, Ns)
        I_gt = scene_hybrid.render(cuda_paths)
        img_concat = [np.hstack([np.rot90(img1),np.rot90(img2)]) for img1, img2 in zip(I_opt,I_gt)]
        img_list.extend(img_concat)

    print("saving gif...")
    output_name = join(output_dir,f"rotation_iter{iteration}_min{min}.mp4")
    # output_name = None
    animate(img_list, interval=100, output_name=output_name, repeat_delay=0)


