import os, sys
my_lib_path = os.path.abspath('../')
sys.path.append(my_lib_path)
from deprecated.scene_lowmem_gpu import *
from camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from classes.checkpoint_wrapper import CheckpointWrapper
from time import time
from classes.optimizer import *

cuda.select_device(0)


########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
beta_cloud = loadmat(join("data", "rico.mat"))["beta"]

# beta_cloud = loadmat(join("data", "rico2.mat"))["vol"]
beta_cloud = beta_cloud.astype(float_reg)
# beta_cloud *= (127/beta_cloud.max())
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
w0_cloud = 0.9
g_cloud = 0.85

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################


focal_length = 60e-3
sensor_size = np.array((40e-3, 40e-3))
ps = 80
pixels = np.array((ps, ps))

N_cams = 9
cameras = []
volume_center = (bbox[:,1] - bbox[:,0])/2
R = 1.5 * edge_z

for cam_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams//2) + cam_ind) * 40
    theta_rad = theta * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi) + volume_center
    euler_angles = np.array((180, theta, 0))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)

# height_factor = 1.5
# focal_length = 45e-3
# sensor_size = np.array((40e-3, 40e-3)) / height_factor
# ps = 55
#
# pixels = np.array((ps, ps))
#
# N_cams = 9
# cameras = []
# volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
# R = height_factor * edge_z
# for cam_ind in range(N_cams):
#     phi = 0
#     theta = (-(N_cams // 2) + cam_ind) * 40
#     theta_rad = theta * (np.pi / 180)
#     t = R * theta_phi_to_direction(theta_rad, phi) + volume_center
#     euler_angles = np.array((180, theta, 0))
#     camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
#     cameras.append(camera)


# Simulation parameters
Np_gt = int(5e7)
Np_max = int(5e7)
Np = int(5e5)
resample_freq = 10
step_size = 2e8
# Ns = 15
Ns = 15
iterations = 10000000
to_mask = True
tensorboard = True
tensorboard_freq = 15
beta_max = beta_cloud.max()
# beta_max = 140
win_size = 100


seed = None
# Cloud mask (GT for now)
cloud_mask = beta_cloud > 0
# cloud_mask = beta_cloud >= 0
cloud_mask = np.load(join("data","Rico_mask_2.npy"))
volume.set_mask(cloud_mask)

scene_lowmem = SceneLowMemGPU(volume, cameras, sun_angles, g_cloud, Ns)

visual = Visual_wrapper(scene_lowmem)

cuda_paths = scene_lowmem.build_paths_list(Np_gt, Ns)
I_gt = scene_lowmem.render(cuda_paths)
del(cuda_paths)
cuda_paths = None
max_val = np.max(I_gt, axis=(1,2))
visual.plot_images(I_gt, "GT")
plt.show()

# mask_thresh = 2e-6
# img_mask = np.zeros(I_gt.shape, dtype=np.bool)
# img_mask[I_gt > mask_thresh] = 1
# scene_gpu.space_curving(img_mask)
# cloud_mask = scene_gpu.volume.cloud_mask


scene_lowmem.init_cuda_param(Np)
alpha = 0.9
beta1 = 0.9
beta2 = 0.999
start_iter = 500
# optimizer = SGD(volume,step_size)
beta_mean = np.mean(beta_cloud[volume.cloud_mask])
# optimizer = MomentumSGD(volume, step_size, alpha, beta_mean, beta_max)
optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_mean, beta_max, 1)
if tensorboard:
    tb = TensorBoardWrapper(I_gt, beta_gt)
    cp_wrapper = CheckpointWrapper(scene_lowmem, optimizer, Np_gt, Np, Ns, resample_freq, step_size, iterations,
                            tensorboard_freq, tb.train_id)
    tb.add_scene_text(str(cp_wrapper))
    pickle.dump(cp_wrapper, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    print("Checkpoint wrapper has been saved")

# Initialization
beta_init = np.zeros_like(beta_cloud)
beta_init[volume.cloud_mask] = 2
# beta_init[volume.cloud_mask] = 0
volume.set_beta_cloud(beta_init)
beta_opt = volume.beta_cloud

# grad_norm = None
non_min_couter = 0
next_phase = False
min_loss = 1
for iter in range(iterations):
    print(f"\niter {iter}")
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    max_dist = np.max(abs_dist)
    rel_dist1 = relative_distance(beta_cloud, beta_opt)

    print(f"rel_dist1={rel_dist1}, max_dist={max_dist}, Np={Np:.2e}, counter={non_min_couter}")

    if iter % resample_freq == 0:
        if non_min_couter >= win_size:
            if Np < Np_max and iter > start_iter:
                Np = int(Np * 1.5)
                resample_freq = 30
                non_min_couter = 0
                step_size *= 1.5
        print("RESAMPLING PATHS ")
        start = time()
        del(cuda_paths)
        cuda_paths = scene_lowmem.build_paths_list(Np, Ns)
        end = time()
        print(f"building path list took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_lowmem.render(cuda_paths, I_gt=I_gt)
    end = time()
    print(f"rendering took: {end-start}")


    dif = (I_opt - I_gt).reshape(1,1,1, N_cams, pixels[0], pixels[1])
    grad_norm = np.linalg.norm(total_grad)

    # updating beta
    # beta_opt -= step_size*total_grad
    # beta_opt[beta_opt >= beta_max] = beta_mean
    # beta_opt[beta_opt < 0] = 0
    # loss calculation
    start = time()
    optimizer.step(total_grad)
    print("gradient step took:",time()-start)
    loss = 0.5 * np.sum(dif ** 2)
    if loss < min_loss:
        min_loss = loss
        non_min_couter = 0
    else:
        non_min_couter += 1
    print(f"loss = {loss}, grad_norm={grad_norm}, max_grad={np.max(total_grad)}")

    # Writing scalar and images to tensorboard
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, max_dist, rel_dist1, Np, iter)




