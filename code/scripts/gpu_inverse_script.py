import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene import *
from classes.scene_gpu import *
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
beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
beta_cloud = beta_cloud.astype(float_reg)
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
sensor_size = np.array((40e-3, 40e-3))
ps = 55
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
# scene = Scene(volume, cameras, sun_angles, phase_function)



# Simulation parameters
Np_gt = int(1e6)
iter_phase = [500, np.inf]#500, 2000, 3000, 4000, 5000, np.inf]
Nps = [int(1e5), int(1e5)]#, int(1e5), int(1e6), int(1e6), int(2e6), int(5e6)]
resample_freqs = [30, 30]#, 30, 30, 30, 30, 30]
# step_sizes = [1e10, 1e11, 5e11, 5e11, 1e12, 5e11]
step_sizes = [1e9, 1e9]#, 1e9, 1e9, 1e9, 1e9, 1e9]

phase = 0
Np = Nps[phase]
resample_freq = resample_freqs[phase]
step_size = step_sizes[phase]
Ns = 15
iterations = 10000000
to_mask = True
tensorboard = True
tensorboard_freq = 15
beta_max = 160
win_size = 100
# grads_window = np.zeros((win_size, *beta_cloud.shape), dtype=float_reg)

threadsperblock = 256
seed = None
# Cloud mask (GT for now)
cloud_mask = beta_cloud > 0
volume.set_mask(cloud_mask)

scene_gpu = SceneGPU(volume, cameras, sun_angles, g_cloud, g_air, Ns)

visual = Visual_wrapper(scene_gpu)

load_gt = False
if load_gt:
    checkpoint_id = "2212-1250-03"
    I_gt = np.load(join("checkpoints",checkpoint_id,"data","gt.npz"))["images"]
    cuda_paths = None
    print("I_gt has been loaded")
else:
    scene_gpu.init_cuda_param(threadsperblock, Np_gt, seed)
    cuda_paths = scene_gpu.build_paths_list(Np_gt, Ns)
    I_gt = scene_gpu.render(cuda_paths, Np_gt)
    del(cuda_paths)
    cuda_paths = None

scene_gpu.init_cuda_param(threadsperblock, Np, seed)
max_val = np.max(I_gt, axis=(1,2))
# visual.plot_images(I_gt, max_val, "GT")
# plt.show()

alpha = 0.9
beta1 = 0.9
beta2 = 0.999
start_iter = 500
# optimizer = MomentumSGD(volume,step_size, alpha)
optimizer = ADAM(volume,step_size, beta1, beta2, start_iter)
if tensorboard:
    tb = TensorBoardWrapper(I_gt, beta_gt)
    cp_wrapper = CheckpointWrapper(scene_gpu, optimizer, Np_gt, Nps, Ns, resample_freqs, step_sizes, iter_phase, iterations,
                            tensorboard_freq, tb.train_id)
    tb.add_scene_text(str(cp_wrapper))
    pickle.dump(cp_wrapper, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    print("Checkpoint wrapper has been saved")

# Initialization
beta_init = np.zeros_like(beta_cloud)
beta_mean = np.mean(beta_cloud[cloud_mask])
beta_init[cloud_mask] = beta_mean
volume.set_beta_cloud(beta_init)
beta_opt = volume.beta_cloud

# grad_norm = None
cool_down = 0
non_min_couter = 0
next_phase = False
min_loss = 1
for iter in range(iterations):
    print(f"\niter {iter}")
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    mean_dist = np.mean(abs_dist)
    max_dist = np.max(abs_dist)
    rel_dist2 = np.linalg.norm(beta_opt - beta_cloud)/np.linalg.norm(beta_cloud)
    rel_dist1 = relative_distance(beta_cloud, beta_opt)

    print(f"mean_dist = {mean_dist}, max_dist={max_dist}, rel_dist1={rel_dist1}, rel_dist2={rel_dist2}, Np={Np:.2e}, counter={non_min_couter}")


    if iter % resample_freq == 0:
        if non_min_couter >= win_size:
            if Np <= Np_gt and iter > start_iter:
                Np = int(Np * 1.5)
                scene_gpu.init_cuda_param(threadsperblock, Np, seed)
                resample_freq = 30#int(resample_freq * 1.5)
            if Np >= Np_gt:
                Np = Np_gt

        print("RESAMPLING PATHS ")
        start = time()
        del(cuda_paths)
        cuda_paths = scene_gpu.build_paths_list(Np, Ns)
        end = time()
        print(f"resampling took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_gpu.render(cuda_paths, Np, I_gt)
    total_grad *= cloud_mask
    end = time()
    print(f"rendering took: {end-start}")


    dif = (I_opt - I_gt).reshape(1,1,1, N_cams, pixels[0], pixels[1])
    grad_norm = np.linalg.norm(total_grad)

    # updating beta
    optimizer.step(total_grad)
    beta_opt[beta_opt >= beta_max] = beta_mean
    volume.beta_cloud[cloud_mask] = beta_mean
    # loss calculation
    loss = 0.5 * np.sum(dif ** 2)
    if loss < min_loss:
        min_loss = loss
        non_min_couter = 0
    else:
        non_min_couter += 1
    print(f"loss = {loss}, grad_norm={grad_norm}, beta={np.mean(beta_opt)}")

    # Writing scalar and images to tensorboard
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, mean_dist, max_dist, rel_dist1, rel_dist2, grad_norm, iter)




