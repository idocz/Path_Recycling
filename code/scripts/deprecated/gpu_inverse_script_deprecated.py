import os, sys
my_lib_path = os.path.abspath('../')
sys.path.append(my_lib_path)
from classes.deprecated.scene_gpu_deprecated import *
from classes.camera import *
from classes.visual import *
from utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from classes.checkpoint_wrapper import CheckpointWrapper
from time import time
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


print(beta_cloud)

beta_air = 0.1
w0_air = 1.0
w0_cloud = 0.8

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
g = 0.5
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################


focal_length = 60e-3
sensor_size = np.array((40e-3, 40e-3))
ps = 55
pixels = np.array((ps, ps))

N_cams = 5
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

scene_gpu = SceneGPU(volume, cameras, sun_angles, g)

visual = Visual_wrapper(scene_gpu)

# Simulation parameters
Np_gt = int(1e5)
iter_phase = [500, 2000, 3000, np.inf]
Nps = [int(1e5), int(1e6), int(1e6), int(1e6)]
resample_freqs = [10, 30, 30, 30]
step_sizes = [1e10, 1e11, 5e11, 1e12]

phase = 0
Np = Nps[phase]
resample_freq = resample_freqs[phase]
step_size = step_sizes[phase]
Ns = 15
iterations = 10000000

tensorboard = True
tensorboard_freq = 5
beta_max = 160

load_gt = True
if load_gt:
    checkpoint_id = "2212-1250-03"
    I_gt = np.load(join("checkpoints",checkpoint_id,"data","gt.npz"))["images"]
    cuda_paths = None
    print("I_gt has been loaded")
else:
    cuda_paths = scene_gpu.build_paths_list(Np_gt, Ns)
    I_gt = scene_gpu.render(cuda_paths)


max_val = np.max(I_gt, axis=(1,2))
visual.plot_images(I_gt, max_val, "GT")
plt.show()


# Cloud mask (GT for now)
cloud_mask = beta_cloud > 0
volume.set_mask(cloud_mask)

if tensorboard:
    tb = TensorBoardWrapper(I_gt, beta_gt)
    cp_wrapper = CheckpointWrapper(scene_gpu, Np_gt, Nps, Ns, resample_freqs, step_sizes, iter_phase, iterations,
                            tensorboard_freq, tb.train_id)
    tb.add_scene_text(str(cp_wrapper))
    pickle.dump(cp_wrapper, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    print("Checkpoint wrapper has been saved")

# Initialization
beta_init = np.zeros_like(beta_cloud, dtype=np.float64)
volume.set_beta_cloud(beta_init)
# beta_init[cloud_mask] = np.mean(beta_cloud[cloud_mask])
beta_opt = np.copy(beta_init)
last_beta = beta_opt

min_rel = 2
grad_norm = None
for iter in range(iterations):
    print(f"iter {iter}")
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    mean_dist = np.mean(abs_dist)
    max_dist = np.max(abs_dist)
    rel_dist = np.linalg.norm(beta_opt - beta_cloud)/np.linalg.norm(beta_cloud)

    print(f"mean_dist = {mean_dist}, max_dist={max_dist}, rel_dist={rel_dist}")
    if rel_dist < min_rel:
        min_rel = rel_dist
        last_beta = np.copy(beta_opt)

    if iter > iter_phase[phase]:
        phase += 1
        print(f"ENTERING PHASE {phase}")
        Np = Nps[phase]
        resample_freq = resample_freqs[phase]
        step_size = step_sizes[phase]



    if iter % resample_freq == 0:
        print("RESAMPLING PATHS ")
        del(cuda_paths)
        cuda_paths = scene_gpu.build_paths_list(Np, Ns)

    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_gpu.render(cuda_paths, I_gt)
    end = time()
    print(f"rendering took: {end-start}")

    if np.isnan(np.linalg.norm(I_opt)):
        print("INF image NORM")
        beta_opt = last_beta
        volume.set_beta_cloud(beta_opt)
        tb.writer.add_text("log", f"error {iter}: img_norm is nan. last_grad_norm={grad_norm}  \n")
        print("RESAMPLING PATHS ")
        del (cuda_paths)
        cuda_paths = scene_gpu.build_paths_list(Np, Ns)
        continue

    elif np.isnan(np.linalg.norm(total_grad)):
        print("INF grad NORM")
        beta_opt = last_beta
        volume.set_beta_cloud(beta_opt)
        tb.writer.add_text("log",f"error {iter}: grad_norm is nan. last_grad_norm={grad_norm}  \n")
        print("RESAMPLING PATHS ")
        del (cuda_paths)
        cuda_paths = scene_gpu.build_paths_list(Np, Ns)
        continue

    dif = (I_opt - I_gt).reshape(1,1,1, N_cams, pixels[0], pixels[1])

    # gradient calculation
    total_grad *= cloud_mask
    grad_norm = np.linalg.norm(total_grad)

    # updating beta
    beta_opt -= step_size * total_grad

    #thresholding beta
    beta_opt[beta_opt<0] = 0
    beta_opt[beta_opt>beta_max] = beta_max
    volume.set_beta_cloud(beta_opt)

    # loss calculation
    loss = 0.5 * np.sum(dif ** 2)
    print(f"loss = {loss}, grad_norm={grad_norm}")

    # Writing scalar and images to tensorboard
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, mean_dist, max_dist, rel_dist, grad_norm, iter)




