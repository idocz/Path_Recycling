import os, sys
my_lib_path = os.path.abspath('../')
sys.path.append(my_lib_path)
import pickle
from utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
from time import time

checkpoint_id = "0201-1719-31_iter1995_iter3000"
load_iter = 20970
dict_init = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{load_iter}.npz"))
dict_gt = np.load(join("checkpoints",checkpoint_id,"data",f"gt.npz"))
beta_init = dict_init["betas"]
beta_cloud = dict_gt["betas"]
cp = pickle.load(open(join("checkpoints",checkpoint_id,"data",f"checkpoint_loader"), "rb" ))
print("Loading the following Scence:")
print(cp)
scene_gpu = cp.recreate_scene()



# Simulation parameters
Np_gt = cp.Np_gt
iter_phase = cp.iter_phase
Nps = cp.Nps
resample_freqs = cp.resample_freqs
step_sizes = cp.step_sizes
step_sizes = [1e9, 1e9, 1e9, 1e9, 1e9, 1e9]
N_cams = scene_gpu.N_cams
Ns = scene_gpu.Ns
pixels = scene_gpu.cameras[0].pixels
cuda_paths = None
phase = 0
while(load_iter > iter_phase[phase]):
    phase += 1

Np = Nps[phase]
resample_freq = resample_freqs[phase]
step_size = step_sizes[phase]
Ns = cp.Ns
iterations = cp.iterations

threadsperblock = 256
seed = None

scene_gpu.init_cuda_param(threadsperblock, Np, seed)

print(f"mask={np.mean(scene_gpu.volume.cloud_mask)}")
tensorboard = True
tensorboard_freq = cp.tensorboard_freq
beta_max = 160

beta_init = dict_init["betas"]
I_gt = dict_gt["images"]
volume = scene_gpu.volume
volume.set_beta_cloud(beta_init)

max_val = np.max(I_gt, axis=(1,2))
# Cloud mask (GT for now)
cloud_mask = volume.cloud_mask



if tensorboard:
    tb = TensorBoardWrapper(I_gt, beta_cloud, title=f"{checkpoint_id}_iter{load_iter}")
    tb.add_scene_text(str(cp))
    pickle.dump(cp, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    print("Checkpoint wrapper has been saved")

optimizer = cp.optimizer


# Initialization

# Initialization
# beta_init[cloud_mask] = np.mean(beta_cloud[cloud_mask])
beta_opt = volume.beta_cloud
grad_norm = None
cuda_paths = scene_gpu.build_paths_list(Np, Ns)
for iter in range(load_iter, iterations):
    print(f"\niter {iter}")
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    mean_dist = np.mean(abs_dist)
    max_dist = np.max(abs_dist)
    rel_dist2 = np.linalg.norm(beta_opt - beta_cloud) / np.linalg.norm(beta_cloud)
    rel_dist1 = relative_distance(beta_cloud, beta_opt)

    print(f"mean_dist = {mean_dist}, max_dist={max_dist}, rel_dist1={rel_dist1}, rel_dist2={rel_dist2}")

    if iter > iter_phase[phase]:
        phase += 1
        print(f"ENTERING PHASE {phase}")
        Np = Nps[phase]
        resample_freq = resample_freqs[phase]
        step_size = step_sizes[phase]
        optimizer.step_size = step_size
        scene_gpu.init_cuda_param(threadsperblock, Np, seed)

    if iter % resample_freq == 0:
        print("RESAMPLING PATHS ")
        start = time()
        del (cuda_paths)
        cuda_paths = scene_gpu.build_paths_list(Np, Ns)
        end = time()
        print(f"resampling took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_gpu.render(cuda_paths, Np, I_gt)
    end = time()
    print(f"rendering took: {end - start}")

    dif = (I_opt - I_gt).reshape(1, 1, 1, N_cams, pixels[0], pixels[1])
    grad_norm = np.linalg.norm(total_grad)

    # updating beta
    optimizer.step(total_grad)
    beta_opt[beta_opt>=beta_max] = beta_max
    # loss calculation
    loss = 0.5 * np.sum(dif ** 2)
    print(f"loss = {loss}, grad_norm={grad_norm}")

    # Writing scalar and images to tensorboard
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, mean_dist, max_dist, rel_dist1, rel_dist2, grad_norm, iter)

