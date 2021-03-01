import os, sys
my_lib_path = os.path.abspath('../')
sys.path.append(my_lib_path)
import pickle
from utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper


checkpoint_id = "1612-1307-25_iter2755_iter3950_iter4685_iter5790_iter6600"
load_iter = 10470
dict_init = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{load_iter}.npz"))
dict_gt = np.load(join("checkpoints",checkpoint_id,"data",f"gt.npz"))
beta_init = dict_init["betas"]
beta_gt = dict_gt["betas"]
beta_gt = loadmat(join("data", "rico.mat"))["beta"]
cp = pickle.load(open(join("checkpoints",checkpoint_id,"data",f"checkpoint_loader"), "rb" ))
print("Loading the following Scence:")
print(cp)
scene_numba = cp.scene



# Simulation parameters
Np_gt = cp.Np_gt
iter_phase = cp.iter_phase
Nps = cp.Nps
resample_freqs = cp.resample_freqs
step_sizes = cp.step_sizes

phase = 0
while(load_iter > iter_phase[phase]):
    phase += 1

Np = Nps[phase]
resample_freq = resample_freqs[phase]
step_size = step_sizes[phase]
Ns = cp.Ns
iterations = cp.iterations

tensorboard = True
tensorboard_freq = cp.tensorboard_freq
beta_max = 160

beta_init = dict_init["betas"]
I_gt = dict_gt["images"]
volume = scene_numba.volume
volume.set_beta_cloud(beta_init)

max_val = np.max(I_gt, axis=(1,2))
# Cloud mask (GT for now)
cloud_mask = volume.cloud_mask



if tensorboard:
    tb = TensorBoardWrapper(I_gt, beta_gt, title=f"{checkpoint_id}_iter{load_iter}")
    tb.add_scene_text(str(cp))
    pickle.dump(cp, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    print("Checkpoint wrapper has been saved")


# Initialization

beta_opt = np.copy(beta_init)
N_cams = len(scene_numba.cameras)
pixels = scene_numba.cameras[0].pixels
last_beta = beta_opt
print("RESAMPLING PATHS ")
paths = scene_numba.build_paths_list(Np, Ns)
min_rel = 2
grad_norm = None
for iter in range(load_iter, iterations):
    print(f"iter {iter}")
    abs_dist = np.abs(beta_gt[cloud_mask] - beta_opt[cloud_mask])
    mean_dist = np.mean(abs_dist)
    max_dist = np.max(abs_dist)
    rel_dist = np.linalg.norm(beta_opt - beta_gt) / np.linalg.norm(beta_gt)
    if rel_dist < min_rel:
        min_rel = rel_dist
        last_beta = np.copy(beta_opt)
    print(f"mean_dist = {mean_dist}, max_dist={max_dist}, rel_dist={rel_dist}")
    if iter > iter_phase[phase]:
        phase += 1
        print(f"ENTERING PHASE {phase}")
        Np = Nps[phase]
        resample_freq = resample_freqs[phase]
        step_size = step_sizes[phase]

    if iter % resample_freq == 0:
        print("RESAMPLING PATHS ")
        del (paths)
        paths = scene_numba.build_paths_list(Np, Ns)

    # differentiable forward model
    I_opt, total_grad = scene_numba.render(paths, differentiable=True)
    # visual.plot_images(I_opt, max_val, "GT")
    # plt.show()
    if np.isnan(np.linalg.norm(I_opt)):
        print("INF image NORM")
        beta_opt = last_beta
        volume.set_beta_cloud(beta_opt)
        tb.writer.add_text("log", f"error {iter}: img_norm is nan. last_grad_norm={grad_norm}  \n")
        print("RESAMPLING PATHS ")
        del (paths)
        paths = scene_numba.build_paths_list(Np, Ns)
        continue
    elif np.isnan(np.linalg.norm(total_grad)):
        print("INF grad NORM")
        beta_opt = last_beta
        volume.set_beta_cloud(beta_opt)
        tb.writer.add_text("log",f"error {iter}: grad_norm is nan. last_grad_norm={grad_norm}  \n")
        print("RESAMPLING PATHS ")
        del (paths)
        paths = scene_numba.build_paths_list(Np, Ns)
        continue

    dif = (I_opt - I_gt)
    dif = dif.reshape(1, 1, 1, N_cams, pixels[0], pixels[1])

    # gradient calculation
    total_grad *= dif
    total_grad = np.sum(total_grad, axis=(3, 4, 5)) / N_cams
    total_grad *= cloud_mask
    grad_norm = np.linalg.norm(total_grad)

    beta_opt -= step_size * total_grad

    # thresholding beta
    beta_opt[beta_opt < 0] = 0
    beta_opt[beta_opt > beta_max] = beta_max
    volume.set_beta_cloud(beta_opt)

    # loss calculation
    loss = 0.5 * np.sum(dif ** 2)
    print(f"loss = {loss}, grad_norm={grad_norm}")

    # Writing scalar and images to tensorboard
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, mean_dist, max_dist, rel_dist, grad_norm, iter)

