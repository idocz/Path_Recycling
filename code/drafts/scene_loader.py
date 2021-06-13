import pickle
from classes.deprecated.scene_numba import *
from os.path import join

checkpoint_id = "1612-0025-00"
iter = 2345
dir_to_save = join("data","vol3d","betas.mat")

dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
betas = dict["betas"]
print(betas.mean())

scene_numba = pickle.load(open( "save.p", "rb" ))
print(scene_numba)
scene_numba.volume.set_beta_cloud(betas)

paths = scene_numba.build_paths_list(int(1e6), 15)
I_total, total_grad = scene_numba.render(paths, differentiable=True)
print(np.linalg.norm(I_total), np.linalg.norm(total_grad))