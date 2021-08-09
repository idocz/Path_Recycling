import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from utils import relative_distance,relative_bias
from scipy.io import loadmat, savemat
# checkpoint_id = "0308-2000-39"
# iter = 12700

checkpoint_id = "0508-1325-20"
iter = 6900
dir_to_save = join("data","res",f"{checkpoint_id}_{iter}.mat")
dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
betas = dict["betas"]
dict_gt = np.load(join("checkpoints",checkpoint_id,"data",f"gt.npz"))
betas_gt = dict_gt["betas"]
savemat(dir_to_save, {"vol": betas, "vol_gt":betas_gt})

