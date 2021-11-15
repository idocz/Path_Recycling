import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
# checkpoint_id = "0308-2000-39"
# iter = 12700

# checkpoint_id = "0808-1822-43"
# iter = 21800
checkpoint_id = "0811-1013-46"
iter = 6600
dir_to_save = join("data","res",f"{checkpoint_id}_{iter}.mat")
dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
betas = dict["betas"]
dict_gt = np.load(join("checkpoints",checkpoint_id,"data",f"gt.npz"))
betas_gt = dict_gt["betas"]
savemat(dir_to_save, {"vol": betas, "vol_gt":betas_gt})
# savemat(dir_to_save, {"vol": betas})
