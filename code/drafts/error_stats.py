import numpy as np
from os.path import join
import matplotlib.pyplot as plt

checkpoint_id = "1012-0956-38"
iter = 1676
dir_to_save = join("data","vol3d","betas.mat")

dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
betas = dict["betas"]

betas_gt = np.load(join("checkpoints",checkpoint_id,"data","gt.npz"))["betas"]

A = betas_gt - betas
print(np.linalg.norm(A)/np.linalg.norm(betas_gt))

plt.scatter(betas_gt.reshape(-1), betas.reshape(-1))
plt.plot([0,127], [0,127])
plt.xlabel("GT")
plt.ylabel("opt")
plt.title(f"iter: {iter}")
plt.show()