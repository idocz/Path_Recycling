import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.io import loadmat
checkpoint_id = "2212-1250-03"
iter = 22405
dir_to_save = join("data","vol3d","betas.mat")

dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
betas = dict["betas"]

betas_gt = loadmat(join("data", "rico.mat"))["beta"]


A = np.abs(betas_gt - betas)
B = np.abs(betas_gt)
print(f"relative error = {np.mean(A)/np.mean(B)}")

mask = betas_gt > 0.5
X = betas_gt[mask].reshape(-1)
Y = betas[mask].reshape(-1)
N = 500
rand_inds = np.random.randint(0,X.shape[0],N)
#
# plt.scatter(X[rand_inds], Y[rand_inds], s=3)
plt.scatter(X, Y, s=3)

plt.plot([0,127], [0,127])
plt.xlabel("GT")
plt.ylabel("opt")
plt.title(f"iter: {iter}")
plt.show()