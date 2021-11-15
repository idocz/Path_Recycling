import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from utils import relative_distance,relative_bias
from scipy.io import loadmat, savemat
# checkpoint_id = "0108-1031-34"
# iter = 12500
checkpoint_id = "0811-1013-46"
iter = 6600
dir_to_save = join("data","res",f"{checkpoint_id}_{iter}.mat")
dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
betas = dict["betas"]
dict = np.load(join("checkpoints",checkpoint_id,"data",f"gt.npz"))
# betas_gt = loadmat(join("data", "rico.mat"))["beta"]
betas_gt = dict["betas"]
savemat(dir_to_save, {"vol": dict["betas"]})

print(f"relative error = {relative_distance(betas_gt,betas)}")
print(f"relative bias = {relative_bias(betas_gt,betas)}")

mask = betas_gt > 0
X = betas_gt[mask].reshape(-1)
Y = betas[mask].reshape(-1)
N = 3000
rand_inds = np.random.randint(0,X.shape[0],N)
#
plt.scatter(X[rand_inds], Y[rand_inds], s=3)
# plt.scatter(X, Y, s=3)

plt.plot([0,127], [0,127], color="red")
# plt.plot([0,10], [0,10])
plt.xlabel("GT")
plt.ylabel("opt")
plt.title(f"iter: {iter}")
plt.show()