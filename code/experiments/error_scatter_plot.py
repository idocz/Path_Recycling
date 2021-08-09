import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from utils import relative_distance,relative_bias
from scipy.io import loadmat, savemat
checkpoint_id = "0508-1926-42"
iter = 18600
dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
betas = dict["betas"]
dict = np.load(join("checkpoints",checkpoint_id,"data",f"gt.npz"))
betas_gt = dict["betas"]
mask = betas >= 0


print(f"relative error = {relative_distance(betas_gt,betas)}")
print(f"relative bias = {relative_bias(betas_gt[mask],betas[mask])}")

X = betas_gt[mask].reshape(-1)
Y = betas[mask].reshape(-1)
max_val = np.max([X.max(), Y.max()])
N = 30000
rand_inds = np.random.randint(0,X.shape[0],N)
plt.scatter(X[rand_inds], Y[rand_inds], s=3)
# plt.scatter(X, Y, s=3)

plt.plot([0,max_val], [0,max_val], color="red")
# plt.plot([0,10], [0,10])
fs = 20
plt.xlabel("ground truth", fontsize=fs)
plt.ylabel("our reconstruction", fontsize=fs)
# plt.title(f"iter: {iter}")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(join("experiments","plots",f"{checkpoint_id}_scatter_plot.pdf"))
plt.show()