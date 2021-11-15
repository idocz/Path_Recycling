import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from utils import relative_distance,relative_bias
from scipy.io import loadmat, savemat
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# csfont = {'fontname':'Times New Roman'}
seed = 125
scene = "smallcf"
# scene = "jplext"
if scene == "smallcf":
    # checkpoint_id = "0808-1822-43"
    # iter = 21800
    # checkpoint_id = "0908-1258-43"
    # iter = 28900
    checkpoint_id = "0711-2025-59"
    iter = 17000
    N=0.1
    seed = 125

elif scene == "jplext":
    # checkpoint_id = "0308-2000-39"
    # iter = 8100
    checkpoint_id = "0711-2006-49"
    iter = 7000
    seed = 110
    # iter = 4300 #mid
    # checkpoint_id = "0908-1509-28"
    # iter = 7200

    N = 0.3
elif scene == "smoke":
    # checkpoint_id = "0508-1125-06"
    # iter = 22000
    iter = 9600 # mid
    # checkpoint_id = "2907-1847-02"
    # iter = 26000
    N=0.1
np.random.seed(seed)
dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
betas = dict["betas"]
dict = np.load(join("checkpoints",checkpoint_id,"data",f"gt.npz"))
betas_gt = dict["betas"]
mask = betas >= 0


print(f"relative error = {relative_distance(betas_gt,betas)}")
print(f"relative bias = {relative_bias(betas_gt,betas)}")

Y = betas_gt[mask].reshape(-1)
X = betas[mask].reshape(-1)
max_val = np.max([X.max(), Y.max()])
N = int(Y.shape[0] * N)
# N = 1500
print()
rand_inds = np.random.randint(0,X.shape[0],N)
fig = plt.figure(figsize=(5,5))
plt.scatter(X[rand_inds], Y[rand_inds])
# plt.scatter(X, Y, s=3)

plt.plot([0,max_val], [0,max_val], color="red")
# plt.plot([0,10], [0,10])
fs = 20
# plt.ylabel(r"$\beta^{c}_{gt}}$", fontsize=fs)
# plt.xlabel(r"$\hat{\beta}^{c}}$", fontsize=fs)
# plt.title(f"iter: {iter}")
plt.xticks([])
plt.yticks([])
plt.axes().set_aspect('equal')
plt.tight_layout()
plt.savefig(join("experiments","plots",f"{checkpoint_id}_{iter}_{scene}_scatter_plot.pdf"))
plt.show()

dir_to_save = join("data","res",f"{checkpoint_id}_{iter}.mat")
savemat(dir_to_save, {"vol": betas, "vol_gt":betas_gt})
