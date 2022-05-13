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
# scene= "smoke"
# scene = "smallcf"
scene = "jplext"
from experiments.ablation_table import exp_dict_jplext, exp_dict_smallcf
if scene == "smallcf":
    # checkpoint_id = "0808-1822-43"
    # iter = 21800
    # checkpoint_id = "0908-1258-43"
    # iter = 28900
    # checkpoint_id = "0711-2025-59"
    # iter = 17000
    # checkpoint_id = "3101-1529-15_smallcf_Nr=10_ss=5.00e+09"
    # iter = 29300
    checkpoint_id = exp_dict_smallcf[(10, True)]
    iter = 9300
    N=0.08
    seed = 150


elif scene == "jplext":
    # checkpoint_id = "0308-2000-39"
    # iter = 8100
    # checkpoint_id = "0711-2006-49"
    # iter = 7000
    # seed = 110
    # iter = 4300 #mid
    # checkpoint_id = "0908-1509-28"
    # iter = 7200
    # checkpoint_id = "2801-0932-51_Nr=10"
    # iter = 4500
    # checkpoint_id = exp_dict_jplext[(10,True)]
    checkpoint_id = "0503-1103-23_jplext_Nr=10_ss=3.0e+09_tosort=1"
    iter = 3140

    seed = 135

    N = 0.3
elif scene == "smoke":
    # checkpoint_id = "0508-1125-06"
    checkpoint_id = "0503-0955-58_smoke_Nr=10_ss=2.00e+11"
    # iter = 22000
    iter = 150 # mid
    # checkpoint_id = "2907-1847-02"
    # iter = 26000
    N=0.1

print(f"{checkpoint_id}_{iter}.mat")
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
fs = 18
# plt.ylabel(r"$\beta^{c}_{gt}}$", fontsize=fs)
# plt.xlabel(r"$\hat{\beta}^{c}}$", fontsize=fs)
# plt.title(f"iter: {iter}")
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
# plt.yticks([])
plt.axes().set_aspect('equal')
plt.tight_layout()
plt.savefig(join("experiments","plots",f"{checkpoint_id}_{iter}_{scene}_scatter_plot.pdf"))
plt.savefig(join("experiments","plots",f"{checkpoint_id}_{iter}_{scene}_scatter_plot.png"))
plt.show()

dir_to_save = join("data","res",f"{checkpoint_id}_{iter}.mat")
savemat(dir_to_save, {"vol": betas, "vol_gt":betas_gt})
