import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from utils import relative_distance, mask_grader


scene = "smallcf"
if scene == "smallcf":
    res_name = "small coud field res - 6 stages l1 1 l 1 w air and ocean 9 sensors SC mask beta0 2 iter.mat"
else:
    res_name = "single cloud res - 8stages l1 1 l2 1 w air and ocean 9 sensors SC mask beta0 2_84.mat"



input_path = join("data","tamar_res",res_name)
res = loadmat(input_path)
iteration = res["iteration"][0,0]
cost_mean = res["cost_mean"]
runtime = res["runtime"] / 3600
plt.figure(figsize=(5.5,4))
# plt.loglog(np.linspace(1,iteration, iteration), cost_mean[0, :iteration], '--',  marker='o', markersize=5, color='blue')
plt.loglog(runtime[0,:iteration], cost_mean[0, :iteration])
plt.title('Loss', fontweight='bold')
plt.grid(True)
plt.xlim((0, runtime[0,iteration-1]))
plt.ylim(0,cost_mean.max())
plt.xlabel('hours')
plt.show()

beta = res["beta"]
beta_gt = res["beta_gt"]
dist = relative_distance(beta_gt, beta)
# dist = np.sum(np.sum(np.sum(abs(beta - beta_gt), 2), 1), 0) / np.sum(beta_gt)

print(dist)

cloud_mask = res["mask"].astype(bool)
