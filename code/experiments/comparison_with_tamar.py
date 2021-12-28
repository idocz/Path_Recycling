import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from utils import relative_distance, mask_grader, get_scalars_from_TB
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# scene = "smallcf"
scene = "jplext"
if scene == "smallcf":
    tamar_exp = "small coud field res - 6 stages l1 1 l 1 w air and ocean 9 sensors SC mask beta0 2 iter.mat"
    # checkpoint_id = "0808-1822-43"
    # iter = 21800
    checkpoint_id = "0711-2025-59"
    iter = 17840
    # iter = 1000

elif scene == "jplext":
    tamar_exp = "single cloud res - 8stages l1 1 l2 1 w air and ocean 9 sensors SC mask beta0 2_84.mat"
    checkpoint_id = "0711-2006-49"
    iter = 7000
# output_dir = join("experiments","plots")
output_dir = "C:\\Users\\idocz\OneDrive - Technion\\Thesis\\my paper\\figures\\comparison_tamar_SUP"

text_size = 13
tick_size = 17
# Loading Tamars results
input_path = join("data","tamar_res",tamar_exp)
tamar_res = loadmat(input_path)
iteration = tamar_res["iteration"][0,0]
cost_mean = tamar_res["cost_mean"]
runtime = tamar_res["runtime"] / 3600
plt.figure(figsize=(8,4.5))
# plt.loglog(np.linspace(1,iteration, iteration), cost_mean[0, :iteration], '--',  marker='o', markersize=5, color='blue')
plt.semilogy(runtime[0,:iteration], cost_mean[0, :iteration], label="[Loeub et al.2020]")
# plt.title('Loss vs Time', fontweight='bold')
plt.grid(True)
# plt.xlim(5e-3, runtime[0,iteration-1])
# plt.ylim(0,cost_mean.max())
# plt.xlabel("runtime [hours]", fontsize=text_size)
# plt.tight_layout()
# plt.xticks(fontsize=tick_size)
# plt.yticks(fontsize=tick_size)
# plt.savefig(join(output_dir,f"{scene}_loss_tamars.pdf"), bbox_inches='tight')
# plt.show()


# Loading my results
max_scalar = 1785
# min_scalar = 100
exp_dir_recycling = join("checkpoints",checkpoint_id)
scalar_names = ["loss"]
names = ["loss"]
# plt.figure(figsize=(5.5,4))
print(f"loading loss scalars..")
scalar_list = get_scalars_from_TB(exp_dir_recycling, "loss")[:max_scalar]
print("plotting..")
ref = scalar_list[0].wall_time
ts_rec = [(scalar.wall_time - ref)/3600 for scalar in scalar_list]
steps_rec = [scalar.step for scalar in scalar_list]
values_rec = [scalar.value for scalar in scalar_list]
values_rec = np.array(values_rec)
values_rec *= (cost_mean.max()/values_rec.max())
plt.semilogy(ts_rec, values_rec, label="Ours")
# plt.ylabel(name)
# plt.xlabel("runtime [hours]", fontsize=text_size)
plt.grid()
# plt.title('Loss vs Time', fontweight='bold')
# plt.xlim(5e-3, np.max(ts_rec))
# plt.legend(fontsize=text_size)
plt.tight_layout()
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
if scene == "jplext":
    plt.xlim(0,6.5)
# fontsize = 13
# plt.text(1.7, 0.03, "Loeub et al.2020", fontsize=fontsize)
# plt.text(0.38, 0.013, "Ours", fontsize=fontsize)
plt.savefig(join(output_dir,f"{scene}_loss_both_{scene}.pdf"), bbox_inches='tight')
plt.show()

print(np.max(ts_rec))
