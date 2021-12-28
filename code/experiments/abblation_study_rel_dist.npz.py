import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate, get_scalars_from_TB
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# output_dir = join("experiments", "plots")
output_dir = "C:\\Users\\idocz\OneDrive - Technion\\Thesis\\my paper\\figures\\loss_and_rel_dist_smoke"
res_dict = np.load(join("data","ablation_rel_dist.npz"))


scalar_name = "rell_dist"
values_rec = res_dict["values_rec"]
values_reg = res_dict["values_reg"]
ts_reg = res_dict["ts_reg"]
ts_rec = res_dict["ts_rec"]
values_rec_surf = res_dict["values_rec_surf"]
values_reg_surf = res_dict["values_reg_surf"]
ts_reg_surf = res_dict["ts_reg_surf"]
ts_rec_surf = res_dict["ts_rec_surf"]


print("plotting..")

text_size = 22
tick_size = 17
linewidth = 3.5
plt.figure(figsize=(8,4.5))
line = plt.semilogx(ts_reg, values_reg, linewidth=linewidth)
plt.semilogx(ts_rec, values_rec, label="scattering tomography",color="#F39C12", linewidth=linewidth)
plt.grid()

plt.xlim(5e-3,2)


# reflectometry

plt.semilogx(ts_rec_surf, values_rec_surf, "--", label="reflectometry",color="#F39C12", linewidth=linewidth)
plt.semilogx(ts_reg_surf, values_reg_surf, "--", color=line[0].get_color(), linewidth=linewidth)

plt.grid()
plt.legend()
plt.tight_layout()
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
leg = plt.legend(fontsize=text_size)
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('black')
output_dir = "C:\\Users\\idocz\\OneDrive - Technion\\Thesis\\my paper\\figures\\ablation_studies"
plt.savefig(join(output_dir,f"ablation_studies_rel_dist.pdf"), bbox_inches='tight')
plt.show()

np.savez(join("data","ablation.npz"), values_reg=values_reg, values_rec=values_rec, values_rec_surf=values_rec_surf,
         values_reg_surf= values_reg_surf, ts_reg=ts_reg, ts_rec=ts_rec, ts_rec_surf=ts_rec_surf,
         ts_reg_surf=ts_reg_surf)