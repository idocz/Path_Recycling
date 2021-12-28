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

#
max_scalar_recycling = 2627
max_scalar_regular = 1513

# max_scalar_recycling = 40000 //15
# max_scalar_regular = 18000 //15
exp_name_recycling = "2907-1847-02"
exp_dir_recycling = join("checkpoints",exp_name_recycling)
exp_name_regular = "0308-1253-59"
exp_dir_regular = join("checkpoints",exp_name_regular)
plot_functions = [plt.plot, plt.semilogy]
text_size = 15
tick_size = 12
text_locs = [((0.242, 0.236), (0.81,0.316)),((0.541, 8.24e-13), (0.98, 3.3e-12))]
scalar_name = "loss"

print(f"loading {scalar_name} scalars..")
scalar_list_recycling = get_scalars_from_TB(exp_dir_recycling, scalar_name)[:max_scalar_recycling]

scalar_list_regular = get_scalars_from_TB(exp_dir_regular, scalar_name)[:max_scalar_regular]

print("plotting..")
ref = scalar_list_recycling[0].wall_time
ts_rec = [(scalar.wall_time - ref)/3600 for scalar in scalar_list_recycling]
# steps_rec = [scalar.step for scalar in scalar_list_recycling]
values_rec = [scalar.value for scalar in scalar_list_recycling]

ref = scalar_list_regular[0].wall_time
ts_reg = [(scalar.wall_time - ref)/3600 for scalar in scalar_list_regular]
# steps_reg = [scalar.step for scalar in scalar_list_regular]
values_reg = [scalar.value for scalar in scalar_list_regular]


plt.figure(figsize=(8,4.5))
# plt.subplot(2,1,1)
plt.loglog(ts_rec, values_rec, "--", label="recycling (Nr=10)")
plt.loglog(ts_reg, values_reg, "--", label="traditional (Nr=1)",color='#F39C12')
h = plt.ylabel("loss",fontsize=text_size)
# plt.xlabel("runtime [hours]",fontsize=text_size)
plt.grid()
plt.legend(fontsize=text_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
# plt.savefig(join(output_dir,f"{scalar_name}_time.pdf"), bbox_inches='tight')
# plt.xlim(5e-3, np.max(ts_rec))
# plt.show()


# reflectometry
exp_name_recycling = "0805-1740-39"
exp_dir_recycling = "C:\\Users\\idocz\\Desktop\\repos\\Path_Recycling_Surfaces\\code\checkpoints\\0805-1740-39"
exp_name_regular = "0805-1745-00"
exp_dir_regular = "C:\\Users\\idocz\\Desktop\\repos\\Path_Recycling_Surfaces\\code\checkpoints\\0805-1745-00"
text_size = 15
tick_size = 12
print(f"loading {scalar_name} scalars..")
scalar_list_recycling = get_scalars_from_TB(exp_dir_recycling, scalar_name)

scalar_list_regular = get_scalars_from_TB(exp_dir_regular, scalar_name)

print("plotting..")
ref = scalar_list_recycling[0].wall_time
ts_rec_surf = [(scalar.wall_time - ref)/60 for scalar in scalar_list_recycling]
steps_rec_surf = [scalar.step for scalar in scalar_list_recycling]
values_rec_surf = [scalar.value for scalar in scalar_list_recycling]

ref = scalar_list_regular[0].wall_time
ts_reg_surf = [(scalar.wall_time - ref)/60 for scalar in scalar_list_regular]
steps_reg_surf = [scalar.step for scalar in scalar_list_regular]
values_reg_surf = [scalar.value for scalar in scalar_list_regular]

norm_scale = (values_reg[0]/values_reg_surf[0])
# norm_scale = 1
values_reg_surf=norm_scale * np.array(values_reg_surf)
values_rec_surf=norm_scale * np.array(values_rec_surf)
output_dir = join("experiments","plots")
# plt.subplot(2,1,2)
plt.loglog(ts_rec_surf, values_rec_surf, label="recycling")
plt.loglog(ts_reg_surf, values_reg_surf, label="regular")
# h = plt.ylabel(name, fontsize=text_size)
# if scalar_name == "relative_dist1":
#     h.set_rotation(0)
# plt.xlabel("r",fontsize=text_size)
plt.grid()
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.xlabel("")
plt.legend()
plt.tight_layout()
output_dir = "C:\\Users\\idocz\\OneDrive - Technion\\Thesis\\my paper\\figures\\ablation_studies"
# plt.savefig(join(output_dir, f"ablation_studies.pdf"), bbox_inches='tight')
plt.show()

np.savez(join("data",f"ablation_{scalar_name}.npz"), values_reg=values_reg, values_rec=values_rec, values_rec_surf=values_rec_surf,
         values_reg_surf= values_reg_surf, ts_reg=ts_reg, ts_rec=ts_rec, ts_rec_surf=ts_rec_surf,
         ts_reg_surf=ts_reg_surf)