import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate, get_scalars_from_TB
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pytorch


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# output_dir = join("experiments", "plots")
output_dir = "C:\\Users\\idocz\OneDrive - Technion\\Thesis\\my paper\\figures\\shdom_comp"
checkpoint_id_our = "1511-1750-31"
checkpoint_id_shdom =  join("log_name-23-Nov-2021-111438","loss","Data term loss")
exp_dir_our = join("checkpoints",checkpoint_id_our)
exp_dir_shdom = join("checkpoints",checkpoint_id_shdom)
iter = 700
#
plot_functions = [plt.plot, plt.semilogy]
text_size = 15
tick_size = 12
scalar_name_our = "loss"
scalar_name_shdom = "loss"

print(f"loading {scalar_name_our} scalars..")
scalar_list_our = get_scalars_from_TB(exp_dir_our, scalar_name_our)
scalar_list_shdom = get_scalars_from_TB(exp_dir_shdom, scalar_name_shdom)

print("plotting..")
ref = scalar_list_our[0].wall_time
ts_our = [(scalar.wall_time - ref)/60 for scalar in scalar_list_our]
values_our = np.array([scalar.value for scalar in scalar_list_our])

ref = scalar_list_shdom[0].wall_time
ts_shdom = [(scalar.wall_time - ref)/60 for scalar in scalar_list_shdom]
values_shdom = np.array([scalar.value for scalar in scalar_list_shdom])

values_our /=values_our[0]
values_shdom /=values_shdom[0]

plt.figure(figsize=(8,4.5))
# plt.subplot(2,1,1)
plt.semilogy(ts_our, values_our, label="Our")
plt.semilogy(ts_shdom, values_shdom, label="Shdom")
h = plt.ylabel("loss",fontsize=text_size)
plt.xlabel("runtime [hours]",fontsize=text_size)
plt.grid()
plt.legend(fontsize=text_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
# plt.savefig(join(output_dir,f"{scalar_name}_time.pdf"), bbox_inches='tight')
# plt.xlim(0, 1)
plt.show()


#
# np.savez(join("data",f"shdom_comp_{scalar_name_our}.npz"), ts_our=ts_our, ts_shdom=ts_shdom, values_our=values_our,
#          values_shdom=values_shdom)