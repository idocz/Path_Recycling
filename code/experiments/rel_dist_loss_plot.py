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


max_scalar_recycling = 2627
max_scalar_regular = 1513
exp_name_recycling = "2907-1847-02"
exp_dir_recycling = join("checkpoints",exp_name_recycling)
exp_name_regular = "0308-1253-59"
exp_dir_regular = join("checkpoints",exp_name_regular)
scalar_names = ["relative_dist1", "loss"]
names = [r"$\epsilon$", "loss"]
plot_functions = [plt.plot, plt.semilogy]
text_size = 15
tick_size = 12
text_locs = [((0.242, 0.236), (0.81,0.316)),((0.541, 8.24e-13), (0.98, 3.3e-12))]
for scalar_name, name, plot, text_loc in zip(scalar_names, names, plot_functions, text_locs):
    print(f"loading {scalar_name} scalars..")
    scalar_list_recycling = get_scalars_from_TB(exp_dir_recycling, scalar_name)[:max_scalar_recycling]

    scalar_list_regular = get_scalars_from_TB(exp_dir_regular, scalar_name)[:max_scalar_regular]

    print("plotting..")
    ref = scalar_list_recycling[0].wall_time
    ts_rec = [(scalar.wall_time - ref)/3600 for scalar in scalar_list_recycling]
    steps_rec = [scalar.step for scalar in scalar_list_recycling]
    values_rec = [scalar.value for scalar in scalar_list_recycling]

    ref = scalar_list_regular[0].wall_time
    ts_reg = [(scalar.wall_time - ref)/3600 for scalar in scalar_list_regular]
    steps_reg = [scalar.step for scalar in scalar_list_regular]
    values_reg = [scalar.value for scalar in scalar_list_regular]


    plt.figure()
    plot(ts_rec, values_rec, label="recycling (Nr=30)")
    plot(ts_reg, values_reg, label="traditional (Nr=1)")
    h = plt.ylabel(name,fontsize=text_size)
    if scalar_name == "relative_dist1":
        h.set_rotation(0)
    plt.xlabel("runtime [hours]",fontsize=text_size)
    plt.grid()
    # plt.legend(fontsize=text_size)
    plt.text(*text_loc[0], r"recycling ($N_r=30$)", fontsize=text_size)
    plt.text(*text_loc[1], r"traditional ($N_r=1$)", fontsize=text_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.savefig(join(output_dir,f"{scalar_name}_time.pdf"), bbox_inches='tight')
    # plt.xlim(5e-3, np.max(ts_rec))
    plt.show()
    #
    # plt.figure()
    # plot(steps_rec, values_rec, label="recycling (Nr=30)")
    # plot(steps_reg, values_reg, label="traditional (Nr=1)")
    # plt.ylabel(name)
    # plt.xlabel("Iteration")
    # plt.grid()
    # plt.legend(fontsize=text_size)
    # plt.xticks(fontsize=tick_size)
    # plt.yticks(fontsize=tick_size)
    # plt.savefig(join(output_dir,f"{scalar_name}_iteration.pdf"), bbox_inches='tight')
    # plt.show()
