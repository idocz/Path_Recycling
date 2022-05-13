import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate, get_scalars_from_TB
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.font_manager._rebuild()
from experiments.ablation_table import exp_dict_jplext, exp_dict_smallcf, get_zero_after_dot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']





scalar_name = "relative_dist1"
long_running_time = 2

text_size = 22
tick_size = 17
output_dir = join("experiments","plots")
figsize = (6,3.5)

def plot_exp(exp_name, ablation):
    if exp_name == "smallcf":
        plt.legend(fontsize=tick_size)
        plt.xticks(np.array([0, 1, 2, 3, 4, 5, 6, 7]), fontsize=tick_size)
        plt.yticks(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1]), fontsize=tick_size)
        # plt.ylim(0.48,0.9)
        # plt.xlim(0,6.5)
        plt.grid()
        plt.tight_layout()
        plt.savefig(join(output_dir,"ablation", f"{exp_name}_sort_{ablation}.pdf"), bbox_inches='tight')
        plt.show()
    else:
        plt.legend(fontsize=tick_size)
        plt.xticks(np.array([0, 0.5, 1, 1.5, 2]), fontsize=tick_size)
        plt.yticks(np.array([0.5, 0.6, 0.7, 0.8, 0.9]), fontsize=tick_size)
        # plt.ylim(0.48,0.9)
        plt.grid()
        plt.tight_layout()
        plt.savefig(join(output_dir,"ablation", f"{exp_name}_{ablation}.pdf"), bbox_inches='tight')
        plt.show()


d = exp_dict_smallcf
# for d in dicts:

exp_names = ["jplext", "smallcf"]
for exp_id, d in enumerate([exp_dict_jplext, exp_dict_smallcf]):
    # SORT ABLATION
    Nr = 10
    exps = [(Nr, True), (Nr, False)]
    plt.figure(figsize=figsize)
    for exp in exps:
        name = d[exp]
        if name is None:
            continue
        exp_dir = join("checkpoints",name)
        scalar_list = get_scalars_from_TB(exp_dir, scalar_name)
        ref = scalar_list[0].wall_time
        time_axis = np.array([(scalar.wall_time - ref) / 3600 for scalar in scalar_list])
        value_axis = np.array([scalar.value for scalar in scalar_list])
        cond = value_axis>=0.5
        plt.plot(time_axis[cond],value_axis[cond],label=f"Nr={exp[0]} (sort={exp[1]})")
        print("takes:",time_axis[cond][-1])


    # plt.xlabel("runtime[hours]",fontsize=text_size)
    # plt.ylabel(r"$\epsilon$")
    plot_exp(exp_names[exp_id],"sort")


    #NR ABLATION
    exps = [(1,False),(2,True), (10,True), (30,True)]
    plt.figure(figsize = figsize)
    for exp in exps:
        name = d[exp]
        if name is None:
            continue
        exp_dir = join("checkpoints",name)
        scalar_list = get_scalars_from_TB(exp_dir, scalar_name)
        ref = scalar_list[0].wall_time
        time_axis = np.array([(scalar.wall_time - ref) / 3600 for scalar in scalar_list])
        value_axis = np.array([scalar.value for scalar in scalar_list])
        cond = value_axis >= 0.5
        plt.plot(time_axis[cond],value_axis[cond],label=f"Nr={exp[0]} (sort={exp[1]})")

    plot_exp(exp_names[exp_id],"Nrs")



