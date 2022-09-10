import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate, get_scalars_from_TB
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


# exp_names = [
# # "2801-1921-07_Nr=1_ss=2.50e+09",
#              "2801-0030-37_Nr=1",
#              "2801-0331-22_Nr=2",
#              "2801-0632-04_Nr=5",
#              "2801-0932-51_Nr=10",
#              "2801-1233-34_Nr=15_ss=2.50e+09 ",
#              "2801-1534-15_Nr=20_ss=2.50e+09 ",
#              "2901-1002-39_Nr=30_ss=2.50e+09"
#
#              ]
# exp_names = ["2602-0128-26_jplext_Nr=1_ss=2.0e+09_tosort=1",
#              "2602-0331-31_jplext_Nr=2_ss=2.0e+09_tosort=1",
#              # "2602-0534-36_jplext_Nr=5_ss=2.0e+09_tosort=1",
#              "2602-0737-36_jplext_Nr=10_ss=2.0e+09_tosort=1",
#              "2602-1143-37_jplext_Nr=30_ss=2.0e+09_tosort=1"
#              ]

exp_names = ["2702-0803-21_smallcf_Nr=1_ss=5.00e+09_tosort=1",
             "2502-0400-56_smallcf_Nr=2_ss=5.00e+09",
             "2702-1306-45_smallcf_Nr=5_ss=5.00e+09_tosort=1",
             "2402-1249-05_smallcf_Nr=10_ss=5.00e+09",
             "2502-1745-41_smallcf_Nr=30_ss=5.00e+09_tosort=1",
             ]
scalar_name = "relative_dist1"
get_label = lambda a:a.split("=")[1].split("_")[0]
results = {}
Nrs = [int(get_label(name)) for name in exp_names]
Nrs_plot = [1,2,5,10,30]
for name in exp_names:

    print("extracting",name)
    exp_dir = join("checkpoints",name)
    scalar_list = get_scalars_from_TB(exp_dir, scalar_name)
    ref = scalar_list[0].wall_time
    time_axis = np.array([(scalar.wall_time - ref) / 3600 for scalar in scalar_list])
    value_axis = np.array([scalar.value for scalar in scalar_list])
    cond = value_axis >= 0.6
    results[name] = (time_axis[cond],value_axis[cond])


max_time = []
text_size = 22
tick_size = 17
output_dir = join("experiments","plots")
figsize = (6,3.5)
plt.figure(figsize=figsize)
for name in exp_names:
    if int(get_label(name)) in Nrs_plot:
        # plt.semilogx(*results[name],label=get_label(name))
        plt.plot(*results[name],label=get_label(name))
# plt.xlabel("runtime[hours]",fontsize=text_size)
# plt.ylabel(r"$\epsilon$")
plt.legend(fontsize=tick_size)
plt.xticks(np.array([0,0.5,1,1.5,2]),fontsize=tick_size)
plt.yticks(np.array([0.5,0.6,0.7,0.8,0.9]),fontsize=tick_size)
# plt.ylim(0.48,0.9)
plt.grid()
plt.tight_layout()
plt.savefig(join(output_dir,f"eps_vs_times_Nrs.pdf"), bbox_inches='tight')
plt.show()


epss = [res[1][-1] for res in results.values()]
figsize = (4,3.5)
plt.figure(figsize=figsize)
plt.plot(Nrs,epss)
# plt.xlabel(r"$N_r$")
# plt.ylabel(scalar_name)
plt.xticks(Nrs, fontsize=tick_size)
ytick = np.arange(48,63,4)/100
plt.yticks(ytick,fontsize=tick_size)
# plt.yticks(fontsize=tick_size)
# plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(join(output_dir,f"final_vs_err.pdf"), bbox_inches='tight')
plt.show()