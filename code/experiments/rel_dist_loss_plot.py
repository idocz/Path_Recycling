import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate, get_scalars_from_TB
from os.path import join
import numpy as np
import matplotlib.pyplot as plt




max_scalar_recycling = 2627
max_scalar_regular = 1513
exp_name_recycling = "2907-1847-02"
exp_dir_recycling = join("checkpoints",exp_name_recycling)
exp_name_regular = "0308-1253-59"
exp_dir_regular = join("checkpoints",exp_name_regular)
scalar_names = ["relative_dist1", "loss"]
names = ["relative error", "loss"]
plot_functions = [plt.plot, plt.semilogy]

for scalar_name, name, plot in zip(scalar_names, names, plot_functions):
    print(f"loading {scalar_name} scalars..")
    scalar_list_recycling = get_scalars_from_TB(exp_dir_recycling, scalar_name)[:max_scalar_recycling]

    scalar_list_regular = get_scalars_from_TB(exp_dir_regular, scalar_name)[:max_scalar_regular]

    print("plotting..")
    ref = scalar_list_recycling[0].wall_time
    ts_rec = [(scalar.wall_time - ref)/60 for scalar in scalar_list_recycling]
    steps_rec = [scalar.step for scalar in scalar_list_recycling]
    values_rec = [scalar.value for scalar in scalar_list_recycling]

    ref = scalar_list_regular[0].wall_time
    ts_reg = [(scalar.wall_time - ref)/60 for scalar in scalar_list_regular]
    steps_reg = [scalar.step for scalar in scalar_list_regular]
    values_reg = [scalar.value for scalar in scalar_list_regular]

    output_dir = join("experiments","plots")
    plt.figure()
    plot(ts_rec, values_rec, label="recycling (Nr=30)")
    plot(ts_reg, values_reg, label="traditional (Nr=1)")
    plt.ylabel(name)
    plt.xlabel("time (minutes)")
    plt.grid()
    plt.legend()
    plt.savefig(join(output_dir,f"{scalar_name}_time.pdf"), bbox_inches='tight')
    plt.show()

    plt.figure()
    plot(steps_rec, values_rec, label="recycling (Nr=30)")
    plot(steps_reg, values_reg, label="traditional (Nr=1)")
    plt.ylabel(name)
    plt.xlabel("Iteration")
    plt.grid()
    plt.legend()
    plt.savefig(join(output_dir,f"{scalar_name}_iteration.pdf"), bbox_inches='tight')
    plt.show()
