import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate, get_scalars_from_TB
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

exp_name_recycling = "0505-1144-28"
exp_dir = join("checkpoints",exp_name_recycling)
scalar_list_recycling = get_scalars_from_TB(exp_dir, "relative_dist1")
exp_name_regular = "0505-1152-02"
exp_dir = join("checkpoints",exp_name_regular)
scalar_list_regular= get_scalars_from_TB(exp_dir, "relative_dist1")

plt.figure()
label = "recycling"
for scalar_list in [scalar_list_recycling, scalar_list_regular]:
    ref = scalar_list[0].wall_time
    ts = [scalar.wall_time - ref for scalar in scalar_list]
    steps = [scalar.step for scalar in scalar_list]
    values = [scalar.value for scalar in scalar_list]
    plt.plot(ts, values, label=label)
    plt.legend()
    plt.grid()
    label = "regular"



plt.show()

plt.figure()
plt.plot(steps, values)
plt.show()