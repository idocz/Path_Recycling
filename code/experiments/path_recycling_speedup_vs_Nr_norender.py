import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene_seed_NEgrad import SceneSeed
from classes.grid import Grid
# from classes.scene_rr_noNEgrad import *
# from classes.scene_rr import *
from classes.camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from classes.checkpoint_wrapper import CheckpointWrapper
from time import time
from classes.optimizer import *
from os.path import join
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
speed_ups = [[],[]]
max_Nr= 30
Nrs = np.arange(1,max_Nr)
sampling_times = [5.515347003936768, 11.966343879699707]
rendering_times = [4.929884910583496,1.1232147216796875]

for to_sort in [0, 1]:
    print("to_sort=", to_sort)
    print("rendering took", rendering_times[to_sort])
    for Nr in Nrs:
        speed_up = (sampling_times[0] + rendering_times[0]) / ((sampling_times[to_sort] / Nr) + rendering_times[to_sort])
        speed_ups[to_sort].append(speed_up)
        if Nr == 10 and to_sort == 1:
            print("speedup 10: ",speed_up)
    print()
# plt.figure()
text_size = 22
tick_size = 17
plt.figure(figsize=(4,4))
output_dir = join("experiments","plots")
plt.plot(Nrs, speed_ups[0], label="no sorting")
plt.plot(Nrs, speed_ups[1], label="sorting")
plt.xlim(1,max_Nr)
# plt.xlabel(r"$N_r$",fontsize=text_size)
# plt.ylabel("speed up",fontsize=text_size)
xticks = np.arange(0,max_Nr+1,5)
xticks[0]+=1
plt.xticks(xticks, fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.legend(fontsize=tick_size)
plt.grid()
plt.tight_layout()
plt.savefig(join(output_dir,f"speedup_vs_Nr.pdf"), bbox_inches='tight')
plt.show()