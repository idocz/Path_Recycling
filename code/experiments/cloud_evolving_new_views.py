import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

exp_name = "0305-1645-13"
exp_dir = join("checkpoints",exp_name)
frames = 600
img_list = get_images_from_TB(exp_dir, "simulated_images/0")
I_gt = get_images_from_TB(exp_dir, "ground_truth/0")[0]
N_images = len(img_list)
print("TB total images:", N_images)
skip = N_images//frames
img_list = img_list[::skip]
img_list = [np.hstack([image, I_gt]) for image in img_list]
animate(img_list, interval=30)
