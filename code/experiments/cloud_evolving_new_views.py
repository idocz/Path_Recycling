from utils import get_images_from_TB, animate
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

exp_name = "0305-1645-13"
exp_dir = join("checkpoints",exp_name)
img_list = get_images_from_TB(exp_dir, "simulated_images")
animate(img_list[:100], interval=60)
