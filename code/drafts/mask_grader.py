import os, sys

my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene import *
from classes.scene_gpu import *
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
from scipy.ndimage.morphology import binary_dilation
from scipy import ndimage


beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
beta_cloud = beta_cloud.astype(float_reg)
cloud_mask_real = beta_cloud > 0
cloud_mask = np.load(join("data","Rico_mask_2.npy"))
print(f"accuracy:", np.mean(cloud_mask == cloud_mask_real))
print(f"fp:", np.mean((cloud_mask == 1)*(cloud_mask_real==0)))
fn = (cloud_mask == 0)*(cloud_mask_real==1)
print(f"fn:", np.mean(fn))
fn_exp = (fn * beta_cloud).reshape(-1)
print(f"fn_exp mean:", np.mean(fn_exp))
print(f"fn_exp max:", np.max(fn_exp))
print(f"fn_exp min:", np.min(fn_exp[fn_exp!=0]))
plt.hist(fn_exp[fn_exp!=0])
print("missed beta:",np.sum(fn_exp)/np.sum(beta_cloud))
print("rel_dit:",relative_distance(beta_cloud, beta_cloud*cloud_mask))
plt.show()
# kernel = ndimage.generate_binary_structure(2, )
# cloud_mask = binary_dilation(cloud_mask, structure=kernel)
cloud_mask = binary_dilation(cloud_mask)
