import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import pickle
from classes.camera import Camera
from tqdm import tqdm
from numba import cuda
cuda.select_device(2)

def transfrom_image(img, gamma):
    img = (img-img.min())/(img.max()-img.min())
    return img**gamma

exp_name = "2907-1847-02"
exp_dir = join("checkpoints",exp_name)

# scene_name = "smoke"
scene_name = "smallcf"

cp = pickle.load(open(join(exp_dir,"data","checkpoint_loader"), "rb" ))
scene_rr = cp.recreate_scene()
# scene_rr.rr_depth = 10
# scene_rr.g_cloud = 0.5

bbox = scene_rr.volume.grid.bbox
if scene_name == "smoke":
    ## smoke ##
    # steps = [0,5000,10000,26000] # for smoke
    steps = [0,5000,26000] # for smoke
    views = [3,7]
    focal_factor = 1.9
    volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.65

elif scene_name == "cloud1":
    # cloud1
    # steps = [0,2000,7000,12500] # for cloud1
    steps = [0,1000,3000,10000] # for cloud1
    views = [3,7]
    focal_factor = 0.8
    volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.8



betas_gt = np.load(join(exp_dir,"data","gt.npz"))["betas"]
betas_list = [np.load(join(exp_dir,"data",f"opt_{step}.npz"))["betas"] for step in steps]
wall_time_list = [int(np.load(join(exp_dir,"data",f"opt_{step}.npz"))["time"]//60) for step in steps]
betas_list.append(betas_gt)
output_dir = join("experiments","plots")

# Adding New view


voxel_size_z = 0.04
edge_z = voxel_size_z * betas_list[0].shape[2]
cameras = [scene_rr.cameras[view] for view in views]
height_factor = 1.5
focal_length = 50e-3
ps = 80
sensor_size = np.array((47e-3, 47e-3)) / height_factor
R = height_factor * edge_z
theta = 90
phi = (-(9 // 2) + 5  ) * 20
thera_rad = theta * (np.pi / 180)
phi_rad = phi * (np.pi / 180)
t = R * theta_phi_to_direction(thera_rad, phi_rad) + volume_center
t[2] -= 0.55
euler_angles = np.array((theta, 0, phi - 90))
camera = Camera(t, euler_angles, cameras[0].focal_length*focal_factor,  cameras[0].sensor_size,  cameras[0].pixels)
cameras.append(camera)
for camera in cameras:
    camera = Camera(camera.t, camera.euler_angles, camera.focal_length*focal_factor,  camera.sensor_size,  np.array([ps,ps]))
scene_rr.reset_cameras(cameras)

N = len(cameras)
M = len(steps) + 1
image_list = np.empty((N,M), dtype=np.object)
# Np_gt = cp.Np_gt
Np_gt = int(2e7)

for j in tqdm(range(M)):
    scene_rr.volume.beta_cloud = betas_list[j]
    cuda_paths = scene_rr.build_paths_list(Np_gt)
    I = scene_rr.render(cuda_paths)
    del(cuda_paths)
    # I = np.zeros(shape=(N,*cameras[0].pixels))
    for i in range(N):
        image_list[i,j] = I[i]

scale = 1
fig, axes = plt.subplots(M,N)
fig.set_size_inches(N*scale*1.05,M*scale, forward=True)
for j in range(N):
    for i in range(M):
        img = image_list[j,i]
        ax = axes[i,j]
        # if i != N-1:
        img = np.rot90(img,k=1)
        # img = transfrom_image(img, 1.3)
        # else:
        #     img = np.rot90(img,k=2)
        ax.imshow(img, cmap="gray")
        # ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        if j == 0:
            if i == M-1:
                # ax.set_title("ground truth")
                ax.set_ylabel("ground truth")
            elif i==0:
                # ax.set_title("initialization")
                ax.set_ylabel("initialization")
            else:
                # ax.set_title(f"{wall_time_list[j]} minutes")
                ax.set_ylabel(f"{wall_time_list[i]} minutes")
        # if j == 0:
        #     if i != N-1:
        #         ax.set_ylabel(f"view {i+1}")
        #     else:
        #         ax.set_ylabel(f"new view")
# fig.tight_layout()
plt.subplots_adjust(left=0.075,
                    bottom=0.005,
                    right=0.99,
                    top=0.995,
                    wspace=0.05,
                    hspace=0.05)
plt.savefig(join(output_dir, f"{scene_name}_convergence_grid_transposed.pdf"), bbox_inches='tight')

plt.show()




# I_gt =
# print("TB total images:", N_images)

# img_list_view0 = img_list_view0[steps]
# img_list_view1 = img_list_view0[steps]
# img_list_view2 = img_list_view0[steps]

# img_list = [np.hstack([im, I_gt]) for image0, image1, image2 \
#             in zip(img_list_view0,img_list_view1,img_list_view2)]
# animate(img_list, interval=30)
