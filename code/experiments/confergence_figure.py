import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pickle

exp_name = "0705-2057-19"
exp_dir = join("checkpoints",exp_name)
# iter = 100
# scene_rr = pickle.load(open(join(exp_dir,"data","checkpoint_loader"), "rb" )).recreate_scene()

output_dir = join("experiments","plots")
views = [0,3,8]
steps = [0,500,2000]
N = len(views)
M = len(steps) + 1
image_list = np.empty((N,M), dtype=np.object)
for i in range(N):
    view = views[i]
    gt_image, _ = get_images_from_TB(exp_dir, f"ground_truth/{view}")
    image_list[i, -1] = gt_image[-1]
    all_image_steps, wall_time_list = get_images_from_TB(exp_dir, f"simulated_images/{view}")
    for j in range(M-1):
        image_list[i, j] = all_image_steps[steps[j]]

plt.figure()
fig, axes = plt.subplots(N,M)
for i in range(N):
    for j in range(M):
        ax = axes[i,j]
        ax.imshow(np.rot90(image_list[i,j],k=1), cmap="gray")
        ax.axis("off")
        if i == 0:
            if j == M-1:
                ax.set_title("ground truth")
            elif j==0:
                ax.set_title("initialization")
            else:
                ax.set_title(f"{wall_time_list[j]} minutes")
fig.tight_layout()
plt.savefig(join(output_dir, f"convergence_grid.pdf"), bbox_inches='tight')

plt.show()




# I_gt =
# print("TB total images:", N_images)

# img_list_view0 = img_list_view0[steps]
# img_list_view1 = img_list_view0[steps]
# img_list_view2 = img_list_view0[steps]

# img_list = [np.hstack([im, I_gt]) for image0, image1, image2 \
#             in zip(img_list_view0,img_list_view1,img_list_view2)]
# animate(img_list, interval=30)
