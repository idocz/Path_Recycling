import sys
from os.path import join
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
from classes.scene_airmspi_backward import *
from classes.camera import AirMSPICamera
from classes.visual import *
from time import time
from utils import *
from cuda_utils import *
import pickle
cuda.select_device(0)

sun_intensity = 1
downscale = 8
a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
airmspi_data = pickle.load(a_file)
beta_cloud = loadmat(join("data", "FINAL_3D_extinction.mat"))["estimated_extinction"][:,:,:,0]
cloud_mask = airmspi_data["cloud_mask"]
camera_array_list = airmspi_data["camera_array_list"]
camera_array_list = [np.ascontiguousarray(camera_array[::downscale, ::downscale, :6]) for camera_array in camera_array_list]
resolutions = airmspi_data["resolution"]
# beta_cloud = np.zeros(airmspi_data["grid_shape"], dtype=float_reg) * 2
# beta_cloud[cloud_mask] = 2


zenith = airmspi_data["sun_zenith"]
azimuth = airmspi_data["sun_azimuth"]
dir_x = -(np.sin(zenith)*np.cos(azimuth))
dir_y = -(np.sin(zenith)*np.sin(azimuth))
dir_z = -np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
#####################
# grid parameters #
#####################
grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
print(grid.bbox)




#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004

print(beta_cloud.shape)
w0_air = 0.912
w0_cloud = 0.99


# Declerations
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
g_cloud = 0.85
#######################
# Cameras declaration #
#######################


Np = int(1e7)
rr_depth = 20
rr_stop_prob = 0.05

# volume.set_mask(cloud_mask)
volume.set_mask(beta_cloud>0)
scene_airmspi = SceneAirMSPI(volume, camera_array_list, sun_direction, sun_intensity, g_cloud, rr_depth, rr_stop_prob)

N_cams = 9
I_gt = airmspi_data["images"]
I_total_pad = np.zeros((N_cams, *scene_airmspi.pixels_shape), dtype=float_reg)
I_gt_pad = np.ones((N_cams, *scene_airmspi.pixels_shape), dtype=float_reg) * np.min(I_gt[0])
scene_airmspi.init_cuda_param(Np, init=True)

for cam_ind in range(N_cams):
    I_gt_downscaled =  I_gt[cam_ind][::downscale, ::downscale]
    spp_map = (Np*I_gt_downscaled/np.sum(I_gt_downscaled)).astype(np.uint32)
    start = time()
    print("building path list")
    scene_airmspi.build_path_list(Np, cam_ind, spp_map, False)
    print(f"rendering image {cam_ind}")
    I_total, total_grad = scene_airmspi.render(I_gt_downscaled)
    print(f"mean {I_total.mean()}")
    print(f"mean grad {total_grad.mean()}")
    # scene_airmspi.init_cuda_param(Np, True)
    # I_total_old = scene_airmspi.render_old(Np, cam_ind, spp_map, False)
    # print(f"mean old {I_total_old.mean()}")
    # print(f"took: {time()-start}")
    width, height = scene_airmspi.pixels_shapes[cam_ind]
    I_total_pad[cam_ind, :width, :height] = I_total
    I_gt_pad[cam_ind, :width, :height] = I_gt_downscaled
    print()



# if False:
visual = Visual_wrapper(grid)
visual.plot_images(I_total_pad,"rendered images")
plt.show()
I_total_concat = np.concatenate([img for img in I_total_pad], axis=1)
plt.figure()
plt.imshow(I_total_concat, cmap="gray")
plt.show()



I_gt_concat = np.concatenate([img for img in I_gt_pad], axis=1)
plt.figure()
plt.imshow(I_gt_concat, cmap="gray")
plt.tight_layout()
plt.show()

# I_total_concat *=(I_gt_concat.mean()/I_total_pad.mean())

I_all = np.concatenate([norm_image(I_gt_concat), norm_image(I_total_concat)], axis=0)
plt.figure()
plt.imshow(I_all, cmap="gray")
plt.tight_layout()
plt.show()



I_sub = np.abs(norm_image(I_gt_concat) - norm_image(I_total_concat))
plt.figure()
plt.imshow(I_sub, cmap="gray")
plt.title("sub")
plt.tight_layout()
plt.show()