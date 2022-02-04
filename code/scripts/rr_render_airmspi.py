
import sys
from os.path import join
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
from deprecated.scene_airmspi_periodic import *
# from classes.scene_airmspi import *
from camera import AirMSPICamera
from classes.visual import *
from time import time
from utils import *
from cuda_utils import *
import pickle
cuda.select_device(0)

sun_intensity = 5e5
downscale = 2
a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
airmspi_data = pickle.load(a_file)
beta_cloud = loadmat(join("data", "FINAL_3D_extinction.mat"))["estimated_extinction"][:,:,:,0]
# beta_cloud = loadmat(join("data", "smoke.mat"))["data"].astype(float_reg)*20
cloud_mask = airmspi_data["cloud_mask"]
# beta_cloud = np.zeros(airmspi_data["grid_shape"], dtype=float_reg)
# beta_cloud = np.zeros(airmspi_data["grid_shape"], dtype=float_reg) * 2
# beta_cloud[cloud_mask] = 2
# beta_cloud[20:40, 20:40, 7:12] = 10




zenith = airmspi_data["sun_zenith"]
azimuth = airmspi_data["sun_azimuth"]
dir_x = -(np.sin(zenith)*np.cos(azimuth))
dir_y = -(np.sin(zenith)*np.sin(azimuth))
dir_z = -np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
# sun_direction = np.array([1e-6,1e-6,-1]).astype(float)
print(sun_direction)
# sun_direction = np.array([1e-6,1e-6,-1.0])
#####################
# grid parameters #
#####################
grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
# grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
print(grid.bbox)




#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004

# cloud_mask = beta_cloud>0



# cloud_preproccess(beta_cloud, 120)
# beta_cloud = np.zeros(grid.shape, dtype=float_reg)
# beta_cloud[cloud_mask] = 2

print(beta_cloud.shape)
print(beta_cloud.mean())
w0_air = 0.912
w0_cloud = 0.99
# w0_air = 1
# w0_cloud = 1

# Declerations
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
g_cloud = 0.0001
#######################
# Cameras declaration #
#######################

ts = airmspi_data["ts"]
# ts = [np.array([grid.bbox_size[0]/2, grid.bbox_size[1]/2, 20]) for _ in range(9)]
Ps = airmspi_data["Ps"]
resolutions = airmspi_data["resolution"]
N_cams = len(ts)
cameras = []
for k in range(N_cams):
    cameras.append(AirMSPICamera(resolutions[k], ts[k], Ps[k]))

Np = int(1e7)
Ns = 15
rr_depth = 20
rr_stop_prob = 0.1

# volume.set_mask(cloud_mask)
volume.set_mask(beta_cloud>0)
scene_airmspi = SceneAirMSPI(volume, cameras, sun_direction, sun_intensity, g_cloud, rr_depth, rr_stop_prob, downscale)



# loading gt
I_gt = airmspi_data["images"]
pad_shape = scene_airmspi.pixels_shape
I_gt_pad = np.zeros((N_cams, *pad_shape), dtype=float_reg)
camera_array_list = np.zeros((N_cams, *scene_airmspi.pixels_shape, 6), dtype=float_reg)
# for k in range(N_cams):
#     I_gt_pad[k,:resolutions[k][0], :resolutions[k][1]]= I_gt[k]
#     camera_array_list[k, :resolutions[k][0], :resolutions[k][1], :] = airmspi_data["camera_array_list"][k][:,:,:6]



image_threshold = np.ones(N_cams, dtype=float_reg) * 0.25
image_threshold[0] = 0.4
image_threshold[7] = 0.35
image_threshold[8] = 0.48
hit_threshold = 0.9
spp = 100000
# cloud_mask = scene_airmspi.space_curving(I_gt_pad, image_threshold=image_threshold, hit_threshold=hit_threshold,
#                                          camera_array_list=camera_array_list, spp=spp)
# print(cloud_mask.mean())
# scene_airmspi.set_cloud_mask(cloud_mask)
# scene_airmspi.volume.beta_cloud = np.zeros_like(beta_cloud)
# scene_airmspi.volume.beta_cloud[cloud_mask] = 2
visual = Visual_wrapper(grid)

# visual.create_grid()
# visual.plot_cameras()
# visual.plot_medium(beta_cloud)
# plt.show()
run_rr = True
run_hybrid = False
fake_cloud = beta_cloud #* 0.5

max_val = None


print("####### GPU AirMSpi renderer ########")
Np_compilation = 1000
cuda_paths = scene_airmspi.build_paths_list(Np_compilation)
# _,_ = scene_airmspi.render(cuda_paths,0)
_ = scene_airmspi.render(cuda_paths)
print("finished compliations")
del(cuda_paths)
print("generating paths")
start = time()
cuda_paths = scene_airmspi.build_paths_list(Np, to_print=True)
end = time()
print(f"building paths took: {end - start}")
start = time()
I_total = scene_airmspi.render(cuda_paths,  to_print=True)
print(f" rendering took: {time() - start}")
# del(cuda_paths)
# cuda_paths = scene_hybrid.build_paths_list(Np, Ns)
# I_total_lowmem2, grad_lowmem2 = scene_hybrid.render(cuda_paths, 0)
I_gt = airmspi_data["images"]
I_gt_pad = np.ones((N_cams, *scene_airmspi.pixels_shape), dtype=float_reg) * np.min(I_gt[0])
for k in range(N_cams):
    I_gt_pad[k,:resolutions[k][0],:resolutions[k][1]] = I_gt[k]

I_gt_pad  = I_gt_pad[:,::downscale,::downscale]
I_total_concat = np.concatenate([img for img in I_total], axis=1)
I_gt_concat = np.concatenate([img for img in I_gt_pad], axis=1)
I_all = np.concatenate([I_gt_concat, I_total_concat], axis=0)
print(f"I_gt_mean:{I_gt_pad[I_gt_pad != 0].mean()}, I_opt_mean:{I_total[I_total != 0].mean()}")

plt.figure()
plt.imshow(I_all, cmap="gray")
# plt.colorbar()
plt.tight_layout()
plt.show()
# I_gt_pad = I_gt_pad ** (0.7)
visual.plot_images_airmspi(I_total, resolutions, f"GPU AirMSPI", downscale)
# visual.plot_images_airmspi_side_by_side(I_total, I_gt_pad, resolutions, f"GPU AirMSPI", downscale)
plt.show()
# visual.plot_images_airmspi(I_gt, resolutions, f"GPU AirMSPI GT")




