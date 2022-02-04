
import sys
from os.path import join
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
# from classes.scene import *
from classes.scene_rr_noNEgrad import SceneRR_noNE
from classes.scene_multi_eff import *
from classes.camera import *
from classes.visual import *
from time import time
from utils import *
from cuda_utils import *

cuda.select_device(0)

###################
# Grid parameters #
###################
# bounding box
# edge_x = 0.64
# edge_y = 0.74
# edge_z = 1.04

x_size = 0.02
y_size = 0.02
z_size = 0.04


########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi/180)


#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004


beta_cloud = np.load(join("data","jpl_ext.npy"))
print(beta_cloud.max())
savemat("jpl_ext.mat", {"vol":beta_cloud})
edge_x = x_size * beta_cloud.shape[0]
edge_y = y_size * beta_cloud.shape[1]
edge_z = z_size * beta_cloud.shape[2]
print(edge_x, edge_y, edge_z)
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]], dtype=float_precis)


# cloud_preproccess(beta_cloud, 120)
beta_cloud = beta_cloud.astype(float_reg)
print(beta_cloud.shape)
w0_air = 0.912
w0_cloud = 0.99
# w0_air = 1
# w0_cloud = 1

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
# phase_function = UniformPhaseFunction()
g_cloud = 0.85
#######################
# Cameras declaration #
#######################
height_factor = 2

focal_length = 50e-3
sensor_size = np.array((50e-3, 50e-3)) / height_factor
ps = 76

pixels = np.array((ps, ps))

N_cams = 9
cameras = []
# volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.7
volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.7
R = height_factor * edge_z
# for cam_ind in range(N_cams):
#     phi = 0
#     theta = (-(N_cams // 2) + cam_ind) * 40
#     theta_rad = theta * (np.pi / 180)
#     t = R * theta_phi_to_direction(theta_rad, phi) + volume_center
#     euler_angles = np.array((180, theta, 0))
#     camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
#     cameras.append(camera)
cam_deg = 360 // (N_cams-1)
theta = 29
theta_rad = theta * (np.pi/180)
for cam_ind in range(N_cams-1):
    phi = (-(N_cams//2) + cam_ind) * cam_deg
    phi_rad = phi * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi_rad) + volume_center
    euler_angles = np.array((180-theta, 0, phi-90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
t = R * theta_phi_to_direction(0,0) + volume_center
euler_angles = np.array((180, 0, -90))
cameras.append(Camera(t, euler_angles, cameras[0].focal_length, cameras[0].sensor_size, cameras[0].pixels))


# cameras = [cameras[0]]
# Np = int(5e7)ג
# Np = int(5e7)
Np = int(5e7)
Ns = 15
rr_depth = 10
rr_stop_prob = 0.05

volume.set_mask(beta_cloud>0)
scene_rr = SceneRR_noNE(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
scene_multi = SceneMultiEff(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)

scene_rr.set_cloud_mask(volume.cloud_mask)
scene_multi.set_cloud_mask(volume.cloud_mask)
visual = Visual_wrapper(scene_rr)

# visual.create_grid()
# visual.plot_cameras()
# visual.plot_medium()
plt.show()
run_multi = True
run_rr = True
fake_cloud = beta_cloud #* 0.5

max_val = None

if run_rr:
    print("####### GPU Rusian Roulette renderer ########")
    Np_compilation = 1000
    volume.set_beta_cloud(fake_cloud)
    cuda_paths = scene_rr.build_paths_list(Np_compilation)
    volume.set_beta_cloud(beta_cloud)
    _, _ = scene_rr.render(cuda_paths, 0)
    print("finished compliations")
    del(cuda_paths)
    I_totals = []
    grads_rr = []
    for i in range(2):
        print("generating paths")
        start = time()
        cuda_paths = scene_rr.build_paths_list(Np, to_print=True)
        end = time()
        print(f"building paths took: {end - start}")
        volume.set_beta_cloud(beta_cloud)
        start = time()
        I_total_rr, grad_rr = scene_rr.render(cuda_paths, 0, to_print=True)
        grads_rr.append(grad_rr)
        # I_total_rr = scene_rr.render(cuda_paths, to_print=True)
        I_totals.append(I_total_rr)
        del(cuda_paths)
        print(f" rendering took: {time() - start}")
        visual.plot_images(I_total_rr, f"GPU Rusian Roulette rr_depth={rr_depth}, prob={rr_stop_prob}")
        plt.show()

    del(scene_rr)

if run_multi:
    print("####### GPU multi renderer ########")
    # Np_compilation = 1000
    # cuda_paths = scene_hybrid.build_paths_list(Np_compilation, Ns)
    # _, _ = scene_hybrid.render(cuda_paths, 0)
    # print("finished compliations")
    # del (cuda_paths)
    I_totals_multi = []
    # grads = []
    for i in range(2):
        print("generating paths")
        start = time()
        cuda_paths = scene_multi.build_paths_list(Np, to_print=True)
        end = time()
        print(f"building paths took: {end - start}")
        start = time()
        I_total1, grad_1 = scene_multi.render(cuda_paths, I_totals[0], to_print=True)
        I_totals_multi.append(I_total1)
        memory_paths = cuda_paths[0].nbytes + cuda_paths[1].nbytes
        total_memory = memory_paths + scene_multi.dpath_contrib.nbytes + cuda_paths[2].nbytes
        total_memory /= int(1e9)
        memory_paths /= int(1e9)
        print(f"paths memory={memory_paths} vs total_memory={total_memory}")
        # I_tota/ls.append(I_total1)
        # grads.append(grad1)
        print(f" rendering took: {time() - start}")
        del(cuda_paths)
        visual.plot_images(I_total1, f"GPU multi")
        plt.show()
    #
    if run_rr:
        visual.scatter_plot_comparison(I_total1, I_total_rr, "multi vs rr")
        plt.show()
        visual.scatter_plot_comparison(I_totals[0], I_totals[1], "rr vs rr")
        plt.show()
        visual.scatter_plot_comparison(I_totals_multi[0], I_totals_multi[1], "multi vs multi")
        plt.show()
        visual.scatter_plot_comparison(grads_rr[1], grad_1, "GRAD: multi vs rr")
        plt.show()
        visual.scatter_plot_comparison(grads_rr[0], grads_rr[1], "GRAD: rr vs rr")
        plt.show()
    #






