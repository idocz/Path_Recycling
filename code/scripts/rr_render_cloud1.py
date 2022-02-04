
import sys
from os.path import join
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
# from classes.scene import *
from classes.scene_rr import *
from deprecated.scene_hybrid_gpu import *
from camera import *
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


beta_cloud = loadmat(join("data", "cloud1.mat"))["beta_cloud1"]
# beta_cloud = np.load(join("data","jpl_ext.npy"))
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
g_cloud = 0.88
#######################
# Cameras declaration #
#######################
height_factor = 2

focal_length = 50e-3
sensor_size = np.array((25e-3, 25e-3)) / height_factor
ps = 80

pixels = np.array((ps, ps))

N_cams = 9
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.7
# volume_center = (bbox[:, 1] - bbox[:, 0])
R = height_factor * edge_z
# for cam_ind in range(N_cams):
#     phi = 0
#     theta = (-(N_cams // 2) + cam_ind) * 40
#     theta_rad = theta * (np.pi / 180)
#     t = R * theta_phi_to_direction(theta_rad, phi) + volume_center
#     euler_angles = np.array((180, theta, 0))
#     camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
#     cameras.append(camera)

for cam_ind in range(N_cams):
    theta = np.pi/2
    phi = (-(N_cams//2) + cam_ind) * 40
    phi_rad = phi * (np.pi/180)
    t = R * theta_phi_to_direction(theta,phi_rad) + volume_center
    t[2] -= 0.5
    print(cam_ind, t-volume_center)
    euler_angles = np.array((90, 0, phi-90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)



# cameras = [cameras[0]]
# Np = int(5e7)ג
# Np = int(5e7)
Np = int(5e7)
Ns = 15
rr_depth = 100
rr_stop_prob = 0.01

volume.set_mask(beta_cloud>0)
scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
scene_hybrid = SceneHybridGpu(volume, cameras, sun_angles, g_cloud, Ns)

scene_rr.set_cloud_mask(volume.cloud_mask)
scene_hybrid.set_cloud_mask(volume.cloud_mask)
visual = Visual_wrapper(scene_rr)

# visual.create_grid()
# visual.plot_cameras()
# visual.plot_medium()
plt.show()

run_rr = True
run_hybrid = False
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
    print("generating paths")
    start = time()
    cuda_paths = scene_rr.build_paths_list(Np, to_print=True)
    end = time()
    print(f"building paths took: {end - start}")
    volume.set_beta_cloud(beta_cloud)
    start = time()
    I_total_rr, grad_rr = scene_rr.render(cuda_paths, 0, to_print=True)
    print(f" rendering took: {time() - start}")
    # del(cuda_paths)
    # cuda_paths = scene_hybrid.build_paths_list(Np, Ns)
    # I_total_lowmem2, grad_lowmem2 = scene_hybrid.render(cuda_paths, 0)
    visual.plot_images(I_total_rr, f"GPU Rusian Roulette rr_depth={rr_depth}, prob={rr_stop_prob}")
    plt.show()
    # visual.scatter_plot_comparison(grad_lowmem, grad_lowmem2, "GRAD: lowmem vs lowmem")
    # plt.show()
    # visual.scatter_plot_comparison(I_total_lowmem, I_total_lowmem2, "lowmem vs lowmem")
    # plt.show()

if run_hybrid:
    print("####### GPU hybrid renderer ########")
    Np_compilation = 1000
    cuda_paths = scene_hybrid.build_paths_list(Np_compilation, Ns)
    _, _ = scene_hybrid.render(cuda_paths, 0)
    print("finished compliations")
    del (cuda_paths)
    I_totals = []
    grads = []
    for i in range(2):
        print("generating paths")
        start = time()
        volume.set_beta_cloud(fake_cloud)
        cuda_paths = scene_hybrid.build_paths_list(Np, Ns, to_print=True)
        volume.set_beta_cloud(beta_cloud)
        end = time()
        print(f"building paths took: {end - start}")
        volume.set_beta_cloud(beta_cloud)
        start = time()
        I_total1, grad1 = scene_hybrid.render(cuda_paths, 0, to_print=True)
        I_totals.append(I_total1)
        grads.append(grad1)
        print(f" rendering took: {time() - start}")
        del(cuda_paths)
    visual.plot_images(I_totals[1], f"GPU Hybrid: maximum scattering={Ns}")
    plt.show()

    if run_rr:
        visual.scatter_plot_comparison(I_totals[0], I_total_rr, "hybrid vs rr")
        plt.show()
        visual.scatter_plot_comparison(I_totals[0], I_totals[1], "hybrid vs hybrid")
        plt.show()
        visual.scatter_plot_comparison(grads[0], grad_rr, "GRAD: hybrid vs rr")
        plt.show()
        visual.scatter_plot_comparison(grads[0], grads[1], "GRAD: hybrid vs hybrid")
        plt.show()







