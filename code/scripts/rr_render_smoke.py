
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
z_size = 0.02

########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi/180)


#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004 / 1000

_, bbox = read_binary_grid3d(join("data","smoke.vol"))
# edge_x, edge_z, edge_y  = bbox[:,1] - bbox[:,0]
beta_cloud = loadmat(join("data", "smoke.mat"))["data"] * 10
print(beta_cloud.max())
beta_cloud = np.ascontiguousarray(np.rot90(beta_cloud, axes=(2,1)))
beta_cloud =np.roll(beta_cloud, axis=0, shift=-20)
# beta_cloud *= 0.1[

voxel_size_x = 0.02
voxel_size_y = 0.02
voxel_size_z = 0.02
edge_x = voxel_size_x * beta_cloud.shape[0]
edge_y = voxel_size_y * beta_cloud.shape[1]
edge_z = voxel_size_z * beta_cloud.shape[2]
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]], dtype=float_precis)

print(edge_x, edge_y, edge_z)

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
g_cloud = 0.5
#######################
# Cameras declaration #
#######################
height_factor = 1.5

focal_length = 50e-3
sensor_size = np.array((50e-3, 50e-3)) / height_factor
ps = 200

pixels = np.array((ps, ps))

N_cams = 9
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.6
R = height_factor * edge_z
# for cam_ind in range(N_cams):
#     phi = 0
#     theta = (-(N_cams // 2) + cam_ind) * 40
#     theta_rad = theta * (np.pi / 180)
#     t = R * theta_phi_to_direction(theta_rad, phi) + volume_center
#     euler_angles = np.array((180, theta, 0))
#     camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
#     cameras.append(camera)
#
# for cam_ind in range(N_cams):
#     theta = np.pi/2
#     phi = (-(N_cams//2) + cam_ind) * 40
#     phi_rad = phi * (np.pi/180)
#     t = R * theta_phi_to_direction(theta,phi_rad) + volume_center
#     t[2] -= 0.5
#     print(cam_ind, t-volume_center)
#     euler_angles = np.array((90, 0, phi-90))
#     camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
#     cameras.append(camera)
cam_deg = 360 // (N_cams-1)
theta = 90
theta_rad = theta * (np.pi/180)
for cam_ind in range(N_cams-1):
    phi = (-(N_cams//2) + cam_ind) * cam_deg
    phi_rad = phi * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi_rad) + volume_center
    t[2] -= 0.5
    euler_angles = np.array((180-theta, 0, phi-90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
t = R * theta_phi_to_direction(0,0) + volume_center
euler_angles = np.array((180, 0, -90))
cameras.append(Camera(t, euler_angles, cameras[0].focal_length, cameras[0].sensor_size, cameras[0].pixels))



# cameras = [cameras[0]]
Np = int(5e7)
Ns = 15
rr_depth = 10
rr_stop_prob = 0.5

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
    cloud_mask = scene_rr.space_curving(I_total_rr, image_threshold=0.05, hit_threshold=0.9, spp=10000)
    mask_grader(cloud_mask, beta_cloud > 0.1, beta_cloud)
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







