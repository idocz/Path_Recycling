
import sys
from os.path import join

from scene_rr_noNEgrad import SceneRR_noNE

sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
# from classes.scene import *
from classes.scene_rr import SceneRR
from classes.scene_seed_NEgrad import *
from classes.scene_multi_eff import  SceneMultiEff
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

cam_deg = 360 // (N_cams-1)
for cam_ind in range(N_cams-1):
    theta = 29
    theta_rad = theta * (np.pi/180)
    phi = (-(N_cams//2) + cam_ind) * cam_deg
    phi_rad = phi * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi_rad) + volume_center
    euler_angles = np.array((180-theta, 0, phi-90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
t = R * theta_phi_to_direction(0,0) + volume_center
euler_angles = np.array((180, 0, -90))
cameras.append(Camera(t, euler_angles, cameras[0].focal_length, cameras[0].sensor_size, cameras[0].pixels))


Np = int(5e7)

rr_depth = 20
rr_stop_prob = 0.05

volume.set_mask(beta_cloud>0)
scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
scene_seed = SceneSeed(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
scene_multi = SceneMultiEff(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)

scene_seed.set_cloud_mask(volume.cloud_mask)
scene_rr.set_cloud_mask(volume.cloud_mask)
scene_multi.set_cloud_mask(volume.cloud_mask)
visual = Visual_wrapper(scene_rr)

# visual.create_grid()
# visual.plot_cameras()
# visual.plot_medium()
plt.show()
run_seed = True
# run_rr = True
run_multi = False
run_rr = True
# fake_cloud = np.zeros_like(beta_cloud)
# fake_cloud[beta_cloud>0] = np.mean(beta_cloud[beta_cloud>0])
fake_cloud = beta_cloud#*0.7
max_val = None
I_diff = np.zeros((len(cameras), *cameras[0].pixels),dtype=float_reg)
if run_seed:
    print("####### GPU Seed renderer ########")
    scene_seed.init_cuda_param(Np, init=True)
    for i in range(1):
        print(f"##{i}##")
        volume.set_beta_cloud(fake_cloud)
        print("generating paths")
        start = time()
        scene_seed.build_paths_list(Np, to_print=True)
        end = time()
        print(f"building paths took: {end - start}")
        volume.set_beta_cloud(beta_cloud)
        # I_total_seed = scene_seed.render(to_print=False)
        # scene_seed.init_cuda_param(Np, init=True)
        # scene_seed.build_paths_list(Np, to_print=False)
        # scene_seed.build_paths_list(Np, to_print=False)
        # I_total_seed2 = scene_seed.render(None, to_print=False)
        start = time()
        I_total_seed2, seed_grad = scene_seed.render(0, to_print=False)
        print(f" rendering took: {time() - start}")
        # I_total_seed
    # del(cuda_paths)
    # cuda_paths = scene_hybrid.build_paths_list(Np, Ns)
    # I_total_lowmem2, grad_lowmem2 = scene_hybrid.render(cuda_paths, 0)
    visual.plot_images(I_total_seed2, f"GPU Seed rr_depth={rr_depth}, prob={rr_stop_prob}")
    plt.show()
    del(scene_seed)

if run_rr:
    print("####### GPU Rusian Roulette renderer ########")
    for _ in range(1):
        Np_compilation = 1000

        cuda_paths = scene_rr.build_paths_list(Np_compilation)

        _, _ = scene_rr.render(cuda_paths, 0)
        print("finished compliations")
        del(cuda_paths)
        print("generating paths")
        start = time()
        volume.set_beta_cloud(fake_cloud)
        cuda_paths = scene_rr.build_paths_list(Np, to_print=True)
        memory = 0
        for cuda_array in cuda_paths:
            memory += cuda_array.nbytes
        # memory += scene_rr.dpath_contrib.nbytes
        memory += scene_rr.dgrad_contrib.nbytes
        memory += scene_rr.dI_total.nbytes
        memory += scene_rr.dtotal_grad.nbytes
        memory += scene_rr.dcloud_mask.nbytes
        memory += scene_rr.rng_states.nbytes
        print("memory:",memory/1e9)
        end = time()
        print(f"building paths took: {end - start}")
        volume.set_beta_cloud(beta_cloud)
        start = time()
        I_total_rr, grad_rr = scene_rr.render(cuda_paths, 0, to_print=True)
        print(f" rendering took: {time() - start}")
        del(cuda_paths)


    # visual.scatter_plot_comparison(I_total_seed, I_total_seed2, "seed vs seed2")
    # plt.show()
    # cloud_mask = scene_rr.space_curving(I_total_seed, image_threshold=0.1, hit_threshold=0.9, spp=100000)
    # mask_grader(cloud_mask, beta_cloud>0,beta_cloud)
    # visual.scatter_plot_comparison(grad_lowmem, grad_lowmem2, "GRAD: lowmem vs lowmem")
    # plt.show()
    # visual.scatter_plot_comparison(I_total_lowmem, I_total_lowmem2, "lowmem vs lowmem")
    # plt.show()

    volume.set_beta_cloud(fake_cloud)
    cuda_paths = scene_rr.build_paths_list(Np, to_print=True)
    volume.set_beta_cloud(beta_cloud)
    I_total_rr2, grad_rr2 = scene_rr.render(cuda_paths, 0, to_print=True)
    # del(cuda_paths)
    # cuda_paths = scene_hybrid.build_paths_list(Np, Ns)
    # I_total_lowmem2, grad_lowmem2 = scene_hybrid.render(cuda_paths, 0)
    visual.plot_images(I_total_rr, f"GPU Rusian Roulette rr_depth={rr_depth}, prob={rr_stop_prob}")
    plt.show()
    # visual.scatter_plot_comparison(grad_lowmem, grad_lowmem2, "GRAD: lowmem vs lowmem")
    # plt.show()
    # visual.scatter_plot_comparison(I_total_lowmem, I_total_lowmem2, "lowmem vs lowmem")
    # plt.show()

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
        I_total1, grad_1 = scene_multi.render(cuda_paths, I_total_seed2, to_print=True)
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
if run_rr:
    visual.scatter_plot_comparison(I_total_seed2, I_total_rr, "seed vs rr")
    plt.show()
    visual.scatter_plot_comparison(I_total_rr2, I_total_rr, "rr vs rr")
    plt.show()
    visual.scatter_plot_comparison(seed_grad, grad_rr, "GRAD: seed vs rr")
    plt.show()
    visual.scatter_plot_comparison(grad_rr2, grad_rr, "GRAD: rr vs rr")
    plt.show()








