
import sys
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
# from classes.scene import *
from classes.scene_gpu import *
from classes.camera import *
from classes.visual import *
from time import time
from classes.phase_function import *
from utils import *
from cuda_utils import *
cuda.select_device(0)
print("scatter_eff branch")

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
sun_angles = np.array([165, 0]) * (np.pi/180)


#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004

# beta_cloud = loadmat(join("data", "clouds_dist.mat"))["beta"]
beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
# beta_cloud = loadmat(join("data", "rico2.mat"))["vol"]
# beta_cloud *= 0.1
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
w0_cloud = 0.9
# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
# phase_function = UniformPhaseFunction()
g_cloud = 0.85
g_air = 0.8
phase_function = HGPhaseFunction(g_cloud)
#######################
# Cameras declaration #
#######################
height_factor = 2.5

focal_length = 50e-3
sensor_size = np.array((40e-3, 40e-3)) / height_factor
ps = 55
pixels = np.array((ps, ps))

N_cams = 9
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
R = height_factor * edge_z
for cam_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams // 2) + cam_ind) * 40
    theta_rad = theta * (np.pi / 180)
    t = R * theta_phi_to_direction(theta_rad, phi) + volume_center
    euler_angles = np.array((180, theta, 0))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)

# cameras = [cameras[0]]
Np = int(5e6)
Ns = 10

volume.set_mask(beta_cloud>0)
scene = Scene(volume, cameras, sun_angles, phase_function)
scene_gpu = SceneGPU(volume, cameras, sun_angles, g_cloud, g_air, Ns)


scene_gpu.init_cuda_param(Np, init=True)
gpu_render = True
basic_render = False

visual = Visual_wrapper(scene)

fake_cloud = beta_cloud * 1.0
# fake_cloud = construct_beta(grid_size, False, beta + 2)

max_val = None
if gpu_render:
    print("####### gpu renderer ########")
    print("generating paths")


    cuda_paths, Np_nonan = scene_gpu.build_paths_list(1000, Ns)
    # exit()
    I_total = scene_gpu.render(cuda_paths, 1000, Np_nonan)
    I_total, _ = scene_gpu.render(cuda_paths, 1000, Np_nonan, I_total)
    print("finished compliations")
    del(cuda_paths)
    for i in range(1):
        start = time()
        volume.set_beta_cloud(fake_cloud)
        cuda_paths, Np_nonan = scene_gpu.build_paths_list(Np, Ns)
        end = time()
        print(f"building paths took: {end - start}")
        volume.set_beta_cloud(beta_cloud)
        start = time()
        I_total = scene_gpu.render(cuda_paths, Np, Np_nonan)
        I_total, grad = scene_gpu.render(cuda_paths, Np, Np_nonan, I_total)
        print(f" rendering took: {time() - start}")
        # print(f"grad_norm:{np.linalg.norm(grad)}")
        del(cuda_paths)

    visual.plot_images(I_total, max_val, f"GPU: maximum scattering={Ns}")
    plt.show()
exit()


if basic_render:
    print("####### basic renderer ########")
    print(" start rendering...")
    start = time()
    I_total = scene.render(Np, Ns)
    print(I_total[0].T)
    print(I_total[1].T)
    print(f" rendering took: {time() - start}")
    if max_val is None:
        max_val = np.max(I_total, axis=(1, 2))
    visual.plot_images(I_total, max_val, "basic")
    plt.show()



