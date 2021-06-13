
import sys
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
# from classes.scene import *
from classes.scene_ddis import *
from classes.scene_lowmem_gpu import *
from classes.camera import *
from classes.visual import *
from time import time
from classes.phase_function import *
from utils import *
from cuda_utils import *
cuda.select_device(0)
print("low_mem fast branch")

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
w0_air = 1
w0_cloud = 0.9
# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
# phase_function = UniformPhaseFunction()
g_cloud = 0.85
#######################
# Cameras declaration #
#######################
height_factor = 2.5

focal_length = 50e-3
sensor_size = np.array((40e-3, 40e-3)) / height_factor
ps = 55

pixels = np.array((ps, ps))

N_cams = 1
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
Np = int(1e7)
Ns = 15
Ns_ddis = Ns-1

volume.set_mask(beta_cloud>0)
scene_ddis = SceneDDIS(volume, cameras, sun_angles, g_cloud, Ns)
visual = Visual_wrapper(scene_ddis)

# fake_cloud = beta_cloud * 0.5
noise_std = 0
fake_cloud = beta_cloud + np.random.randn(*beta_cloud.shape)*noise_std
fake_cloud[fake_cloud<0] = 0
# fake_cloud = construct_beta(grid_size, False, beta + 2)


run_lowmem = True

print("####### ddis renderer ########")
scene_ddis.init_cuda_param(Np, init=True)
print("STARTING COMPILATION")
Np_compilation = 10000
cuda_paths = scene_ddis.build_paths_list(Np_compilation, Ns_ddis)
# exit()
_, _ = scene_ddis.render(cuda_paths, 0)
print("FINISHED COMPILATION")
del(cuda_paths)
print("generating paths")
start = time()
volume.set_beta_cloud(fake_cloud)
scene_ddis.init_cuda_param(Np)
cuda_paths = scene_ddis.build_paths_list(Np, Ns_ddis)#, to_print=True)
end = time()
print(f"building paths took: {end - start}")
volume.set_beta_cloud(beta_cloud)
start = time()
I_total_ddis = scene_ddis.render(cuda_paths)#, to_print=True)
print(f" rendering took: {time() - start}")
# print(f"grad_norm:{np.linalg.norm(grad)}")
del(cuda_paths)
print("generating paths")
start = time()
volume.set_beta_cloud(fake_cloud)
scene_ddis.init_cuda_param(Np)
cuda_paths = scene_ddis.build_paths_list(Np, Ns_ddis)#, to_print=True)
end = time()
print(f"building paths took: {end - start}")
volume.set_beta_cloud(beta_cloud)
start = time()
I_total_ddis2 = scene_ddis.render(cuda_paths)#, to_print=True)
print(f" rendering took: {time() - start}")
# print(f"grad_norm:{np.linalg.norm(grad)}")
del(cuda_paths)
visual.plot_images(I_total_ddis, f"ddis: maximum scattering={Ns}, e_ddis={e_ddis}")
plt.show()
visual.scatter_plot_comparison(I_total_ddis, I_total_ddis2, "ddis vs ddis")
plt.show()

if run_lowmem:
    Np*=2
    scene_lowmem = SceneLowMemGPU(volume, cameras, sun_angles, g_cloud, Ns)
    print("####### lowmem renderer ########")
    scene_lowmem.init_cuda_param(Np, init=True)
    print("STARTING COMPILATION")
    Np_compilation = 10000
    cuda_paths = scene_lowmem.build_paths_list(Np_compilation, Ns)
    _, _ = scene_lowmem.render(cuda_paths, 0)
    print("FINISHED COMPILATION")
    del (cuda_paths)
    print("generating paths")
    start = time()
    volume.set_beta_cloud(fake_cloud)
    cuda_paths = scene_lowmem.build_paths_list(Np, Ns)#, to_print=True)
    end = time()
    print(f"building paths took: {end - start}")
    volume.set_beta_cloud(beta_cloud)
    start = time()
    I_total_lowmem = scene_lowmem.render(cuda_paths)#, to_print=True)
    print(f" rendering took: {time() - start}")
    # print(f"grad_norm:{np.linalg.norm(grad)}")
    del (cuda_paths)


    print("generating paths")
    start = time()
    volume.set_beta_cloud(fake_cloud)
    scene_lowmem.init_cuda_param(Np)
    cuda_paths = scene_lowmem.build_paths_list(Np, Ns)#, to_print=True)
    end = time()
    print(f"building paths took: {end - start}")
    volume.set_beta_cloud(beta_cloud)
    start = time()
    I_total_lowmem2 = scene_lowmem.render(cuda_paths)#, to_print=True)

    visual.plot_images(I_total_lowmem2, f"lowmem: maximum scattering={Ns}")
    plt.show()
    visual.scatter_plot_comparison(I_total_lowmem, I_total_lowmem2, "lowmem vs lowmem")
    plt.show()
    visual.scatter_plot_comparison(I_total_lowmem, I_total_ddis, "lowmem vs ddis")
    plt.show()



