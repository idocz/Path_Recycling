
import sys
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
# from classes.scene import *
from deprecated.scene_lowmem_gpu import *
from deprecated.scene_ddis import *
from camera import *
from classes.visual import *
from time import time
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
beta_cloud *= 0.1
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
# Np = int(5e7) // 55**2
Np = int(5e7)
Ns = 15
Ns_ddis = 15
volume.set_mask(beta_cloud>0)
scene_lowmem = SceneLowMemGPU(volume, cameras, sun_angles, g_cloud, Ns)
scene_ddis = SceneDDIS(volume, cameras, sun_angles, g_cloud, Ns_ddis)
visual = Visual_wrapper(scene_lowmem)

fake_cloud = beta_cloud #* 0.5
# fake_cloud = construct_beta(grid_size, False, beta + 2)


print("####### Convergence Rate Script ########")
scene_lowmem.init_cuda_param(Np, init=True)
scene_ddis.init_cuda_param(Np, init=True, seed=scene_lowmem.seed)


N = 100
Nps = np.logspace(3,np.log10(Np), N)
res = np.zeros(N, dtype=np.float64)
res_ddis = np.zeros(N, dtype=np.float64)
for i,Np_loop in enumerate(Nps):
    start = time()
    # scene_lowmem.init_cuda_param(int(Np_loop),True)
    cuda_paths = scene_lowmem.build_paths_list(int(Np_loop), Ns)
    cuda_paths_ddis = scene_ddis.build_paths_list(int(Np_loop), Ns_ddis)
    print(scene_lowmem.total_num_of_scatter == scene_ddis.total_num_of_scatter)
    I_total_lowmem = scene_lowmem.render(cuda_paths, to_print=True)
    I_total_ddis= scene_ddis.render(cuda_paths_ddis, to_print=True)
    del(cuda_paths)
    res[i] = I_total_lowmem.mean()#[0,ps//2,ps//2]
    res_ddis[i] = I_total_ddis.mean()#[0,ps//2,ps//2]
    print(f"{i}: iteration took: {time() - start}")
# print(f"grad_norm:{np.linalg.norm(grad)}")
y_max = np.max([res.max(), res_ddis.max()])
plt.figure()
plt.semilogx(Nps, res, label="lowmem")
plt.semilogx(Nps, res_ddis, label="DDIS")
plt.title(f"LOWMEM vs DDIS: Np={Np:.0e} Pixels={ps}, g={g_cloud}, Ns={Ns}, e_ddis={e_ddis}")
plt.ylim(0, y_max)
plt.legend()
# plt.figure()
# plt.semilogx(Nps, res_ddis)
# plt.title(f"DDIS: Np={Np} Pixels={ps}, g={g_cloud}, Ns={Ns}")
# plt.ylim(0, y_max)
plt.show()







