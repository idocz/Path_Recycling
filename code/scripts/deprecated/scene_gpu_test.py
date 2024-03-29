
import sys
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
# from classes.scene import *
from classes.deprecated.scene_numba import *
from classes.scene_gpu import *
from classes.camera import *
from classes.visual import *
from time import time
from classes.phase_function import *
from utils import *
from cuda_utils import *
cuda.select_device(0)

###################
# Grid parameters #
###################
# bounding box
edge_x = 0.64
edge_y = 0.74
edge_z = 1.04
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]], dtype=float_precis)

########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi/180)


#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.1

beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
# beta_cloud = np.array([[[0,0,0],
#                         [0,5,0],
#                         [0,0,0]],
#                        [[0,2,0],
#                         [3,4,3],
#                         [0,2,0]],
#                         [[0,0,0],
#                          [0,2,0],
#                          [0,0,0]]])

# cloud_preproccess(beta_cloud, 120)
beta_cloud = beta_cloud.astype(float_reg)
print(beta_cloud.shape)
w0_air = 0.8
w0_cloud = 0.7
# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
# phase_function = UniformPhaseFunction()
g = 0.5
g_cloud = 0.5
g_air = 0.5
phase_function = HGPhaseFunction(g)
#######################
# Cameras declaration #
#######################
focal_length = 60e-3
sensor_size = np.array((40e-3, 40e-3))
ps = 20
pixels = np.array((ps, ps))

N_cams = 1
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
R = 1.5 * edge_z
for cam_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams // 2) + cam_ind) * 40
    theta_rad = theta * (np.pi / 180)
    t = R * theta_phi_to_direction(theta_rad, phi) + volume_center
    euler_angles = np.array((180, theta, 0))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)

# cameras = [cameras[4], cameras[8]]
Np = int(1e6)
Ns = 15

volume.set_mask(beta_cloud>0)
scene = Scene(volume, cameras, sun_angles, phase_function)
scene_numba = SceneNumba(volume, cameras, sun_angles, g)
scene_gpu = SceneGPU(volume, cameras, sun_angles, g_cloud, g_air, Ns)

scene_gpu.init_cuda_param(Np, seed=None)

gpu_render = True
numba_render = False
basic_render = False

visual = Visual_wrapper(scene)

fake_cloud = beta_cloud #* (2/3)
# fake_cloud = construct_beta(grid_size, False, beta + 2)

max_val = None

print("####### gpu renderer ########")
print("generating paths")

volume.set_beta_cloud(fake_cloud)
cuda_paths, Np_nonan = scene_gpu.build_paths_list(1000, Ns)
I_total = scene_gpu.render(cuda_paths, 1000, Np_nonan)
I_total, _ = scene_gpu.render(cuda_paths, 1000, Np_nonan, I_total)
print("finished compliations")
del(cuda_paths)
for i in range(1):
    start = time()
    cuda_paths, Np_nonan = scene_gpu.build_paths_list(Np, Ns)
    end = time()
    print(f"building paths took: {end - start}")

    start = time()
    I_total_gpu = scene_gpu.render(cuda_paths, Np, Np_nonan)
    # I_total, grad = scene_gpu.render(cuda_paths, Np, I_total)
    print(f" rendering took: {time() - start}")
    # print(f"grad_norm:{np.linalg.norm(grad)}")
    del(cuda_paths)

    visual.plot_images(I_total_gpu, max_val, f"GPU: maximum scattering={Ns}")
    plt.show()




print("####### basic renderer ########")
print(" start rendering...")
start = time()
I_total1 = scene.render(Np, Ns)
I_total2 = scene.render(Np, Ns)
print(f" rendering took: {time() - start}")
if max_val is None:
    max_val = np.max(I_total, axis=(1, 2))
visual.plot_images(I_total1, max_val, "basic")
plt.show()

plt.figure()
X = I_total1.reshape(-1)
Y = I_total_gpu.reshape(-1)
plt.scatter(X,Y)
plt.plot([0, 0.0001],[0, 0.0001])
plt.title("gt1 vs GPU")
mask = X != 0
rel_err = np.sum(np.abs(X[mask] - Y[mask]))/np.sum(np.abs(X[mask]))
print(f"gt1 vs GPU err: {rel_err}")

plt.figure()
X = I_total1.reshape(-1)
Y = I_total2.reshape(-1)
mask = X != 0
rel_err = np.sum(np.abs(X[mask] - Y[mask]))/np.sum(np.abs(X[mask]))
print(f"gt1 vs gt2 err: {rel_err}")


plt.scatter(X,Y)
plt.plot([0, 0.0001],[0, 0.0001])
plt.title("gt1 vs gt2")
plt.show()
