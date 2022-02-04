
import sys
sys.path.append("/home/idocz/repos/3D_Graph_Renderer/code/")
# from classes.scene import *
from deprecated.scene_ddis import *
from camera import *
from classes.visual import *
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
Np = int(1e5)
Np_mean = int(1e7)
Ns = 15

volume.set_mask(beta_cloud>0)
scene_ddis = SceneDDIS(volume, cameras, sun_angles, g_cloud, Ns)
visual = Visual_wrapper(scene_ddis)

fake_cloud = beta_cloud #* 0.5
# fake_cloud = construct_beta(grid_size, False, beta + 2)

max_val = None

run_lowmem = False

print("####### ddis renderer ########")
scene_ddis.init_cuda_param(Np_mean, init=True)
# RENDERING I_MEAN
scene_ddis.init_cuda_param(Np_mean)
cuda_paths = scene_ddis.build_paths_list(Np_mean, Ns)#, to_print=True)
I_mean = scene_ddis.render(cuda_paths)#, to_print=True)
del(cuda_paths)
# RENDERING STD
cuda_paths = scene_ddis.build_paths_list(Np, Ns)#, to_print=True)
I_total_ddis, I_std = scene_ddis.render_std(cuda_paths, I_mean)#, to_print=True)
print("ddis var:", I_std.mean())
del(cuda_paths)
# PLOTTING
visual.plot_images(I_total_ddis, f"ddis: Ns={Ns}, e_ddis={e_ddis}")
plt.show()
visual.plot_images(I_std, f"ddis: Ns={Ns}, e_ddis={e_ddis} sum={I_std.sum():.2e} max={I_std.max():.2e}")
plt.show()
exit()



