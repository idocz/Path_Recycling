import os, sys
my_lib_path = os.path.abspath('../')
sys.path.append(my_lib_path)
from deprecated.scene_gpu import *
from camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from classes.checkpoint_wrapper import CheckpointWrapper
from classes.optimizer import *
from scipy.optimize import minimize
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
                 [0, edge_z]])

########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
beta_cloud = beta_cloud.astype(float_reg)
beta_shape = beta_cloud.shape
# beta_cloud *= (127/beta_cloud.max())


print(beta_cloud)

beta_air = 0.1
w0_air = 1.0
w0_cloud = 0.8
g_cloud = 0.5
g_air = 0.5

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################


focal_length = 60e-3
sensor_size = np.array((40e-3, 40e-3))
ps = 55
pixels = np.array((ps, ps))

N_cams = 9
cameras = []
volume_center = (bbox[:,1] - bbox[:,0])/2
R = 1.5 * edge_z
for cam_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams//2) + cam_ind) * 40
    theta_rad = theta * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi) + volume_center
    euler_angles = np.array((180, theta, 0))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
# scene = Scene(volume, cameras, sun_angles, phase_function)



# Simulation parameters
Np_gt = int(1e6)

phase = 0
Np = int(1e5)
resample_freq = 10
step_size = 1e9
Ns = 15
iterations = 10000000
to_mask = True
tensorboard = False
tensorboard_freq = 15
beta_max = 160
win_size = 300
# grads_window = np.zeros((win_size, *beta_cloud.shape), dtype=float_reg)

seed = None
# Cloud mask (GT for now)
cloud_mask = beta_cloud > 0
# cloud_mask = beta_cloud >= 0
volume.set_mask(cloud_mask)

scene_gpu = SceneGPU(volume, cameras, sun_angles, g_cloud, g_air, Ns)

visual = Visual_wrapper(scene_gpu)

load_gt = False
if load_gt:
    checkpoint_id = "2212-1250-03"
    I_gt = np.load(join("checkpoints",checkpoint_id,"data","gt.npz"))["images"]
    cuda_paths = None
    print("I_gt has been loaded")
else:
    scene_gpu.init_cuda_param(Np_gt, init=True)
    cuda_paths, Np_nonan = scene_gpu.build_paths_list(Np_gt, Ns)
    print(Np_nonan)
    I_gt = scene_gpu.render(cuda_paths, Np_gt, Np_nonan)
    del(cuda_paths)
    cuda_paths = None
max_val = np.max(I_gt, axis=(1,2))
visual.plot_images(I_gt, max_val, "GT")
plt.show()

scene_gpu.init_cuda_param(Np)
alpha = 0.9
beta1 = 0.9
beta2 = 0.999
optimizer = None
if tensorboard:
    tb = TensorBoardWrapper(I_gt, beta_gt)
    cp_wrapper = CheckpointWrapper(scene_gpu, optimizer, Np_gt, Np, Ns, resample_freq, step_size, iterations,
                            tensorboard_freq, tb.train_id)
    tb.add_scene_text(str(cp_wrapper))
    pickle.dump(cp_wrapper, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    print("Checkpoint wrapper has been saved")

# Initialization
beta_init = np.zeros_like(beta_cloud)
beta_mean = np.mean(beta_cloud[cloud_mask])
beta_init[cloud_mask] = beta_mean
volume.set_beta_cloud(beta_init)
beta_opt = volume.beta_cloud

# grad_norm = None
non_min_couter = 0
next_phase = False
min_loss = 1

Np_nonan = 0

args = [*scene_gpu.build_paths_list(Np, Ns), 0]
global iteraion
iteration = 0

def differntiable_fun(x):
    beta_opt = x.reshape(*beta_shape)
    cuda_paths, Np_nonan, _ = args
    scene_gpu.volume.beta_cloud = beta_opt
    I_opt, grad=scene_gpu.render(cuda_paths, Np, Np_nonan, I_gt)
    dif = (I_opt - I_gt).reshape(1,1,1, N_cams, pixels[0], pixels[1])
    loss = 0.5 * np.sum(dif ** 2)
    grad = grad.reshape(-1)
    return loss, grad

def callback_fun(x):
    beta_opt = x.reshape(*beta_shape)
    if args[2] % resample_freq ==1000:
        print("resampling paths")
        cuda_paths, Np_nonan = scene_gpu.build_paths_list(Np, Ns)
        del(args[0])
        args[0] = cuda_paths
        args[1] = Np_nonan
    rel_dist1 = relative_distance(beta_gt, beta_opt)
    print("rel_dist1=",rel_dist1)
    print("iteration=",args[2])
    args[2] += 1
    return False

options = {"disp":1,  "gtol":1e-13, "ftol":1e-100, "maxls":5}
print("res")
x0 = beta_init.reshape(-1)
res = minimize(fun=differntiable_fun, x0=beta_init, method='L-BFGS-B', jac=True, options=options, callback=callback_fun)
print(res)