import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene_seed_NEgrad import SceneSeed
from classes.camera import *
from classes.grid import *
from utils import *
from cuda_utils import *
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from classes.checkpoint_wrapper import CheckpointWrapper
from time import time
from classes.optimizer import *
from os.path import join
from datetime import datetime
from tqdm import tqdm

device = 0
cuda.select_device(device)
print("DEVICE:", device)



########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
beta_cloud = loadmat(join("data", "smoke.mat"))["data"] * 10
beta_cloud = np.ascontiguousarray(np.rot90(beta_cloud, axes=(2,1)))
beta_cloud =np.roll(beta_cloud, axis=0, shift=-15)
# beta_cloud = beta_cloud.T
# beta_cloud = loadmat(join("data", "rico2.mat"))["vol"]
beta_cloud = beta_cloud.astype(float_reg)
# beta_cloud *= (127/beta_cloud.max())
# Grid parameters #
# bounding box
voxel_size_x = 0.02
voxel_size_y = 0.02
voxel_size_z = 0.02
edge_x = voxel_size_x * beta_cloud.shape[0]
edge_y = voxel_size_y * beta_cloud.shape[1]
edge_z = voxel_size_z * beta_cloud.shape[2]
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]])

print(beta_cloud.shape)
print(bbox)

beta_air = 0.004
# w0_air = 1.0 #0.912
w0_air = 0.912
# w0_cloud = 0.8 #0.9
w0_cloud = 0.99
g_cloud = 0.85

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################
#
N_cams = 9
ps = 200
pixels = np.array([ps, ps])
height_factor = 1.5
focal_length = 50e-3
sensor_size = np.array((50e-3, 50e-3)) / height_factor
volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.6
cameras = []
R = height_factor * edge_z

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

# mask parameters
image_threshold = 0.03
hit_threshold = 0.9
spp = 10000

# Simulation parameters
# Np_gt = int(1e9)
Np_gt = int(5e7)
Np_gt_batch = int(5e7)
batches = Np_gt // Np_gt_batch
print(f"Np_gt={Np_gt_batch*batches:.1f}")
Np = int(5e7)
resample_freq = int(sys.argv[1])
runtime = int(sys.argv[2]) #minutes
to_sort = int(sys.argv[3])

step_size = 5e9
beta_scalar_start = 2
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.05
iterations = 10000000
to_mask = True
tensorboard = True
tensorboard_freq = 5
beta_max = beta_cloud.max()



scene_seed = SceneSeed(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
# compile
print("compiling")
scene_seed.set_cloud_mask(beta_cloud>0)
Np_compile = 1000
scene_seed.init_cuda_param(Np_compile,init=True)
scene_seed.build_paths_list(Np_compile)
scene_seed.render(I_gt=0)
print("compiled")
# scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)

# visual.create_grid()
# visual.plot_cameras()
# visual.plot_medium()
# plt.show()
print("rendering GT:")
I_gt = np.zeros((N_cams,*cameras[0].pixels),dtype=float_reg)
scene_seed.init_cuda_param(Np_gt_batch,init=True)
for _ in tqdm(range(batches)):
    cuda_paths = scene_seed.build_paths_list(Np_gt_batch)
    I_gt += scene_seed.render(cuda_paths)
    del(cuda_paths)
I_gt /= batches
cuda_paths = None
max_val = np.max(I_gt, axis=(1,2))
# visual.plot_images(I_gt, "GT")
# plt.show()

print("Calculating Cloud Mask")
cloud_mask = scene_seed.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold, spp=spp)
mask_grader(cloud_mask, beta_gt>0.1, beta_gt)
scene_seed.set_cloud_mask(cloud_mask)
# exit()
# beta_scalar_init = scene_rr.find_best_initialization(beta_gt, I_gt,0,30,10,Np_gt,True)

scene_seed.init_cuda_param(Np, init=True)
alpha = 0.9
beta1 = 0.9
beta2 = 0.999
start_iter = 1
# optimizer = SGD(volume,step_size)
beta_mean = np.mean(beta_cloud[volume.cloud_mask])
print("beta_mean:",beta_mean)
optimizer = MomentumSGD(volume, step_size, alpha, beta_mean, beta_max)
# optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_mean, beta_max, 1)



if tensorboard:
    tb = TensorBoardWrapper(I_gt, beta_gt, title= datetime.now().strftime("%d%m-%H%M-%S")+f"_smoke_Nr={resample_freq}_ss={step_size:.2e}_tosort={int(to_sort)}")
    cp_wrapper = CheckpointWrapper(scene_seed, optimizer, Np_gt, Np, rr_depth, rr_stop_prob, None, None, resample_freq, step_size, iterations,
                                   tensorboard_freq, tb.train_id, image_threshold, hit_threshold, spp)
    tb.add_scene_text(str(cp_wrapper))
    pickle.dump(cp_wrapper, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    print("Checkpoint wrapper has been saved")
    tb.update_gt(I_gt)



# Initialization
beta_init = np.zeros_like(beta_cloud)
beta_init[volume.cloud_mask] = beta_scalar_start
# beta_init[volume.cloud_mask] = beta_cloud[volume.cloud_mask]
volume.set_beta_cloud(beta_init)
beta_opt = volume.beta_cloud
loss = 1
start_loop = time()
i = 0
reset_timer = time()
timer = 30 * 60
while time() - start_loop < runtime*60:
    if time() - reset_timer > timer:
        optimizer.reset()
        reset_timer = time()
        print("resettttttttttttttttttt")

# for iter in range(iterations):
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    max_dist = np.max(abs_dist)
    rel_dist1 = relative_distance(beta_cloud, beta_opt)

    print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps}")

    if i % resample_freq == 0:

        print("RESAMPLING PATHS ")
        start = time()
        scene_seed.build_paths_list(Np, to_print=False, to_sort=to_sort)
        end = time()
        print(f"building path list took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_seed.render( I_gt=I_gt, to_print=False)
    total_grad *= (ps*ps)
    end = time()
    print(f"rendering took: {end-start}")

    dif = (I_opt - I_gt).reshape(1,1,1, N_cams, *scene_seed.pixels_shape)
    grad_norm = np.linalg.norm(total_grad)

    start = time()
    optimizer.step(total_grad)

    loss = 0.5 * np.sum(dif * dif)

    if tensorboard and i % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, max_dist, rel_dist1, Np, i, time()-start_loop)

    i+=1


