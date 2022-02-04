import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from deprecated.scene_airmspi import *
from camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from time import time
from classes.optimizer import *
from os.path import join

cuda.select_device(0)


a_file = open(join("data", "airmspi_data.pkl"), "rb")
airmspi_data = pickle.load(a_file)

zenith = airmspi_data["sun_zenith"]
azimuth = airmspi_data["sun_azimuth"]
dir_x = -(np.sin(zenith)*np.cos(azimuth))
dir_y = -(np.sin(zenith)*np.sin(azimuth))
dir_z = -np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
sun_intensity = 1e7
downscale = 4


load_shdom = True
if load_shdom:
    beta_gt = loadmat(join("data", "FINAL_3D_extinction.mat"))["estimated_extinction"][:,:,:,0]

if load_shdom:
    beta_init = np.zeros_like(beta_gt)
else:
    beta_init = np.zeros(np.array([10, 10, 10], dtype=float_reg))

#####################
# grid parameters #
#####################
# grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
grid = Grid(airmspi_data["bbox"], beta_init.shape)
# grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
print(grid.bbox)




#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004

# cloud_mask = beta_cloud>0
cloud_mask = airmspi_data["cloud_mask"]
# cloud_mask = np.ones_like(cloud_mask)

# beta_init[volume.cloud_mask] = beta_mean
# beta_init[volume.cloud_mask] = beta_scalar_init


w0_air = 0.912
w0_cloud = 0.99
# w0_air = 1
# w0_cloud = 1

# Declerations
volume = Volume(grid, beta_init, beta_air, w0_cloud, w0_air)
g_cloud = 0.85
#######################
# Cameras declaration #
#######################

ts = airmspi_data["ts"]
Ps = airmspi_data["Ps"]
resolutions = airmspi_data["resolution"]
# I_gt = airmspi_data["images"]
camera_array_list_fromdata = airmspi_data["camera_array_list"]
N_cams = len(ts)
cameras = []
for k in range(N_cams):
    cameras.append(AirMSPICamera(resolutions[k], ts[k], Ps[k]))
Np = int(1e8)
Ns = 15
rr_depth = 10
rr_stop_prob = 0.1

volume.set_mask(cloud_mask)
scene_airmspi = SceneAirMSPI(volume, cameras, sun_direction, sun_intensity, g_cloud, rr_depth, rr_stop_prob, downscale)

scene_airmspi.set_cloud_mask(volume.cloud_mask)
# Simulation parameters
Np_max = int(1e8)
Np = int(5e7)
Np_gt = int(1e8)
resample_freq = 10
step_size = 1e-18
# Ns = 15
rr_depth = 10
rr_stop_prob = 0.1
iterations = 10000000
to_mask = True
tensorboard = True
tensorboard_freq = 10
beta_max = 10
win_size = 100


# I_gt = airmspi_data["images"]
# pad_shape = scene_airmspi.pixels_shape//downscale
# I_gt_pad = np.zeros((N_cams, *pad_shape), dtype=float_reg)
# for k in range(N_cams):
#     I_gt_downscale = I_gt[k][::downscale, ::downscale]
#     fix_i = pad_shape[0] - I_gt_downscale.shape[0]
#     if fix_i == -1:
#         I_gt_downscale = I_gt_downscale[:-1, :]
#     fix_j = pad_shape[1] - I_gt_downscale.shape[1]
#     if fix_j == -1:
#         I_gt_downscale = I_gt_downscale[:, :-1]
#
#     I_gt_pad[k,:I_gt_downscale.shape[0], :I_gt_downscale.shape[1]] = I_gt_downscale

scene_airmspi.downscale = 1
cuda_paths = scene_airmspi.build_paths_list(Np_gt)
I_gt = scene_airmspi.render(cuda_paths)

scene_airmspi.downscale = downscale

print("Calculating Cloud Mask")
#mask parameters

# pad_shape = scene_airmspi.pixels_shape
# I_gt_pad = np.zeros((N_cams, *pad_shape), dtype=float_reg)
camera_array_list = np.zeros((N_cams, *scene_airmspi.pixels_shape, 6), dtype=float_reg)
for k in range(N_cams):
    # I_gt_pad[k,:resolutions[k][0], :resolutions[k][1]]= I_gt[k]
    camera_array_list[k, :resolutions[k][0], :resolutions[k][1], :] = camera_array_list_fromdata[k][:,:,:6]

compute_mask = True
if compute_mask:
    print("Calculating Cloud Mask")
    image_threshold = np.ones(N_cams, dtype=float_reg) * 0.25
    image_threshold[0] = 0.4
    image_threshold[7-3] = 0.35
    image_threshold[8-3] = 0.48
    hit_threshold = 0.9
    spp = 100000
    cloud_mask = scene_airmspi.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold,
                                             camera_array_list=camera_array_list, spp=spp)
    mask_grader(cloud_mask, beta_gt>0.1, beta_gt)
# cloud_mask = scene_rr.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold, spp=spp)
# cloud_mask = airmspi_data["cloud_mask"]
scene_airmspi.set_cloud_mask(cloud_mask)
beta_init[cloud_mask] = 2
volume.set_beta_cloud(beta_init)
# beta_scalar_init = scene_rr.find_best_initialization(beta_gt, I_gt,0,30,10,Np_gt,True)

# downscaling
I_gt = I_gt[:,::downscale, ::downscale]
cuda_paths = None

# plot GT
visual = Visual_wrapper(grid)
visual.plot_images_airmspi(I_gt, resolutions, "GT", downscale)
plt.show()

scene_airmspi.init_cuda_param(Np)
alpha = 0.9
beta1 = 0.9
beta2 = 0.999
start_iter = 100
scaling_factor = 1.5
optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_max, beta_max, 1)


if tensorboard:
    tb = TensorBoardWrapper(I_gt, None)
    # cp_wrapper = CheckpointWrapper(scene_airmspi, optimizer, Np, Np, rr_depth, rr_stop_prob, None, None, resample_freq, step_size, iterations,
    #                                tensorboard_freq, tb.train_id, image_threshold, hit_threshold, spp)
    # tb.add_scene_text(str(cp_wrapper))
    # pickle.dump(cp_wrapper, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    # print("Checkpoint wrapper has been saved")
# scene_rr.upscale_cameras(ps)




# grad_norm = None
non_min_couter = 0
next_phase = False
min_loss = 1000#

beta_opt = volume.beta_cloud
loss = 1000
start_loop = time()
for iter in range(iterations):
    print(f"\niter {iter}")
    rel_dist1 = relative_distance(beta_gt, beta_opt)
    print(f"loss={loss}, Np={Np:.2e}, rel_dist1={rel_dist1}, beta_max={beta_opt.max()}, counter={non_min_couter}")
    if iter % resample_freq == 0:
        if non_min_couter >= win_size and iter > start_iter:
            if Np < Np_max :
                Np = int(Np * scaling_factor)
                resample_freq = 30
                non_min_couter = 0
                # step_size *= scaling_factor
                if Np > Np_max:
                    Np = Np_max

        print("RESAMPLING PATHS ")
        start = time()
        del(cuda_paths)
        cuda_paths = scene_airmspi.build_paths_list(Np)
        end = time()
        print(f"building path list took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_airmspi.render(cuda_paths, I_gt=I_gt)
    if iter % 10 == 0:
        visual.plot_images_airmspi(I_opt, resolutions, f"Iter {iter}: GPU AirMSPI", downscale)
        plt.show()


    end = time()
    print(f"rendering took: {end-start}")


    dif = (I_opt - I_gt).reshape(1,1,1, N_cams, *scene_airmspi.pixels_shape_downscaled)
    grad_norm = np.linalg.norm(total_grad)
    print(f"I_gt_mean:{I_gt[I_gt!=0].mean()}, I_opt_mean:{I_opt[I_opt!=0].mean()}, grad_norm:{grad_norm}")
    # updating beta
    beta_opt -= step_size*total_grad
    beta_opt[beta_opt >= beta_max] = beta_max
    beta_opt[beta_opt < 0] = 0
    # loss calculation
    # start = time()
    # optimizer.step(total_grad)
    # print("gradient step took:",time()-start)
    loss = 0.5 * np.sum(dif * dif)
    # loss = 0.5 * np.sum(np.abs(dif))
    if loss < min_loss:
        min_loss = loss
        non_min_couter = 0
    else:
        non_min_couter += 1
    # print(f"loss = {loss}, grad_norm={grad_norm}, max_grad={np.max(total_grad)}")

    # Writing scalar and images to tensorboard
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, None, rel_dist1, Np, iter, time()-start_loop)




