import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene_airmspi_backward import *
from classes.camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from classes.checkpoint_wrapper import CheckpointWrapper
from time import time
from classes.optimizer import *
from os.path import join
from tqdm import tqdm
cuda.select_device(0)


a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
airmspi_data = pickle.load(a_file)

zenith = airmspi_data["sun_zenith"]
azimuth = airmspi_data["sun_azimuth"]
dir_x = -(np.sin(zenith)*np.cos(azimuth))
dir_y = -(np.sin(zenith)*np.sin(azimuth))
dir_z = -np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
downscale = 8
# sun_intensity = 1e1/7
sun_intensity = 5e0

#####################
# grid parameters #
#####################
# grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
grid = Grid(airmspi_data["bbox"], np.array([60, 60, 20]))
# grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
print(grid.bbox)




#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004


cloud_mask = airmspi_data["cloud_mask"]
beta_init = np.zeros(grid.shape, dtype=float_reg)


w0_air = 0.912
w0_cloud = 0.99
N_cams = 1
# Declerations
volume = Volume(grid, beta_init, beta_air, w0_cloud, w0_air)
g_cloud = 0.85
#######################
# Cameras declaration #
#######################


resolutions = airmspi_data["resolution"]#[:3]
I_gt = airmspi_data["images"]#[:3]
camera_array_list = airmspi_data["camera_array_list"]#[:1]
camera_array_list = [np.ascontiguousarray(camera_array[::downscale, ::downscale, :6]) for camera_array in camera_array_list]
# camera_array_list = [np.ascontiguousarray(camera_array[:, :, :6]) for camera_array in camera_array_list]

# Simulation parameters
Np_max = int(5e8)
Np = int(5e6)
resample_freq = 1
step_size = 5e2
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.1
iterations = 10000000
to_mask = True
tensorboard = False
tensorboard_freq = 10
beta_max = 100
win_size = 40

scene_airmspi = SceneAirMSPI(volume, camera_array_list, sun_direction, sun_intensity, g_cloud, rr_depth, rr_stop_prob)
pad_shape = scene_airmspi.pixels_shape

compute_mask = False

# if compute_mask:
#     print("Calculating Cloud Mask")
#     image_threshold = np.ones(N_cams, dtype=float_reg) * 0.25
#     image_threshold[0] = 0.2
#     # image_threshold[7] = 0.35
#     # image_threshold[8] = 0.48
#     hit_threshold = 0.9
#     spp = 100000
#     cloud_mask = scene_airmspi.space_curving(I_gt_pad, image_threshold=image_threshold, hit_threshold=hit_threshold,
#                                              camera_array_list=camera_array_list, spp=spp)
#     print(cloud_mask.mean())
# cloud_mask = scene_rr.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold, spp=spp)
# mask_grader(cloud_mask, beta_gt>0.1, beta_gt)
scene_airmspi.set_cloud_mask(cloud_mask)
beta_init[cloud_mask] = 2
volume.set_beta_cloud(beta_init)
# beta_scalar_init = scene_rr.find_best_initialization(beta_gt, I_gt,0,30,10,Np_gt,True)

# downscaling
# I_gt_pad = I_gt_pad[:,::downscale, ::downscale]

# plot GT
visual = Visual_wrapper(grid)
# visual.plot_images_airmspi(I_gt_pad, resolutions, "GT", downscale)
plt.show()

alpha = 0.9
beta1 = 0.9
beta2 = 0.999
start_iter = 100
scaling_factor = 1.5
optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_max, beta_max, 1)


# if tensorboard:
#     tb = TensorBoardWrapper(I_gt_pad, None)
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
cam_ind = 0

pixel_shape = scene_airmspi.pixels_shapes[cam_ind]
scene_airmspi.init_cuda_param(Np, init=True)

I_gt_downscaled = I_gt[cam_ind][::downscale, ::downscale]
# I_gt_downscaled = I_gt[cam_ind]
plt.figure()
plt.imshow(I_gt_downscaled, cmap="gray")
plt.title("GT")
plt.show()
spp_map = (Np*I_gt_downscaled/np.sum(I_gt_downscaled)).astype(np.uint32)
for iter in range(iterations):
    print(f"\niter {iter}")

    print(f"loss={loss:.2e}, Np={Np:.2e}, beta_mean={beta_opt[cloud_mask].mean()}, beta_max={beta_opt.max()}, counter={non_min_couter}")

    if iter % resample_freq == 0:
        if non_min_couter >= win_size and iter > start_iter:
            if Np < Np_max :
                Np = int(Np * scaling_factor)
                # resample_freq = 30
                non_min_couter = 0
                step_size *= scaling_factor
                if Np > Np_max:
                    Np = Np_max

        print("RESAMPLING PATHS ")
        start = time()
        scene_airmspi.build_path_list(Np, cam_ind, spp_map, False)
        print(f"rendering image {cam_ind}")
        I_total, total_grad = scene_airmspi.render(I_gt_downscaled)
        end = time()
        print(f"building path list took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_airmspi.render(I_gt=I_gt_downscaled)
    if iter % 10 == 0:
        plt.figure()
        plt.imshow(I_opt, cmap="gray")
        plt.title(f"Iter {iter}: GPU AirMSPI")
        plt.show()


    end = time()
    print(f"rendering took: {end-start}")


    dif = (I_opt - I_gt_downscaled).reshape(1,1,1, N_cams, *pixel_shape)
    grad_norm = np.linalg.norm(total_grad)
    print(f"I_gt_mean:{I_gt_downscaled[I_gt_downscaled!=0].mean()}, I_opt_mean:{I_opt[I_opt!=0].mean()}, grad_norm:{grad_norm:.2e}")
    # updating beta
    beta_opt -= step_size*total_grad
    beta_opt[beta_opt >= beta_max] = beta_max
    beta_opt[beta_opt < 0] = 0
    # loss calculation
    start = time()
    # optimizer.step(total_grad)
    # print("gradient step took:",time()-start)
    loss = 0.5 * np.mean(dif * dif)
    # loss = 0.5 * np.sum(np.abs(dif))
    if loss < min_loss:
        min_loss = loss
        non_min_couter = 0
    else:
        non_min_couter += 1
    # print(f"loss = {loss}, grad_norm={grad_norm}, max_grad={np.max(total_grad)}")

    # Writing scalar and images to tensorboard
    # if tensorboard and iter % tensorboard_freq == 0:
    #     tb.update(beta_opt, I_opt, loss, None, None, Np, iter, time()-start_loop)




