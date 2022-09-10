import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
# from classes.scene_airmspi_backward import *
from classes.scene_airmspi_backward_recycling import *
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


a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
beta_shdom = loadmat(join("data","shdom_exp","FINAL_3D_extinction.mat"))['estimated_extinction'][:,:,:,0]
airmspi_data = pickle.load(a_file)

zenith = np.deg2rad(airmspi_data["sun_zenith"])
azimuth = np.deg2rad(airmspi_data["sun_azimuth"])
dir_x = (np.sin(zenith)*np.cos(azimuth))
dir_y = (np.sin(zenith)*np.sin(azimuth))
dir_z = np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
print("sun direction:", sun_direction)
downscale = 1
# sun_intensity = 1e1/7
sun_intensity = -1/np.cos(zenith)
print("sun intensity:",sun_intensity)
# sun_intensity = 1
TOA = 20
ocean_albedo = 0.05
exclude_index = 7
background = np.load(join("data",f"background_{str(ocean_albedo).split('.')[-1]}_{downscale}_9.npy"))
# background[exlude_index,:,:] = 0
# background = 0
#####################
# grid parameters #
#####################
# grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
grid = Grid(airmspi_data["bbox"], np.array([50, 50, 50]))
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
# Declerations
volume = Volume(grid, beta_init, beta_air, w0_cloud, w0_air)
g_cloud = 0.85
#######################
# Cameras declaration #
#######################

cam_inds = np.arange(9).tolist()
cam_inds.remove(exclude_index)
cam_inds = np.array(cam_inds)
print(cam_inds)
# exit()
N_cams = len(cam_inds)

resolutions = airmspi_data["resolution"]#[:3]
I_gt = airmspi_data["images"]#[:3]
print([I.shape for I in I_gt])
camera_array_list = airmspi_data["camera_array_list"]#[:1]
total_num_cam = len(camera_array_list)
camera_array_list = [np.ascontiguousarray(camera_array[::downscale, ::downscale, :6]) for camera_array in camera_array_list]
# camera_array_list = [np.ascontiguousarray(camera_array[:, :, :6]) for camera_array in camera_array_list]

# Simulation parameters
Np_max = int(5e8)
Np = int(5e7)
resample_freq = 5
step_size = 5e11
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.1
iterations = 10000000
to_mask = True
tensorboard = False
tensorboard_freq = 10
beta_max = 35
update_max = 0.5
win_size = 10
max_grad_norm = np.inf

scene_airmspi = SceneAirMSPI(volume, camera_array_list, sun_direction, sun_intensity, TOA, background, g_cloud, rr_depth, rr_stop_prob)
pad_shape = scene_airmspi.pixels_shape




I_gt_pad = np.zeros((total_num_cam, *scene_airmspi.pixels_shape), dtype=float_reg)
for cam_ind in range(total_num_cam):
    pix_shape = scene_airmspi.pixels_shapes[cam_ind]
    I_gt_pad[cam_ind,:pix_shape[0],:pix_shape[1]] = I_gt[cam_ind][::downscale, ::downscale]

compute_mask = True
if compute_mask:
    print("Calculating Cloud Mask")
    image_threshold = np.ones(total_num_cam, dtype=float_reg) * 0.25
    image_threshold[0] = 0.35
    image_threshold[1] = 0.25
    image_threshold[2] = 0.2
    image_threshold[3] = 0.18
    image_threshold[4] = 0.22
    image_threshold[5] = 0.23
    image_threshold[6] = 0.23
    image_threshold[7] = 0.35
    image_threshold[8] = 0.4
    hit_threshold = 0.9
    spp = 1000
    cloud_mask = scene_airmspi.space_curving(I_gt_pad, image_threshold=image_threshold, hit_threshold=hit_threshold,
                                             spp=spp)
    print(cloud_mask.mean())
# exit()

I_gt_pad[exclude_index, :,:] = 0


scene_airmspi.set_cloud_mask(cloud_mask)
beta_init[cloud_mask] = 1
volume.set_beta_cloud(beta_init)




# plot GT
visual = Visual_wrapper(grid)
# visual.plot_images_airmspi(I_gt_pad, resolutions, "GT", downscale)
plt.show()

scaling_factor = 1.5



if tensorboard:
    tb = TensorBoardWrapper(I_gt_pad, None)
    print("Tensorboard wrapper was initiated")





# grad_norm = None
non_min_couter = 0
next_phase = False
min_loss = 1000#

beta_opt = volume.beta_cloud
loss = 1000
start_loop = time()
spp_map_freq = 100


# I_gt_downscaled = I_gt[cam_ind]
visual.plot_images(I_gt_pad, "GT")
plt.show()
spp_map = (Np*I_gt_pad/np.sum(I_gt_pad)).astype(np.uint32)
scene_airmspi.create_job_list(spp_map)
scene_airmspi.init_cuda_param(Np, init=True)
pix_shape = scene_airmspi.pixels_shape
resample = False
init_cuda = True
find_initialization = True
compute_spp = False
for iter in range(iterations):
    print(f"\niter {iter}")
    rel_dist = relative_distance(beta_shdom, beta_opt)
    print(f"loss={loss:.3e}, Np={Np:.2e}, rel_dist={rel_dist:.4f}, beta_mean={beta_opt[cloud_mask].mean()}, beta_max={beta_opt.max()}, counter={non_min_couter}")

    if iter % resample_freq == 0 or resample:
        resample = False
        if non_min_couter >= win_size :
            if Np < Np_max :
                print("\n\n\n INCREASING NP \n\n\n")
                Np = int(Np * scaling_factor)
                # resample_freq = 30
                non_min_couter = 0
                # step_size *= scaling_factor
                if Np > Np_max:
                    Np = Np_max
                spp_map = (Np*I_gt_pad/np.sum(I_gt_pad)).astype(np.uint32)
                scene_airmspi.create_job_list(spp_map)
                init_cuda = True
                compute_spp = True

        print("RESAMPLING PATHS ")
        start = time()
        compute_spp = (Np > 7e7 and iter % spp_map_freq == 0) and False
        scene_airmspi.build_path_list(Np, cam_inds, init_cuda, sort=True, compute_spp_map=compute_spp)
        compute_spp = False
        init_cuda = False
        end = time()
        print(f"building path list took: {end - start}")

    # differentiable forward modelI
    start = time()
    print(f"rendering images")
    I_opt, total_grad = scene_airmspi.render(I_gt_pad)
    end = time()
    print(f"rendering took: {end-start}")
    dif = (I_opt - I_gt_pad).reshape(1,1,1, total_num_cam, *pix_shape)
    grad_norm = np.linalg.norm(total_grad[cloud_mask])
    grad_mean = np.mean(total_grad[cloud_mask])

    update = -step_size*total_grad
    cond = np.abs(update)>update_max
    print(f"I_gt_mean:{I_gt_pad[I_gt_pad!=0].mean()}, I_opt_mean:{I_opt[I_opt!=0].mean()}, step_violation:{np.mean(cond[cloud_mask])}, step_mean:{update[cloud_mask].mean():.4f}, step_norm:{np.linalg.norm(update):.4f}, imgs_max:{I_gt_pad.max():.4f},{I_opt.max():.4f}")
    # update[cond] = update_max * np.sign(update[cond])
    if np.linalg.norm(update) > max_grad_norm:
        print(f"\n\n\nSTEP SKIPPED DUE TO LARGE GRAD NORM * STEPSIZE (>{max_grad_norm})\n\n\n")
        resample = True
        plt.figure()
        plt.hist(total_grad[cloud_mask]*step_size)
        plt.title(f"iter:{iter}")
        plt.show()
        continue

    # updating beta
    if find_initialization:
        total_grad[cloud_mask] = np.mean(total_grad[cloud_mask])

    beta_opt += update
    beta_opt[beta_opt >= beta_max] = beta_max
    beta_opt[beta_opt < 0] = 0
    # loss calculation
    loss = 0.5 * np.mean(dif * dif)
    if loss < min_loss:
        min_loss = loss
        non_min_couter = 0
    else:
        non_min_couter += 1

    # Writing scalar and images to tensorboard
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, None, rel_dist, Np, iter, time()-start_loop)




