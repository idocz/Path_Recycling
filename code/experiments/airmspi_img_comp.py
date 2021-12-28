from classes.scene_airmspi_backward_recycling import *
from classes.camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import pickle
from classes.optimizer import *
from os.path import join
cuda.select_device(2)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']


# checkpoint_id = "1211-2016-37"
# checkpoint_id = "1311-1602-58"
# checkpoint_id = "1411-1314-38"
# checkpoint_id = "1411-1654-05"
# checkpoint_id = "1411-1807-40"
checkpoint_id = "1511-1750-31"
iter = 700
# checkpoint_id = "1411-1335-03"
dict = np.load(join("checkpoints",checkpoint_id,"data",f"opt_{iter}.npz"))
beta_cloud = dict["betas"]
a_file = open(join("data", "airmspi_data_modified.pkl"), "rb")
airmspi_data = pickle.load(a_file)
exclude_index = 7

dir_to_save = join("data","res",f"{checkpoint_id}_{iter}_airmspi.mat")
savemat(dir_to_save, {"vol": beta_cloud})


zenith = np.deg2rad(airmspi_data["sun_zenith"])
azimuth = np.deg2rad(airmspi_data["sun_azimuth"])
dir_x = np.sin(zenith)*np.cos(azimuth)
dir_y = np.sin(zenith)*np.sin(azimuth)
dir_z = np.cos(zenith)
sun_direction = np.array([dir_x, dir_y, dir_z])
downscale = 1
ocean_albedo = 0.05
background = np.load(join("data",f"background_{str(ocean_albedo).split('.')[-1]}_{downscale}_9.npy"))

sun_intensity = -1/np.cos(zenith)
TOA = 20
#####################
# grid parameters #
#####################
# grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
grid = Grid(airmspi_data["bbox"],beta_cloud.shape)
# grid = Grid(airmspi_data["bbox"], beta_cloud.shape)
print(grid.bbox)



#####################
# Volume parameters #
#####################
# construct betas
beta_air = 0.004


cloud_mask = airmspi_data["cloud_mask"]



w0_air = 0.912
w0_cloud = 0.99
# Declerations
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
g_cloud = 0.85
#######################
# Cameras declaration #
#######################

# exit()
N_cams = 9

resolutions = airmspi_data["resolution"]#[:3]
I_gt = airmspi_data["images"]#[:3]
camera_array_list = airmspi_data["camera_array_list"]#[:1]
total_num_cam = len(camera_array_list)
camera_array_list = [np.ascontiguousarray(camera_array[::downscale, ::downscale, :6]) for camera_array in camera_array_list]

# Simulation parameters
Np = int(7e8)
rr_depth = 20
rr_stop_prob = 0.1


scene_airmspi = SceneAirMSPI(volume, camera_array_list, sun_direction, sun_intensity, TOA, background, g_cloud, rr_depth, rr_stop_prob)
pad_shape = scene_airmspi.pixels_shape

I_gt_pad = np.zeros((total_num_cam, *scene_airmspi.pixels_shape), dtype=float_reg)
for cam_ind in range(total_num_cam):
    pix_shape = scene_airmspi.pixels_shapes[cam_ind]
    I_gt_pad[cam_ind, :pix_shape[0], :pix_shape[1]] = I_gt[cam_ind][::downscale, ::downscale]

N_repeat = 10
scene_airmspi.init_cuda_param(Np, init=True)
for cam_ind in tqdm(np.arange(N_cams)):
    # for cam_ind in tqdm([exclude_index]):
    cam_inds = np.array([cam_ind])
    spp_map = np.copy(I_gt_pad)
    for cam_ind2 in range(total_num_cam):
        if cam_ind2 != cam_ind:
            spp_map[cam_ind2,:,:] = 0
    spp_map = (Np * spp_map / np.sum(spp_map)).astype(np.uint32)

    scene_airmspi.create_job_list(spp_map)
    I_opt = np.zeros_like(I_gt_pad)
    for _ in range(N_repeat):
        scene_airmspi.build_path_list(Np, cam_inds, False)
        ex_pix = scene_airmspi.pixels_shapes[cam_ind]
        I_opt += scene_airmspi.render()
    # I_concat = np.concatenate([I_opt, I_gt_pad[exclude_index,:ex_pix[0],:ex_pix[1]]], axis=1)
    I_opt /= N_repeat
    I_opt_norm = I_opt[cam_ind,:ex_pix[0], :ex_pix[1]]
    I_gt_norm = I_gt_pad[cam_ind,:ex_pix[0],:ex_pix[1]]
    min_val = np.min([np.min(I_opt_norm), np.min(I_gt_norm)])
    max_val = np.min([np.max(I_opt_norm), np.max(I_gt_norm)])
    I_opt_norm = (I_opt_norm-min_val)/(max_val-min_val)
    I_gt_norm = (I_gt_norm-min_val)/(max_val-min_val)
    plt.figure()
    ax = plt.subplot(1,2,1)
    ax.imshow(I_opt_norm, cmap="gray")
    ax.axis("off")
    ax = plt.subplot(1,2,2)
    ax.imshow(I_gt_norm, cmap="gray")
    ax.axis("off")
    # plt.imshow(I_concat, cmap="gray")
    plt.tight_layout()
    filename = f"{checkpoint_id}_{iter}_images_airmspi_{cam_ind}"
    if cam_ind == exclude_index:
        filename +="_excluded.pdf"
    else:
        filename += ".pdf"
    plt.savefig(join("experiments","plots","airmspi_results",filename), bbox_inches="tight")
    plt.show()



    print(f"relative error = {relative_distance(I_gt_norm,I_opt_norm)}")
    print(f"relative bias = {relative_bias(I_gt_norm,I_opt_norm)}")

    if cam_ind != exclude_index:
        continue
    # mask = I_opt_norm !=0
    X = I_gt_norm.reshape(-1)
    Y = I_opt_norm.reshape(-1)
    max_val = np.max([X.max(), Y.max()])
    # N = int(Y.shape[0] * N)
    N = 1500
    relative_point_num = N/Y.shape[0]
    print("N=",N)
    print()
    rand_inds = np.random.randint(0,X.shape[0],N)
    fig = plt.figure(figsize=(5,5))
    plt.scatter(X[rand_inds], Y[rand_inds])
    # plt.scatter(X, Y, s=3)

    plt.plot([0,max_val], [0,max_val], color="red")
    # plt.plot([0,10], [0,10])
    fs = 20
    plt.xlabel("Ground Truth", fontsize=fs)
    plt.ylabel("Estimated", fontsize=fs)
    # plt.title(f"iter: {iter}")
    plt.xticks([])
    plt.yticks([])
    # plt.axes().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(join("experiments","plots","airmspi_results",f"{checkpoint_id}_{iter}_airmspi_scatter_plot.pdf"))
    plt.show()
