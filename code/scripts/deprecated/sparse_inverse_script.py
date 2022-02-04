from classes.deprecated.scene_sparse import *
from camera import *
from classes.visual import *
from deprecated.phase_function import HGPhaseFunction
from utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper

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
# beta_cloud = cloud_loader("clouds_dist.mat", beta_max=5)
beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
print(beta_cloud)

beta_air = 0.1
w0_air = 1.0
w0_cloud = 0.8

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
phase_function = HGPhaseFunction(g=0.5)
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################


focal_length = 60e-3
sensor_size = np.array((40e-3, 40e-3))
ps = 30
pixels = np.array((ps, ps))

t1 = np.array([0.5, 0.5, 2])
euler_angles1 = np.array((180, 0, 0))
camera1 = Camera(t1, euler_angles1, focal_length, sensor_size, pixels)

t2 = np.array([0.5, 0.9, 2])
euler_angles2 = np.array((160, 0, 0))
camera2 = Camera(t2, euler_angles2, focal_length, sensor_size, pixels)
#
# t3 = np.array([0.5, 0.9, -1])
# euler_angles3 = np.array((20, 0, 0))
# camera3 = Camera(t3, euler_angles3, focal_length, sensor_size, pixels)
#
# cameras = [camera1, camera2]
# N_cams = len(cameras)
N_cams = 5
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
scene = Scene(volume, cameras, sun_angles, phase_function)

scene_sparse = SceneSparse(volume, cameras, sun_angles, phase_function)
visual = Visual_wrapper(scene)

plot_3D = False
if plot_3D:
    visual.plot_cloud()
    visual.create_grid()
    visual.plot_cameras()
    plt.show()


Np = int(1e5)
Np_gt = int(1e5)
Ns = 15
iteration = 5000
resample_freq = 10
step_size = 1e10
I_gt = scene.render(Np_gt, Ns)
# paths = scene_sparse.build_paths_list(Np*10, Ns)
# I_gt = scene_sparse.render(paths, False)
max_val = np.max(I_gt, axis=(1,2))
visual.plot_images(I_gt, max_val, "GT")
plt.show()

# Cloud mask (GT for now)
cloud_mask = beta_cloud > 0
volume.set_mask(cloud_mask)

beta_init = np.zeros_like(beta_cloud)
# beta_init[cloud_mask] = np.mean(beta_cloud[cloud_mask])
beta_opt = np.copy(beta_init)
tensorboard = True
tensorboard_freq = 1



if tensorboard:
    scene_descr = str(scene_sparse)
    scene_descr += f"Simualtion Parameters:  \nNp_gt={Np_gt:.2E}  \nNp={Np:.2E}  \nNs={Ns}  \niterations={iteration}  \n" \
                    f"step_size={step_size:.2E} \nresample_freq={resample_freq}  \ntensorboard_freq={tensorboard_freq}"
    tb = TensorBoardWrapper(I_gt, beta_gt, scene_descr)

for iter in range(iteration):
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    mean_dist = np.mean(abs_dist)
    max_dist = np.max(abs_dist)
    rel_dist = np.linalg.norm(beta_opt - beta_cloud)/np.linalg.norm(beta_cloud)
    print(f"mean_dist = {mean_dist}, max_dist={max_dist}, rel_dist={rel_dist}")
    if iter > 100:
        Np = int(1e5)
        resample_freq = 30
    beta_opt[beta_opt<0] = 0
    volume.set_beta_cloud(beta_opt)
    if iter % resample_freq == 0:
        if iter > 0:
            visual.plot_images(I_opt, max_val, f"iter:{iter}")
            plt.show()
            print(f"iter {iter}")
            # print(beta_opt)
            print()
        paths = scene_sparse.build_paths_list(Np, Ns)

    I_opt, total_grad = scene_sparse.render(paths, differentiable=True)
    dif = (I_opt - I_gt).reshape(1,1,1, N_cams, pixels[0], pixels[1])

    total_grad *= dif
    total_grad = np.sum(total_grad, axis=(3,4,5)) / N_cams
    total_grad *= cloud_mask
    beta_opt -= step_size * total_grad
    loss = 0.5 * np.sum(dif ** 2)
    print(f"loss = {loss}, grad_norm={np.linalg.norm(total_grad)}")
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, mean_dist, max_dist, rel_dist, iter)

    # if iter % tensorboard_freq == 0:
    #     update_tensorboard(writer, loss, iter)



