from classes.scene import *
from classes.scene_graph import *
from classes.scene_sparse import *
from classes.camera import *
from classes.visual import *
from classes.phase_function import HGPhaseFunction
from utils import construct_beta
from time import time
import matplotlib.pyplot as plt

def main():
    ###################
    # Grid parameters #
    ###################
    # bounding box
    edge = 1
    bbox = np.array([[0, edge],
                     [0, edge],
                     [0, edge]])

    ########################
    # Atmosphere parameters#
    ########################
    sun_angles = np.array([180, 0]) * (np.pi/180)


    #####################
    # Volume parameters #
    #####################
    # construct betas
    grid_size = 3
    mid = grid_size // 2
    beta = 4
    # beta_cloud = construct_beta(grid_size, False, beta)
    # beta_cloud = np.zeros((grid_size,grid_size,grid_size))
    # beta_cloud[mid,mid,mid] = beta
    beta_cloud = np.ones((1,1,1)) * beta
    beta_air = 0.1
    w0_air = 0.8
    w0_cloud = 0.7
    # Declerations
    grid = Grid(bbox, beta_cloud.shape)
    volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
    phase_function = HGPhaseFunction(g=0.5)
    print(volume.betas)
    #######################
    # Cameras declaration #
    #######################
    N_cams = 1
    focal_length = 70e-3
    sensor_size = np.array((40e-3, 40e-3))
    ps = 1
    pixels = np.array((ps, ps))

    t1 = np.array([0.5, 0.5, 2])
    euler_angles1 = np.array((180, 0, 0))
    camera1 = Camera(t1, euler_angles1, focal_length, sensor_size, pixels)

    t2 = np.array([0.5, 0.9, 2])
    euler_angles2 = np.array((160, 0, 0))
    camera2 = Camera(t2, euler_angles2, focal_length, sensor_size, pixels)

    cameras = [camera1,camera2]

    # N_cams = 7
    # cameras = []
    # for cam_ind in range(N_cams):
    #     phi = 0
    #     theta = (-(N_cams // 2) + cam_ind) * 40
    #     theta_rad = theta * (np.pi / 180)
    #     t = 1.5 * theta_phi_to_direction(theta_rad, phi) + np.array([0.5, 0.5, 0.5])
    #     euler_angles = np.array((180, theta, 0))
    #     camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    #     cameras.append(camera)


    scene = Scene(volume, cameras, sun_angles, phase_function)
    scene_sparse = SceneSparse(volume, cameras, sun_angles, phase_function)

    Np = int(1e5)
    Ns = 15

    plot_scene = True
    sparse_render = True
    basic_render = True


    visual = Visual_wrapper(scene)
    if plot_scene:
        visual.plot_cloud()
        visual.create_grid()
        visual.plot_cameras()
        plt.show()

    fake_cloud = beta_cloud #* 1.5
    # fake_cloud = construct_beta(grid_size, False, beta + 2)
    if sparse_render:
        print("####### sparse renderer ########")
        print("generating paths")

        volume.set_beta_cloud(fake_cloud)
        paths = scene_sparse.build_paths_list(Np, Ns)
        volume.set_beta_cloud(beta_cloud)
        print(" start rendering...")
        start = time()
        I_total, total_grad = scene_sparse.render(paths, differentiable=True)
        max_val = np.max(I_total, axis=(1, 2))
        print(f" rendering took: {time() - start}")
        print(f"I_total={I_total}")
        print(f"total_grad={total_grad}")
        visual.plot_images(I_total, max_val, "sparse")
        plt.show()


    if basic_render:
        print("####### basic renderer ########")
        print(" start rendering...")
        start = time()
        I_total,total_grad = scene.render_differentiable(Np, Ns)
        print(f" rendering took: {time() - start}")
        print(f"I_total={I_total}")
        print(f"total_grad={total_grad}")
        visual.plot_images(I_total,max_val, "basic")
        plt.show()



if __name__ == '__main__':
    main()