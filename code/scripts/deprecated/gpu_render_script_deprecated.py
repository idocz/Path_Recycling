from classes.deprecated.scene_numba import *
from classes.scene_gpu import *
from classes.camera import *
from classes.visual import *
from time import time
import matplotlib.pyplot as plt
from classes.phase_function import *
from scipy.io import loadmat
from os.path import join
def main():
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
    sun_angles = np.array([180, 0]) * (np.pi/180)


    #####################
    # Volume parameters #
    #####################
    # construct betas
    grid_size = 3
    beta_air = 0.1
    beta_cloud = np.array([[[0, 0, 0],
                            [0, 2, 0],
                            [0, 0, 0]],

                           [[0, 3, 0],
                            [3, 4, 3],
                            [0, 3, 0]],

                           [[0, 0, 0],
                            [0, 5, 0],
                            [0, 0, 0]]])
    beta_cloud = loadmat(join("data", "rico.mat"))["beta"]
    w0_air = 0.8
    w0_cloud = 0.7
    # Declerations
    grid = Grid(bbox, beta_cloud.shape)
    volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
    # phase_function = UniformPhaseFunction()
    g = 0.5
    phase_function = HGPhaseFunction(g)
    print(volume.betas)
    #######################
    # Cameras declaration #
    #######################
    N_cams = 2
    focal_length = 70e-3
    sensor_size = np.array((80e-3, 80e-3))
    ps = 55
    pixels = np.array((ps, ps))

    t1 = np.array([0.5, 0.5, 2])
    euler_angles1 = np.array((180, 0, 0))
    camera1 = Camera(t1, euler_angles1, focal_length, sensor_size, pixels)

    t2 = np.array([0.5, 0.9, 2])
    euler_angles2 = np.array((160, 0, 0))
    camera2 = Camera(t2, euler_angles2, focal_length, sensor_size, pixels)

    cameras = [camera1,camera2]

    scene = Scene(volume, cameras, sun_angles, phase_function)
    scene_numba = SceneNumba(volume, cameras, sun_angles, g)
    scene_gpu = SceneGPU(volume, cameras, sun_angles, g)

    Np = int(1e5)
    Ns = 15

    plot_scene = False
    gpu_render = False
    numba_render = True
    basic_render = False


    visual = Visual_wrapper(scene)
    if plot_scene:
        visual.plot_cloud()
        visual.create_grid()
        visual.plot_cameras()
        plt.show()

    fake_cloud = beta_cloud * 1#.5
    # fake_cloud = construct_beta(grid_size, False, beta + 2)

    max_val = None
    if gpu_render:
        print("####### gpu renderer ########")
        print("generating paths")

        volume.set_beta_cloud(fake_cloud)
        cuda_paths = scene_gpu.build_paths_list(Np, Ns)
        I_total, total_grad = scene_gpu.render(cuda_paths)
        start = time()
        I_total, total_grad = scene_gpu.render(cuda_paths)
        print(f" rendering took: {time() - start}")
        max_val = np.max(I_total, axis=(1, 2))
        # print(I_total)
        visual.plot_images(I_total, max_val, "gpu")
        plt.show()

    if numba_render:
        print("####### numba renderer ########")
        print("generating paths")

        volume.set_beta_cloud(fake_cloud)
        paths = scene_numba.build_paths_list(Np, Ns)
        I_total = scene_numba.render(paths)
        max_val = np.max(I_total, axis=(1,2))
        start = time()
        print(f" rendering took: {time() - start}")
        # print(I_total)
        visual.plot_images(I_total, max_val, "numba")
        plt.show()



    if basic_render:
        print("####### basic renderer ########")
        print(" start rendering...")
        start = time()
        I_total = scene.render(Np, Ns)
        print(I_total[0].T)
        print(I_total[1].T)
        print(f" rendering took: {time() - start}")
        if max_val is None:
            max_val = np.max(I_total, axis=(1, 2))
        visual.plot_images(I_total, max_val, "basic")
        plt.show()



if __name__ == '__main__':
    main()