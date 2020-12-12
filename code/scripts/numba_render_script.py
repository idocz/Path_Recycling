from classes.scene import *
from classes.scene_sparse import *
from classes.scene_numba import *
from classes.camera import *
from classes.visual import *
from utils import construct_beta
from time import time
import matplotlib.pyplot as plt
from classes.phase_function import *
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
    ps = 1
    pixels = np.array((ps, ps))

    t1 = np.array([0.5, 0.5, 2])
    euler_angles1 = np.array((180, 0, 0))
    camera1 = Camera(t1, euler_angles1, focal_length, sensor_size, pixels)

    t2 = np.array([0.5, 0.9, 2])
    euler_angles2 = np.array((160, 0, 0))
    camera2 = Camera(t2, euler_angles2, focal_length, sensor_size, pixels)

    cameras = [camera1,camera2]

    scene = Scene(volume, cameras, sun_angles, phase_function)
    scene_sparse = SceneSparse(volume, cameras, sun_angles, phase_function)
    scene_numba = SceneNumba(volume, cameras, sun_angles, g)

    Np = int(1e6)
    Ns = 15

    plot_scene = True
    numba_render = True
    sparse_render = True
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
    if numba_render:
        print("####### sparse renderer ########")
        print("generating paths")

        volume.set_beta_cloud(fake_cloud)
        paths = scene_numba.build_paths_list(Np, Ns)
        I_total = scene_numba.render(paths)
        max_val = np.max(I_total, axis=(1,2))
        start = time()
        print(f" rendering took: {time() - start}")
        print(I_total)
        visual.plot_images(I_total, max_val, "sparse")
        plt.show()

    if sparse_render:
        print("####### sparse renderer ########")
        print("generating paths")

        volume.set_beta_cloud(fake_cloud)
        paths = scene_sparse.build_paths_list(Np, Ns)
        volume.set_beta_cloud(beta_cloud)
        print(" start rendering...")
        start = time()
        I_total = scene_sparse.render(paths)
        if max_val is None:
            max_val = np.max(I_total, axis=(1, 2))

        print(f" rendering took: {time() - start}")
        print(I_total)
        visual.plot_images(I_total, max_val, "sparse")
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