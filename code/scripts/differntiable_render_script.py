from classes.scene import *
from classes.scene_graph import *
from classes.scene_sparse import *
from classes.camera import *
from classes.visual import *
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
    beta_cloud = np.zeros((grid_size,grid_size,grid_size))
    beta_cloud[mid,mid,mid] = beta
    print(beta_cloud)
    beta_air = 0.1

    # Declerations
    grid = Grid(bbox, beta_cloud.shape)
    volume = Volume(grid, beta_cloud, beta_air)
    print(volume.betas)
    #######################
    # Cameras declaration #
    #######################
    N_cams = 2
    focal_length = 70e-3
    sensor_size = np.array((40e-3, 40e-3))
    ps = 15
    pixels = np.array((ps, ps))

    t1 = np.array([0.5, 0.5, 2])
    euler_angles1 = np.array((180, 0, 0))
    camera1 = Camera(t1, euler_angles1, focal_length, sensor_size, pixels)

    t2 = np.array([0.5, 0.9, 2])
    euler_angles2 = np.array((160, 0, 0))
    camera2 = Camera(t2, euler_angles2, focal_length, sensor_size, pixels)

    cameras = [camera1,camera2]

    scene = Scene(volume, cameras, sun_angles)
    scene_graph = SceneGraph(volume, cameras, sun_angles)
    scene_sparse = SceneSparse(volume, cameras, sun_angles)

    Np = int(1e2)
    Ns = 15

    plot_scene = True
    sparse_render = True
    graph_render = True
    basic_render = True


    visual = Visual_wrapper(scene)
    if plot_scene:
        visual.plot_cloud()
        visual.create_grid()
        visual.plot_cameras()
        plt.show()

    fake_cloud = beta_cloud * 1.5
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
        print(I_total[0].T)
        print(I_total[1].T)
        print(f" rendering took: {time() - start}")
        visual.plot_images(I_total, max_val, "sparse")
        plt.show()

    if graph_render:
        print("####### graph renderer ########")
        print("generating paths")

        volume.set_beta_cloud(fake_cloud)
        paths = scene_graph.build_paths_list(Np, Ns)
        volume.set_beta_cloud(beta_cloud)
        print(" start rendering...")
        start = time()
        I_total = scene_graph.render(paths)
        print(I_total[0].T)
        print(I_total[1].T)
        print(f" rendering took: {time() - start}")
        visual.plot_images(I_total, "graph")
        plt.show()

    if basic_render:
        print("####### basic renderer ########")
        print(" start rendering...")
        start = time()
        I_total = scene.render(Np, Ns, 1)
        print(I_total[0].T)
        print(I_total[1].T)
        print(f" rendering took: {time() - start}")
        visual.plot_images(I_total, "basic")
        plt.show()



if __name__ == '__main__':
    main()