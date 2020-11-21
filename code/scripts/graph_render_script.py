from classes.scene import *
from classes.scene_graph import *
from classes.camera import *
from classes.visual import *
from utils import construct_beta
from time import time
import matplotlib.pyplot as plt
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
sun_angles = np.array([85, 0])


#####################
# Volume parameters #
#####################
# construct betas
grid_size = 20

beta_cloud = construct_beta(grid_size)
print(beta_cloud)
beta_air = 0

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air)
print(volume.betas)
#######################
# Cameras declaration #
#######################
N_cams = 2
focal_length = 20e-3
sensor_size = np.array((40e-3, 40e-3))
pixels = np.array((90, 90))

t1 = np.array([0.5, 0.0, 0.2])
euler_angles1 = np.array((-60, 0, 0))
camera1 = Camera(t1, euler_angles1, focal_length, sensor_size, pixels)

t2 = np.array([0.1, 0.9, 0.9])
euler_angles2 = np.array((-110, 0, 220))
camera2 = Camera(t2, euler_angles2, focal_length, sensor_size, pixels)

cameras = [camera1,camera2]

scene = Scene(volume, cameras, sun_angles)
scene_graph = SceneGraph(volume, cameras, sun_angles)
print(scene.sun_direction)

Np = int(1e4 )
Ns = 15

plot_scene = True
render_scene = True

if plot_scene:
    visual = Visual_wrapper(scene)
    visual.plot_cloud()
    visual.create_grid()
    visual.plot_cameras()
    plt.show()

if render_scene:
    print("####### basic scene renderer ########")

    print(" start rendering...")
    start = time()
    I_total = scene.render(Np, Ns)
    print(f" rendering took: {time() - start}")
    visual.plot_images(I_total, "basic")
    plt.show()
