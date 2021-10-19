from os.path import join
import matplotlib.pyplot as plt
import pickle
from grid import Grid
from visual import Visual_wrapper
import numpy as np
a_file = open(join("data", "airmspi_data.pkl"), "rb")
airmspi_data = pickle.load(a_file)


xs = airmspi_data["x"]
ys = airmspi_data["y"]
zs = airmspi_data["z"]
zeniths = airmspi_data["zenith"] * (np.pi/180)
azimuths = airmspi_data["azimuth"] * (np.pi/180)


plt.figure()
for i, image in enumerate(airmspi_data["images"]):
    ax = plt.subplot(3,3,i+1)
    ax.imshow(image, cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()

grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
visual = Visual_wrapper(grid)
visual.create_3d_plot()
visual.ax.set_xlabel("x")
visual.ax.set_xlim(-40,40)
visual.ax.set_ylabel("y")
visual.ax.set_ylim(-2,8)
visual.ax.set_zlabel("z")
visual.ax.set_zlim(0,20)


pixnum0 = np.prod(airmspi_data["resolution"][0])
# for x, y, z in zip(airmspi_data["x"], airmspi_data["y"], airmspi_data["z"]):
total_pixel_num = len(airmspi_data["x"][:pixnum0])
N = total_pixel_num
N = 100
inds = np.random.randint(0,total_pixel_num, N)
visual.ax.scatter(xs[inds], ys[inds], zs[inds])
for ind in inds:
    zenith = zeniths[ind]
    azimuth = azimuths[ind]
    direction = np.array([np.sin(zenith)*np.sin(azimuth), np.sin(zenith)*np.cos(azimuth), np.cos(zenith)])
    direction *= 200
    # plt.quiver(xs[ind], ys[ind], zs[ind], direction[0], direction[1], direction[2])
    # plt.quiver(xs[ind], ys[ind], zs[ind], -0.1*direction[0], -0.1*direction[1], -0.1*direction[2])
plt.show()