from os.path import join
import matplotlib.pyplot as plt
import pickle
from grid import Grid
from visual import Visual_wrapper
import numpy as np
from utils import zentih_azimuth_to_direction

a_file = open(join("data", "airmspi_data.pkl"), "rb")
airmspi_data = pickle.load(a_file)




xs = airmspi_data["x"]
ys = airmspi_data["y"]
zs = airmspi_data["z"]
zeniths = airmspi_data["zenith"] * (np.pi/180)
azimuths = airmspi_data["azimuth"] * (np.pi/180)
cloud_mask = airmspi_data["cloud_mask"]

images = airmspi_data["images"]
resolutions = airmspi_data["resolution"]
N_cams = len(resolutions)
offset = 0
camera_array_list = []
for k in range(N_cams):
    pixel_num = np.prod(resolutions[k])
    cam_shape = (resolutions[k][0], resolutions[k][1], 1)
    x = xs[offset: offset+pixel_num].reshape(cam_shape,order='F')
    y = ys[offset: offset+pixel_num].reshape(cam_shape,order='F')
    z = zs[offset: offset+pixel_num].reshape(cam_shape,order='F')
    zenith = zeniths[offset: offset+pixel_num]
    azimuth = azimuths[offset: offset+pixel_num]
    # dir_x = -(np.sin(zenith)*np.cos(azimuth)).reshape(cam_shape,order='F')
    # dir_y = -(np.sin(zenith)*np.sin(azimuth)).reshape(cam_shape,order='F')
    # dir_z = -np.cos(zenith).reshape(cam_shape,order='F')
    # dir_x = (np.sin(zenith)*np.sin(azimuth)).reshape(cam_shape,order='F')
    # dir_y = (np.sin(zenith)*np.cos(azimuth)).reshape(cam_shape,order='F')
    # dir_z = -np.cos(zenith).reshape(cam_shape,order='F')
    dir_x, dir_y, dir_z = zentih_azimuth_to_direction(zenith, azimuth, cam_shape)
    camera_array = np.concatenate([x,y,z,dir_x, dir_y, dir_z,images[k][:,:,None]], axis=-1)
    camera_array_list.append(camera_array)
    offset += pixel_num


grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
cloud_points = []
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        for k in range(grid.shape[2]):
            if cloud_mask[i,j,k]:
                cloud_points.append(np.array([i*grid.voxel_size[0], j*grid.voxel_size[1], k*grid.voxel_size[2]])[None,:])
cloud_points = np.concatenate(cloud_points, axis=0)
mean_point = np.mean(cloud_points, axis=0)
print("mean_point", mean_point)

visual = Visual_wrapper(grid)
# visual.create_grid()

k = 0
for i in range(1, resolutions[k][0]-1):
    for j in range(1, resolutions[k][1]-1):
        dist_x = np.linalg.norm(camera_array_list[k][i,j,:3]-camera_array_list[k][i,j-1,:3])
        dist_y = np.linalg.norm(camera_array_list[k][i,j,:3]-camera_array_list[k][i-1,j,:3])
        # print(f"{[i,j]}:", dist_x, dist_y)

plot_pixels_rows = True
plot_pixels = False
plot_images = False
plot_cloud = False

if plot_pixels_rows:
    ts = [np.mean(camera_array_list[k][:, :, :3], axis=(0, 1)) for k in range(N_cams)]
    visual.create_3d_plot()
    visual.ax.set_xlabel("x")
    visual.ax.set_ylabel("y")
    visual.ax.set_zlabel("z")
    axis = 1
    scale = 10
    for k in range(9):
        visual.ax.scatter(*ts[k], s=5)
        resolution = resolutions[k]
        for i in range(0, resolution[axis], 100):# range(0, resolution[1], 10):
            if axis == 0:
                x, y, z, dir_x, dir_y, dir_z = camera_array_list[k][i, :, :-1].T
            elif axis == 1:
                x, y, z, dir_x, dir_y, dir_z = camera_array_list[k][:, i, :-1].T
            else:
                assert "axis must be 1 or 0"
            # x, y, z, dir_x, dir_y, dir_z = camera_array_list[k][:, i, :-1].T
            visual.ax.scatter(x, y, z, s=5)
            if i in [0]:
                dir_scaled = scale * np.array([dir_x, dir_y, dir_z])
                visual.ax.quiver(x, y, z, *dir_scaled, color="orange", linewidth=0.1,arrow_length_ratio=0)
                #     break
                visual.ax.quiver(x, y, z, *-dir_scaled, color="red", linewidth=0.1, arrow_length_ratio=0)
                # visual.ax.quiver(x[0], y[0], z[0], -scale*dir_x[0], -scale*dir_y[0], -scale*dir_z[0], color="red", linewidth=1, arrow_length_ratio=0)
                # visual.ax.quiver(x[-1], y[-1], z[-1], -scale*dir_x[-1], -scale*dir_y[-1], -scale*dir_z[-1], color="red", linewidth=1, arrow_length_ratio=0)
                # visual.ax.quiver(x[140], y[140], z[140], -scale*dir_x[-1], -scale*dir_y[-1], -scale*dir_z[-1], color="blue", linewidth=1, arrow_length_ratio=0)
        # visual.ax.quiver(x, y, z, -scale * dir_x, -scale * dir_y, -scale * dir_z, color="red", linewidth=0.1,
        #                  arrow_length_ratio=0)
    if not plot_cloud:
        plt.show()

if plot_images:
    plt.figure()
    for i, image in enumerate(airmspi_data["images"]):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if plot_cloud:
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                if cloud_mask[i,j,k] and np.random.rand() < 0.01:
                    visual.ax.scatter(i*grid.voxel_size[0], j*grid.voxel_size[1], k*grid.voxel_size[2], color="red")
                    # print(k*grid.voxel_size[2])
    plt.show()

cam_velocities = np.empty(N_cams, dtype=float)
for k in range(N_cams):
    resolution = resolutions[k]
    lengths_list = []
    for j in range(resolution[1]):
        x, y, z, dir_x, dir_y, dir_z = camera_array_list[k][:, j, :-1].T
        lengths = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 +(z[1:] - z[:-1])**2).tolist()
        lengths_list.extend(lengths)
    cam_velocities[k] = np.mean(lengths_list)

