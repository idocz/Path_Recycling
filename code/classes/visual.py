from classes.scene import *
import matplotlib.pyplot as plt
from utils import add_camera_to_ax
from matplotlib import cm


class Visual_wrapper(object):
    def __init__(self, scene):
        self.scene = scene
        self.ax = None

    def plot_cloud(self):
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        beta_cloud = self.scene.volume.beta_cloud
        grid = self.scene.volume.grid
        if beta_cloud.max() != beta_cloud.min():
            beta_norm = (beta_cloud - beta_cloud.min()) / (beta_cloud.max() - beta_cloud.min())
        else:
            beta_norm = 0.5 * np.ones_like(beta_cloud)
        beta_colors = 1 - beta_norm
        # set the colors of each object
        x = np.linspace(grid.bbox[0,0], grid.bbox[0,1], grid.shape[0] + 1)
        y = np.linspace(grid.bbox[1,0], grid.bbox[1,1], grid.shape[1] + 1)
        z = np.linspace(grid.bbox[2,0], grid.bbox[2,1], grid.shape[2] + 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        colors = cm.gray(beta_colors+0.5)
        self.ax.voxels(xx, yy, zz, beta_cloud > self.scene.volume.beta_air, alpha=0.7, facecolors=colors, edgecolors='gray')

    def create_grid(self):
        grid = self.scene.volume.grid
        ax = self.ax
        ticks_x = np.linspace(grid.bbox[0,0], grid.bbox[0,1], grid.shape[0] + 1)
        ticks_y = np.linspace(grid.bbox[1,0], grid.bbox[1,1], grid.shape[1] + 1)
        ticks_z = np.linspace(grid.bbox[2,0], grid.bbox[2,1], grid.shape[2] + 1)
        ax.set_xticks(ticks_x)
        ax.set_yticks(ticks_y)
        ax.set_zticks(ticks_z)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(grid.bbox[0,0], grid.bbox[0,1])
        ax.set_ylim(grid.bbox[1,0], grid.bbox[1,1])
        ax.set_zlim(grid.bbox[2,0], grid.bbox[2,1])
        ax.grid()

    def plot_cameras(self):
        cameras = self.scene.cameras
        for i, cam in enumerate(cameras):
            add_camera_to_ax(self.ax, cam.t, cam.R, i)

    def plot_images(self, I_total, max_val, title):
        N_cams = I_total.shape[0]
        plt.figure()
        N_ax = int(np.ceil(np.sqrt(N_cams)))
        for i in range(N_cams):
            ax = plt.subplot(N_ax, N_ax, i + 1)
            ax.set_title(f"camera {i}")
            ax.imshow(I_total[i].T, cmap="gray", vmin=0, vmax=max_val[i])
        plt.suptitle(title)


