# from classes.scene_rr import *
import matplotlib.pyplot as plt
from utils import add_camera_to_ax
from matplotlib import cm
import numpy as np
class Visual_wrapper(object):
    def __init__(self, grid):
        self.grid = grid
        self.ax = None

    def plot_medium(self, beta_cloud):
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        grid = self.grid
        # set the colors of each object
        x = np.linspace(grid.bbox[0,0], grid.bbox[0,1], grid.shape[0] + 1)
        y = np.linspace(grid.bbox[1,0], grid.bbox[1,1], grid.shape[1] + 1)
        z = np.linspace(grid.bbox[2,0], grid.bbox[2,1], grid.shape[2] + 1)
        yy, xx, zz = np.meshgrid(y, x, z,)

        medium = np.ones(grid.shape) * 0.5
        medium[beta_cloud<0.1] = 0
        colors = cm.gray(medium)
        self.ax.voxels(xx, yy, zz, medium, alpha=0.7, edgecolors='gray', facecolors=colors)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def create_3d_plot(self):
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = plt.axes(projection='3d')
        # ax = self.ax
        # ticks_x = np.linspace(grid.bbox[0, 0], grid.bbox[0, 1], grid.shape[0] // 3 + 1)
        # ticks_y = np.linspace(grid.bbox[1, 0], grid.bbox[1, 1], grid.shape[1] // 3 + 1)
        # ticks_z = np.linspace(grid.bbox[2, 0], grid.bbox[2, 1], grid.shape[2] // 3 + 1)
        # ax.set_xticks(ticks_x)
        # ax.set_yticks(ticks_y)
        # ax.set_zticks(ticks_z)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # ax.set_xlim(grid.bbox[0, 0], grid.bbox[0, 1])
        # ax.set_ylim(grid.bbox[1, 0], grid.bbox[1, 1])
        # ax.set_zlim(grid.bbox[2, 0], grid.bbox[2, 1])
        # ax.grid()

    def create_grid(self):
        grid = self.grid
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = plt.axes(projection='3d')
        ax = self.ax
        ticks_x = np.linspace(grid.bbox[0,0], grid.bbox[0,1], grid.shape[0]//3 + 1)
        ticks_y = np.linspace(grid.bbox[1,0], grid.bbox[1,1], grid.shape[1]//3 + 1)
        ticks_z = np.linspace(grid.bbox[2,0], grid.bbox[2,1], grid.shape[2]//3 + 1)
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

    def plot_cameras(self, cameras):
        for i, cam in enumerate(cameras):
            add_camera_to_ax(self.ax, cam.t, cam.R, i)

    def plot_images(self, I_total, title):
        N_cams = I_total.shape[0]
        fig = plt.figure()

        N_ax = int(np.ceil(np.sqrt(N_cams)))
        for i in range(N_cams):
            ax = plt.subplot(N_ax, N_ax, i + 1)
            ax.set_title(f"camera {i}")
            ax.axis("off")
            ax.imshow(I_total[i], cmap="gray")#, vmin=0, vmax=max_val[i])

        plt.suptitle(title)

    def plot_images_airmspi(self, I_total, resolutions, title, downscale=1):
        N_cams = len(I_total)
        fig = plt.figure()

        N_ax = int(np.ceil(np.sqrt(N_cams)))
        for i in range(N_cams):
            ax = plt.subplot(N_ax, N_ax, i + 1)
            ax.set_title(f"camera {i}")
            ax.axis("off")
            ax.imshow(I_total[i][:resolutions[i][0]//downscale, :resolutions[i][1]//downscale], cmap="gray")  # , vmin=0, vmax=max_val[i])
        plt.tight_layout()
        plt.suptitle(title)

    def plot_images_airmspi_side_by_side(self, I_total, I_gt, resolutions, title, downscale=1):
        N_cams = len(I_total)
        fig = plt.figure()

        N_ax = int(np.ceil(np.sqrt(N_cams)))
        for i in range(N_cams):
            I_show = np.concatenate([I_total[i][:resolutions[i][0]//downscale, :resolutions[i][1]//downscale],
                                     I_gt[i][:resolutions[i][0]//downscale, :resolutions[i][1]//downscale]], axis=1)
            ax = plt.subplot(N_ax, N_ax, i + 1)
            ax.set_title(f"camera {i}")
            ax.axis("off")
            ax.imshow(I_show, cmap="gray")  # , vmin=0, vmax=max_val[i])
        plt.tight_layout()
        plt.suptitle(title)

    def scatter_plot_comparison(self, signal1, signal2, title):
        plt.figure()
        X = signal1.reshape(-1)
        Y = signal2.reshape(-1)
        max_val = np.max([X.max(), Y.max()])
        min_val = np.min([X.min(), Y.min()])
        mask = X != 0
        rel_err = np.sum(np.abs(X[mask] - Y[mask])) / np.sum(np.abs(X[mask]))
        print(f"{title} err: {rel_err}")
        plt.scatter(X, Y)
        plt.plot([min_val, max_val], [min_val,max_val])
        plt.title(title)



