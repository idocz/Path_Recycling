import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
from os.path import join
from datetime import datetime

def theta_phi_to_direction(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def euler_to_rotmat(angles):
    # This is an EXTRINSIC rotation in X->Y->Z
    # But     an INTRINSIC rotation in Z->Y->X
    # Ang is in DEGREES

    angles = np.deg2rad(angles)

    rx = angles[0]  # Psi
    ry = angles[1]  # Theta
    rz = angles[2]  # Phi

    RX = np.array(((1, 0, 0),
                   (0, np.cos(rx), -np.sin(rx)),
                   (0, np.sin(rx), np.cos(rx))))

    RY = np.array(((np.cos(ry), 0, np.sin(ry)),
                   (0, 1, 0),
                   (-np.sin(ry), 0, np.cos(ry))))

    RZ = np.array(((np.cos(rz), -np.sin(rz), 0),
                   (np.sin(rz), np.cos(rz), 0),
                   (0, 0, 1)))

    R = RZ.dot(RY.dot(RX))
    return R

def add_camera_to_ax(ax, t, R, label):
    dist = 0.2
    camera_x = R[:, 0]
    camera_y = R[:, 1]
    camera_z = R[:, 2]
    ax.quiver(t[0], t[1], t[2], *camera_x, color='r', length=dist)
    ax.quiver(t[0], t[1], t[2], *camera_y, color='b', length=dist)
    ax.quiver(t[0], t[1], t[2], *camera_z, color='g', length=dist)
    ax.scatter(*t, s=20)
    text_x = t + camera_x * dist
    text_y = t + camera_y * dist
    text_z = t + camera_z * dist
    ax.text(text_x[0], text_x[1], text_x[2], "x")
    ax.text(text_y[0], text_y[1], text_y[2], "y")
    ax.text(text_z[0], text_z[1], text_z[2], "z")
    ax.text(t[0], t[1], t[2], label)


def construct_beta(shape, is_step, beta):
    x = np.linspace(0, 1, shape)
    y = np.linspace(0, 1, shape)
    z = np.linspace(0, 1, shape)
    xx, yy, zz = np.meshgrid(x, y, z)
    R = 0.3
    a = 0.5
    b = 0.5
    c = 0.5
    beta_cloud = np.zeros((shape, shape, shape), dtype=np.float64)
    cond = ((xx - a) ** 2 + (yy - b) ** 2 + (zz - c) ** 2) <= R ** 2
    beta_cloud[cond] = beta
    if is_step:
        beta_cloud[cond * zz>0.6] = beta + 2
    return beta_cloud


def init_tensorboard(I_gt):
    time = datetime.now().strftime("%d%m-%H%M")
    train_id = time
    writer = SummaryWriter(log_dir=f"checkpoints/{train_id}")
    os.mkdir(join("checkpoints",train_id,"data"))
    # np.savez(join("checkpoints",train_id,"data","gt"), betas=betas_gt, images=I_gt, max_pixel_val=max_pixel_val)
    writer.add_image("images/ground_truth", I_gt)
    return writer

def update_tensorboard(writer, loss, iter):
    writer.add_scalar("loss", loss, global_step=iter)



