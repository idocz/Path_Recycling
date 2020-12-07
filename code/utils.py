import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
from os.path import join
from datetime import datetime
from scipy.ndimage import zoom
from scipy.io import loadmat

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
    time = datetime.now().strftime("%d%m-%H%M-%S")
    train_id = time
    writer = SummaryWriter(log_dir=f"checkpoints/{train_id}")
    os.mkdir(join("checkpoints",train_id,"data"))
    # np.savez(join("checkpoints",train_id,"data","gt"), betas=betas_gt, images=I_gt, max_pixel_val=max_pixel_val)
    for i in range(I_gt.shape[0]):
        writer.add_image(f"ground_truth/{i}", transform_image_for_tensorboard(I_gt[i].T[None,:,:]))
    return writer

def update_tensorboard(writer, I_opt, loss, mean_dist, max_dist, iter):
    for i in range(I_opt.shape[0]):
        writer.add_image(f"simulated_images/{i}", transform_image_for_tensorboard(I_opt[i].T[None,:,:]), global_step=iter)
    writer.add_scalar("loss", loss, global_step=iter)
    writer.add_scalar("mean_dist", mean_dist, global_step=iter)
    writer.add_scalar("max_dist", max_dist, global_step=iter)

def transform_image_for_tensorboard(im):
    im_max = im.max()
    im_min = im.min()
    im = (im - im_min)/(im_max - im_min)
    im *= 255
    return im.astype("uint8")

def angles_between_vectors(v1, v2):
    angle = np.arccos(np.dot(v1, v2))
    return angle

def remove_zero_planes(beta_cloud):

    print(f"original shape:{beta_cloud.shape}")
    Xsize, Ysize, Zsize = beta_cloud.shape
    # remove zero ZY planes
    flag1, flag2 = False, False
    top_cut, bottom_cut = 0, 0
    for i in range(Xsize):
        if beta_cloud[i, :, :].sum() == 0 and not flag1:
            top_cut += 1
        else:
            flag1 = True

        if beta_cloud[Xsize - 1 - i, :, :].sum() == 0 and not flag2:
            bottom_cut += 1
        else:
            flag2 = True

        if flag1 and flag2:
            break
    beta_cloud = beta_cloud[top_cut:Xsize - bottom_cut, :, :]
    Xsize -= bottom_cut + top_cut

    # remove zero ZX planes
    flag1, flag2 = False, False
    top_cut, bottom_cut = 0, 0
    for i in range(Ysize):
        if beta_cloud[:, i, :].sum() == 0 and not flag1:
            top_cut += 1
        else:
            flag1 = True

        if beta_cloud[:, Ysize - 1 - i, :].sum() == 0 and not flag2:
            bottom_cut += 1
        else:
            flag2 = True

        if flag1 and flag2:
            break
    beta_cloud = beta_cloud[:, top_cut:Ysize - bottom_cut, :]
    Ysize -= bottom_cut + top_cut

    # remove zero ZX planes
    flag1, flag2 = False, False
    top_cut, bottom_cut = 0, 0
    for i in range(Zsize):
        if beta_cloud[:, :, i].sum() == 0 and not flag1:
            top_cut += 1
        else:
            flag1 = True

        if beta_cloud[:, :, Zsize - 1 - i].sum() == 0 and not flag2:
            bottom_cut += 1
        else:
            flag2 = True

        if flag1 and flag2:
            break
    beta_cloud = beta_cloud[:, :, top_cut:Zsize - bottom_cut]
    Zsize -= bottom_cut + top_cut

    print(f"new shape:{beta_cloud.shape}  validate:{(Xsize, Ysize, Zsize)}")
    return beta_cloud

def resize_to_cubic_shape(beta_cloud):
    size = max(beta_cloud.shape)
    cubic_data = np.zeros([size] * 3)
    Xsize, Ysize, Zsize = beta_cloud.shape
    cubic_data[(size - Xsize) // 2:(size + Xsize) // 2, (size - Ysize) // 2:(size + Ysize) // 2,
    (size - Zsize) // 2:(size + Zsize) // 2] = beta_cloud

    print(f"new cubic shape:{cubic_data}")
    return cubic_data

def downsample_3D(beta_cloud, new_shape):
    factor = np.array(new_shape) / np.array(beta_cloud.shape)
    return zoom(beta_cloud, factor)

def cloud_loader(file_name, beta_max):
    beta_cloud = loadmat(join("data", file_name))["beta"]
    beta_cloud = remove_zero_planes(beta_cloud)
    # beta_cloud = resize_to_cubic_shape(beta_cloud)
    beta_cloud = downsample_3D(beta_cloud, (16, 16, 16))
    beta_cloud[beta_cloud<=0] = 0
    beta_cloud = (beta_cloud - beta_cloud.min())/(beta_cloud.max()-beta_cloud.min())
    beta_cloud *= 5
    return beta_cloud