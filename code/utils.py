import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from struct import pack, unpack
import matplotlib.animation as animation
from tensorboard.backend.event_processing import event_accumulator
import cv2
from scipy.io import loadmat, savemat
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
    ax.quiver(t[0], t[1], t[2], *camera_z, color='g', length=dist*5)
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

def cloud_preproccess(beta_cloud, beta_max):
    # beta_cloud = remove_zero_planes(beta_cloud)
    # beta_cloud = resize_to_cubic_shape(beta_cloud)
    # beta_cloud = downsample_3D(beta_cloud, (16, 16, 16))
    beta_cloud[beta_cloud<=0] = 0
    beta_cloud *= beta_max / beta_cloud.max()
    return beta_cloud

def relative_distance(A, B):
    return np.sum(np.abs(A-B)) / np.sum(np.abs(A))

def relative_bias(A, B):
    return (np.sum(np.abs(A))-np.sum(np.abs(B))) / np.sum(np.abs(A))

def cuda_weight(cuda_path):
    sum = 0
    for array in cuda_path:
        sum += array.nbytes
    return sum/1e9


def get_density(filename, scale):
    """
    Generates 3D matrix (ndarray) from a binary of .vol type
    Output
      volume: 3D matrix of float representing the voxels values of the object
      bounding_box: bounding box of the object [xmin, ymin, zmin, xmax, ymax, zmax]
    """

    fid = open(filename, encoding="utf-8")

    # Reading first 48 bytes of volFileName as header , count begins from zero
    header = fid.read(48)

    # Converting header bytes 8-21 to volume size [xsize,ysize,zsize] , type = I : 32 bit integer
    size = unpack(3 * 'I', bytearray(header[8:20]))

    # Converting header bytes 24-47 to bounding box [xmin,ymin,zmin],[xmax,ymax,zmax] type = f : 32 bit float
    # bounding_box = unpack(6*'f', bytearray(header[24:48]))

    # Converting data bytes 49-* to a 3D matrix size of [xsize,ysize,zsize],
    # type = f : 32 bit float
    binary_data = fid.read()
    nCells = size[0] * size[1] * size[2]
    volume = np.array(unpack(nCells * 'f', bytearray(binary_data)))
    volume = volume.reshape(size, order='F')
    fid.close()
    for ax in range(3):
        u_volume, counts = np.unique(volume, axis=ax, return_counts=True)
        if np.all(counts == 2):
            volume = u_volume

    volume *= scale
    return volume

def animate(image_list, interval, repeat_delay=250, output_name=None):
    ims = []
    fig = plt.figure()
    for image in image_list:
        plt.axis("off")
        im = plt.imshow(image, animated=True, cmap="gray")

        ims.append([im])


    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                    repeat_delay=repeat_delay)
    if not output_name is None:
        writer = animation.FFMpegWriter(fps=15)
        # writer = animation.PillowWriter(fps=15)
        ani.save(output_name, writer=writer)
    plt.title("Reconstructed Vs Ground Truth")
    plt.show()

def get_images_from_TB(exp_dir, label):
    ea = event_accumulator.EventAccumulator(exp_dir, size_guidance={"images": 0})
    ea.Reload()
    img_list = []
    wall_time_list = []
    for img_bytes in ea.images.Items(label):
        img = np.frombuffer(img_bytes.encoded_image_string, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img_list.append(img)
        wall_time_list.append(img_bytes.wall_time)
    wall_time_list = [int((wall_time - wall_time_list[0])*60) for wall_time in wall_time_list]
    return img_list, wall_time_list

def get_scalars_from_TB(exp_dir, label):
    ea = event_accumulator.EventAccumulator(exp_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return ea.scalars.Items(label)


def mask_grader(cloud_mask, cloud_mask_real, beta_cloud):
    print(f"accuracy:", np.mean(cloud_mask == cloud_mask_real))
    print(f"fp:", np.mean((cloud_mask == 1) * (cloud_mask_real == 0)))
    fn = (cloud_mask == 0) * (cloud_mask_real == 1)
    print(f"fn:", np.mean(fn))
    fn_exp = (fn * beta_cloud).reshape(-1)
    print(f"fn_exp mean:", np.mean(fn_exp))
    print(f"fn_exp max:", np.max(fn_exp))
    print(f"fn_exp min:", np.min(fn_exp[fn_exp != 0]))
    plt.hist(fn_exp[fn_exp != 0])
    print("missed beta:", np.sum(fn_exp) / np.sum(beta_cloud))
    print("rel_dit:", relative_distance(beta_cloud, beta_cloud * cloud_mask))
    print("number of voxels:",np.sum(cloud_mask==1))
    print("percentage of voxels:",np.mean(cloud_mask==1))
    initialization = np.zeros_like(beta_cloud)
    initialization[cloud_mask] = np.mean(beta_cloud)
    print("initialization distance:", relative_distance(beta_cloud, initialization))

def read_binary_grid3d(filename):
    end = 'little'
    dt = np.dtype(np.float32)
    dt = dt.newbyteorder('<')

    with open(filename, 'rb') as f:
        if f.read(3) != b'VOL':
            raise ValueError(f"File {filename} is not a Mitsuba volume file!")
        version = int.from_bytes(f.read(1), byteorder=end)
        file_type = int.from_bytes(f.read(4), byteorder=end)
        size0 = int.from_bytes(f.read(4), byteorder=end)
        size1 = int.from_bytes(f.read(4), byteorder=end)
        size2 = int.from_bytes(f.read(4), byteorder=end)
        channels = int.from_bytes(f.read(4), byteorder=end)

        bbox_x_min = np.frombuffer(f.read(4), dtype=dt)[0]
        bbox_y_min = np.frombuffer(f.read(4), dtype=dt)[0]
        bbox_z_min = np.frombuffer(f.read(4), dtype=dt)[0]
        bbox_x_max = np.frombuffer(f.read(4), dtype=dt)[0]
        bbox_y_max = np.frombuffer(f.read(4), dtype=dt)[0]
        bbox_z_max = np.frombuffer(f.read(4), dtype=dt)[0]
        bbox = np.array([[bbox_x_min, bbox_x_max],
                         [bbox_y_min,bbox_y_max],
                         [bbox_z_min, bbox_z_max]])
        values = np.frombuffer(f.read(4 * size0 * size1 * size2 * channels), dtype=dt)

        if channels == 1:
            return np.reshape(values, (size0, size1, size2)), bbox
        else:
            return np.reshape(values, (size0, size1, size2, channels)), bbox

def calc_bbox_sample_range(sun_direction, grid):
    if sun_direction[2] == -1:
        return grid.bbox[:2,:]
    sun_temp = sun_direction.copy()
    sun_temp[0] = 0
    sun_temp /= np.linalg.norm(sun_temp)
    sun_y = np.arccos(np.dot(sun_temp, np.array([0, 0, -1])))
    sun_temp = sun_direction.copy()
    sun_temp[1] = 0
    sun_temp /= np.linalg.norm(sun_temp)
    sun_x = np.arccos(np.dot(sun_temp, np.array([0, 0, -1])))

    if sun_direction[0] < 0:
        sign_x = 1
    else:
        sign_x = -1
    delta_x = sign_x * np.tan(sun_x) * grid.bbox_size[2]

    if sun_direction[1] < 0:
        sign_y = 1
    else:
        sign_y = -1
    delta_y = sign_y * np.tan(sun_y) * grid.bbox_size[2]
    sample_range = np.zeros((2,2), dtype=float)
    if delta_x >= 0:
        sample_range[0,0] = grid.bbox[0,0]
        sample_range[0,1] = grid.bbox[0, 1] + delta_x
    else:
        sample_range[0,0] = grid.bbox[0,0] + delta_x
        sample_range[0,1] = grid.bbox[0, 1]

    if delta_y >= 0:
        sample_range[1,0] = grid.bbox[1,0]
        sample_range[1,1] = grid.bbox[1, 1] + delta_y
    else:
        sample_range[1,0] = grid.bbox[1,0] + delta_y
        sample_range[1,1] = grid.bbox[1, 1]

    return sample_range

def zentih_azimuth_to_direction(zenith, azimuth, cam_shape):
    mu = -np.cos(zenith)
    phi = np.pi + azimuth
    u = (np.sqrt(1 - mu*mu) * np.cos(phi)).reshape(cam_shape,order='F')
    v = (np.sqrt(1 - mu*mu) * np.sin(phi)).reshape(cam_shape,order='F')
    w = mu.reshape(cam_shape,order='F')
    return u,v,w



def plank(llambda=660, T=5800):
    h = 6.62607004e-34  # Planck constant
    c = 3.0e8
    k = 1.38064852e-23  # Boltzmann constant
    # https://en.wikipedia.org/wiki/Planck%27s_law
    a = 2.0 * h * (c ** 2)
    b = (h * c) / (llambda * k * T)
    spectral_radiance = a / ((llambda ** 5) * (np.exp(b) - 1.0))
    return spectral_radiance

def norm_image(img):
    return (img - img.min())/(img.max() - img.min())


def imgs2grid(I_total):
    I1 = [I_total[0], I_total[1], I_total[2]]
    I2 = [I_total[3], I_total[4], I_total[5]]
    I3 = [I_total[6], I_total[7], I_total[8]]
    I1 = np.concatenate(I1, axis=1)
    I2 = np.concatenate(I2, axis=1)
    I3 = np.concatenate(I3, axis=1)
    return np.concatenate([I1,I2,I3], axis=0)

def imgs2rows(img1, img2):
    concat1 = np.concatenate([img for img in img1], axis=1)
    concat2 = np.concatenate([img for img in img2], axis=1)
    all = np.concatenate([concat1, concat2], axis=0)
    return all