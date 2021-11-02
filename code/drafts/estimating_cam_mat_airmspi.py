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
    dir_x, dir_y, dir_z = zentih_azimuth_to_direction(zenith, azimuth, cam_shape)
    camera_array = np.concatenate([x,y,z,dir_x, dir_y, dir_z,images[k][:,:,None]], axis=-1)
    camera_array_list.append(camera_array)
    offset += pixel_num


ts = [np.mean(camera_array_list[k][:,:,:3], axis=(0,1)) for k in range(N_cams)]
airmspi_data["ts"] = ts

mean_point = np.array([1.92995085, 1.34717543, 0.75312834])
Ms = []
for k in range(N_cams):
    resolution = resolutions[k]
    res_list = []
    # N = 1000
    # i_inds = np.random.randint(0,resolution[0],N)
    # j_inds = np.random.randint(0,resolution[1],N)
    # for i,j in zip(i_inds, j_inds):
    row_list_i = []
    row_list_j = []
    i_list = []
    j_list = []
    move = np.array([0,0.5,1])
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            # if np.random.rand() < 0.99:
            #     continue
            x, y, z, dir_x, dir_y, dir_z = camera_array_list[k][i, j, :-1]
            # if k == 1:
            # scale = 7
            # x += move[0] * scale
            # y += move[1] * scale
            # z += move[2] * scale
            # length = np.random.rand()*10 + 100
            length = np.sqrt((x-mean_point[0])**2 + (y-mean_point[1])**2 + (z-mean_point[2])**2)
            # length *= 0.1*np.random.rand() + 0.9
            X = x + dir_x * length
            Y = y + dir_y * length
            Z = z + dir_z * length
            a_i = np.array([[X, Y, Z, 1, -i*X, -i*Y, -i*Z, -i]])
            a_j = np.array([[X, Y, Z, 1]])
            row_list_i.append(a_i)
            row_list_j.append(a_j)
            # i_list.append(i)
            j_list.append(j)
    A_i = np.concatenate(row_list_i, axis=0)
    A_j = np.concatenate(row_list_j, axis=0)
    w,v = np.linalg.eig(A_i.T@A_i)
    min_ind = np.argmin(w)

    m1 = ((np.linalg.inv(A_j.T @ A_j) @ A_j.T) @ np.array(j_list))[None,:]
    m23 = -v[:,min_ind].reshape((2,4))
    M = np.concatenate([m1, m23], axis=0)
    # m /= np.linalg.norm(m)

    abs_err_i = []
    abs_err_j = []
    for i in range(resolution[0]):
        for j in range(resolution[1]):
            # if np.random.rand() < 0.99:
            #     continue
            x, y, z, dir_x, dir_y, dir_z = camera_array_list[k][i, j, :-1]
            length = np.sqrt((x-mean_point[0])**2 + (y-mean_point[1])**2 + (z-mean_point[2])**2)
            X = x + dir_x * length
            Y = y + dir_y * length
            Z = z + dir_z * length
            point_hom = np.array([X,Y,Z,1])[:,None]
            est_j, est_iw, w = M@point_hom
            if w < 0:
                M[1:, :] *= -1
                est_j, est_iw, w = M @ point_hom
            est_i = est_iw / w


            # est_j = X * m[0,0] + Y * m[1,0] + Z * m[2,0] + m[3,0]
            # print((int(est_i), int(est_j)), "  ", (i,j))
            abs_err_i.append(np.abs(est_i - i))
            abs_err_j.append(np.abs(est_j - j))


# plt.hist(res_list)
# plt.show()
    print(np.mean(abs_err_i),  np.mean(abs_err_j))
    Ms.append(M)


    # testing
    grid = Grid(airmspi_data["bbox"], airmspi_data["grid_shape"])
    image = np.zeros((resolution[0], resolution[1]), dtype=bool)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for l in range(grid.shape[2]):
                if cloud_mask[i,j,l]:
                    for _ in range(10):
                        x_noise, y_noise, z_noise = np.random.rand(3)
                        x_noise, y_noise, z_noise = (0,0,0)
                    # x_noise, y_noise, z_noise = [0,0,0]
                        cloud_point = np.array([(i+x_noise)*grid.voxel_size[0], (j+y_noise)*grid.voxel_size[1], (l+z_noise)*grid.voxel_size[2], 1])[:,None]
                        est_j, est_iw, w = Ms[k] @ cloud_point
                        est_i = est_iw / w
                        if est_i < 0 or est_j < 0 or est_i >= resolution[0] or est_j >= resolution[1]:
                            continue
                        image[int(est_i), int(est_j)] =True
    plt.figure()
    plt.title(f"image {k}")#, scale {scale}" )
    plt.imshow(np.concatenate([image,images[k]],axis=1), cmap="gray")
    plt.imshow(image, cmap="gray")
    plt.show()

airmspi_data["Ps"] = Ms
airmspi_data["camera_array_list"] = camera_array_list
a_file = open(join("data","airmspi_data_modified.pkl"), "wb")
pickle.dump(airmspi_data, a_file)
a_file.close()