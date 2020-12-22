from numba import cuda
import numpy as np
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

#### SCENE FUNCTIONS ####
@cuda.jit()
def generate_path(Ns, betas, bbox, bbox_size, voxel_size, sun_direction, N_cams, pixels_shape, ts, Ps, is_in_medium, g,
                  scatter_points, scatter_voxels, camera_pixels, camera_ISs):
    grid_shape = betas.shape
    # path list
    ISs = np.zeros(Ns, dtype=np.float64)
    scatter_tensor = np.zeros((3, Ns), dtype=np.int32)
    camera_pixels = np.zeros((2, N_cams, Ns), dtype=np.int32)
    camera_ISs = np.zeros((N_cams, Ns), dtype=np.float64)

    # helpful list
    voxels =  [np.array([x,x,x], dtype=np.int32) for x in range(0)]
    cam_vec = [np.uint8(x) for x in range(0)]
    seg_vec = [np.uint8(x) for x in range(0)]
    lengths = [np.float32(x) for x in range(0)]

    direction =np.copy(sun_direction)
    # sample entering point
    start_x = bbox_size[0] * np.random.rand() + bbox[0, 0]
    start_y = bbox_size[1] * np.random.rand() + bbox[1, 0]
    start_z = bbox[2,1]

    current_point = np.array([start_x, start_y, start_z])
    current_voxel = get_voxel_of_point(current_point, grid_shape, bbox, bbox_size)
    visible_by_any_camera = False
    for seg in range(Ns):
        p = np.random.rand()
        tau_rand = -math.log(1 - p)
        current_point, current_voxel, in_medium, seg_voxels, seg_lengths, seg_size, beta\
            = voxel_traversal_algorithm_save(current_point, current_voxel, direction, tau_rand, betas, voxel_size)
        if in_medium == False:
            break
        voxels.extend(seg_voxels)
        cam_vec.extend([-1]*seg_size)
        seg_vec.extend([seg]*seg_size)
        lengths.extend(seg_lengths)
        scatter_tensor[0,seg] = current_voxel[0]
        scatter_tensor[1,seg] = current_voxel[1]
        scatter_tensor[2,seg] = current_voxel[2]

        # segments_size.append(seg_size)
        ISs[seg] = 1 / (beta * (1 - p))

        # measuring local estimations to each camera
        for k in range(N_cams):
            P = Ps[k]
            t = ts[k]
            pixel = project_point(current_point, P, pixels_shape)
            camera_pixels[0,k,seg] = pixel[0]
            camera_pixels[1,k,seg] = pixel[1]
            if (pixel == -1).any():
                continue
            else:
                visible_by_any_camera = True
                cam_direction = t - current_point
                distance_to_camera = np.linalg.norm(cam_direction)
                cam_direction /= distance_to_camera
                if is_in_medium[k]:
                    dest = t
                else:
                    dest = get_intersection_with_borders(current_point, cam_direction, bbox)
                local_est, cam_seg_voxels, cam_seg_lengths = \
                    local_estimation_save(current_point, current_voxel, cam_direction, dest, betas, voxel_size)
                cam_seg_size = len(cam_seg_voxels)
                voxels.extend(cam_seg_voxels)
                cam_vec.extend([k] * cam_seg_size)
                seg_vec.extend([seg] * cam_seg_size)
                lengths.extend(cam_seg_lengths)
                cos_theta = np.dot(direction, cam_direction)
                camera_ISs[k,seg] = (1 / (distance_to_camera**2)) * pdf(cos_theta, g)
        direction = sample_direction(direction, g)

    N_seg = seg + in_medium

    if N_seg == 0 or not visible_by_any_camera:
        return None

    # cut tensors to N_seg
    ISs = ISs[:N_seg]
    ISs = np.cumprod(ISs)
    camera_ISs = camera_ISs[:,:N_seg]
    camera_pixels = camera_pixels[:,:,:N_seg]
    scatter_tensor = scatter_tensor[:,:N_seg]

    # calculate ISs matrix
    ISs_mat = camera_ISs * ISs.reshape(1,-1)

    # reshape
    voxels = vertical_concat(voxels)
    lengths = np.array(lengths, dtype=np.float64)
    cam_vec = np.array(cam_vec, dtype=np.int32).reshape(-1,1)
    seg_vec = np.array(seg_vec, dtype=np.int32).reshape(-1,1)
    length_inds = np.hstack((voxels, cam_vec, seg_vec))
    return length_inds, lengths, ISs_mat, scatter_tensor, camera_pixels





#### GRID FUNCTIONS #####
@njit()
def get_voxel_of_point(point, grid_shape, bbox, bbox_size):
    if point[0] == bbox[0, 1]:
        x = grid_shape[0] - 1
    else:
        x = int(((point[0] - bbox[0, 0]) / bbox_size[0]) * grid_shape[0])

    if point[1] == bbox[1, 1]:
        y = grid_shape[1] - 1
    else:
        y = int(((point[1] - bbox[1, 0]) / bbox_size[1]) * grid_shape[1])

    if point[2] == bbox[2, 1]:
        z = grid_shape[2] - 1
    else:
        z = int(((point[2] - bbox[2, 0]) / bbox_size[2]) * grid_shape[2])
    return np.array([x, y, z], dtype=np.int32)


@njit()
def is_in_bbox(point, bbox):
    if (point < bbox[:, 0]).any() or (point > bbox[:, 1]).any():
        return False
    else:
        return True

@njit()
def travel_to_voxels_border(current_point, direction, current_voxel, voxel_size):
    next_point = np.copy(current_point)
    next_voxel = np.copy(current_voxel)
    inc = np.sign(direction).astype(np.int32)
    voxel_fix = (inc - 1) / 2
    ts = np.ones((3,), dtype=np.float64) * 10

    if direction[0] != 0:
        ts[0] = (((current_voxel[0] + 1 + voxel_fix[0]) * voxel_size[0]) - current_point[0]) / direction[0]
    if direction[1] != 0:
        ts[1] = (((current_voxel[1] + 1 + voxel_fix[1]) * voxel_size[1]) - current_point[1]) / direction[1]
    if direction[2] != 0:
        ts[2] = (((current_voxel[2] + 1 + voxel_fix[2]) * voxel_size[2]) - current_point[2]) / direction[2]

    min_ind = np.argmin(ts)
    t = ts[min_ind]
    inc_condition = ts == t
    next_voxel[inc_condition] += inc[inc_condition]
    next_point += t * direction
    return abs(t), next_voxel, next_point, min_ind

@njit()
def get_intersection_with_borders(point, direction, bbox):
    ts = np.ones((3,), dtype=np.float64) * 10
    if direction[0] > 0:
        ts[0] = (bbox[0, 1] - point[0]) / direction[0]
    elif direction[0] < 0:
        ts[0] = (bbox[0, 0] - point[0]) / direction[0]

    if direction[1] > 0:
        ts[1] = (bbox[1, 1] - point[1]) / direction[1]
    elif direction[1] < 0:
        ts[1] = (bbox[1, 0] - point[1]) / direction[1]

    if direction[2] > 0:
        ts[2] = (bbox[2, 1] - point[2]) / direction[2]
    elif direction[2] < 0:
        ts[2] = (bbox[2, 0] - point[2]) / direction[2]

    t = np.min(ts)
    res = point + t*direction
    return res


#### VOLUME FUNCTIONS #####

@njit()
def voxel_traversal_algorithm_save(start_point, current_voxel, direction, tau_rand, betas, voxel_size):
    # easy assignment
    grid_shape = betas.shape
    current_tau = 0.0
    next_voxel = np.copy(current_voxel)
    seg_voxels = []
    seg_lengths = []
    seg_size = 0
    beta = 0
    current_point = np.copy(start_point)
    while True:
        if current_voxel[0] < 0 or current_voxel[0] >= grid_shape[0] or current_voxel[1] < 0 or current_voxel[1] >= \
                grid_shape[1] or current_voxel[2] < 0 or current_voxel[2] >= grid_shape[2]:
            in_medium = False
            break
        beta = betas[current_voxel[0], current_voxel[1], current_voxel[2]]
        seg_voxels.append(np.copy(current_voxel))#.reshape(1, 3))
        length, next_voxel, next_point, _ = travel_to_voxels_border(current_point, direction, next_voxel, voxel_size)
        seg_lengths.append(length)
        current_tau += length * beta
        seg_size += 1
        if current_tau >= tau_rand:
            step_back = (current_tau - tau_rand) / beta
            seg_lengths[-1] -= step_back
            current_point = next_point - step_back * direction
            in_medium = True
            break

        # update current voxel and point
        current_voxel = next_voxel
        current_point = next_point

    # seg_voxels = vertical_concat(seg_voxels)
    # seg_lengths = np.array(seg_lengths)
    return current_point, current_voxel, in_medium, seg_voxels, seg_lengths, seg_size, beta


@cuda.jit(device=True)
def local_estimation_save(current_voxel, current_point, camera_direction, dest, betas, voxel_size):
    grid_shape = betas.shape
    seg_voxels = []
    seg_lengths = []
    distance = np.linalg.norm(current_voxel - dest)
    current_distance = 0.0
    tau = 0.0
    while True:
        if current_voxel[0] < 0 or current_voxel[0] >= grid_shape[0] or current_voxel[1] < 0 or current_voxel[1] >= \
                grid_shape[1] or current_voxel[2] < 0 or current_voxel[2] >= grid_shape[2]:
            break
        seg_voxels.append(np.copy(current_voxel))#.reshape(1, 3))
        beta = betas[current_voxel[0], current_voxel[1], current_voxel[2]]
        length, next_voxel, next_point, _ = travel_to_voxels_border(current_point, camera_direction, current_voxel, voxel_size)
        seg_lengths.append(length)
        tau += length * beta
        current_distance += length
        if current_distance >= distance:
            step_back = current_distance - distance
            tau -= step_back * beta
            seg_lengths[-1] -= step_back
            current_point = next_point - step_back * camera_direction
            break
        current_point = next_point
        current_voxel = next_voxel
    local_est = np.exp(-tau)
    # seg_voxels = np.vstack(seg_voxels)
    # seg_lengths = np.array(seg_lengths)
    return local_est, seg_voxels, seg_lengths



#### CAMERA FUNCTIONS ####
@cuda.jit(device=True)
def project_point(x, P, pixels_shape, res):
    z = x[0]*P[2,0] + x[1]*P[2,1] + x[2]*P[2,2] + P[2,3]
    x = (x[0]*P[0,0] + x[1]*P[0,1] + x[2]*P[0,2] + P[0,3]) / z
    y = (x[0]*P[1,0] + x[1]*P[1,1] + x[2]*P[1,2] + P[1,3]) / z
    if x < 0 or x > pixels_shape[0]:
        x = 255
    if y < 0 or y > pixels_shape[1]:
        y = -1
    res[0] = np.uint8(x)
    res[1] = np.uint8(y)
    return res


#### PHASE FUNCTION FUNCTIONS ####
@cuda.jit(device=True)
def pdf(cos_theta, g):
    theta_pdf = 0.5*(1 - g**2)/(1 + g**2 - 2*g * cos_theta) ** 1.5
    phi_pdf = 1 / (2*np.pi)
    return theta_pdf * phi_pdf

@cuda.jit(device=True)
def sample_direction(old_direction, g, new_direction, rng_states, tid):
    p1 = xoroshiro128p_uniform_float32(rng_states, tid)
    p2 = xoroshiro128p_uniform_float32(rng_states, tid)
    cos_theta = (1 / (2 * g)) * (1 + g**2 - ((1 - g**2)/(1 - g + 2*g*p1))**2)
    phi = p2 * 2 * np.pi
    sin_theta = np.sqrt(1 - cos_theta**2)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    if np.abs(old_direction[2]) > 0.99999:#|z| ~ 1
        z_sign = np.sign(old_direction[2])
        new_direction[0] = sin_theta * cos_phi
        new_direction[1] = z_sign * sin_theta * sin_phi
        new_direction[2] = z_sign * cos_theta
    else:
        denom = np.sqrt(1 - old_direction[2]**2)
        z_cos_phi = old_direction[2] * cos_phi
        new_direction[0] = (sin_theta * (old_direction[0] * z_cos_phi - old_direction[1] * sin_phi) / denom) + old_direction[0] * cos_theta
        new_direction[1] = (sin_theta * (old_direction[1] * z_cos_phi + old_direction[0] * sin_phi) / denom) + old_direction[1] * cos_theta
        new_direction[2] = old_direction[2] * cos_theta - denom * sin_theta * cos_phi
    return new_direction



#### UTILS FUNCTIONS ###
@cuda.jit()
def vertical_concat(arrays, res):
    # res = np.empty((len(arrays),arrays[0].shape[0]), dtype=np.int32)
    for i in range(len(arrays)):
        res[i,:] = arrays[i]
    return res

@cuda.jit()
def horizontal_concat(arrays, res):
    # res = np.empty((arrays[0].shape[0], len(arrays)))
    for i in range(len(arrays)):
        res[i] = arrays[i]
    return res