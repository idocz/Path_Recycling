from classes.volume import *
from classes.path import CudaPaths
from utils import  theta_phi_to_direction
from tqdm import tqdm
import math
from numba import njit
from numba import cuda

class SceneGPU(object):
    def __init__(self, volume: Volume, cameras, sun_angles, g, Ns = 15):
        self.Ns = Ns
        self.volume = volume
        self.sun_angles = sun_angles
        self.sun_direction = theta_phi_to_direction(*sun_angles)
        self.cameras = cameras
        self.g = g
        self.N_cams = len(cameras)
        self.N_pixels = cameras[0].pixels
        self.is_camera_in_medium = np.zeros(self.N_cams, dtype=np.bool)
        for k in range(self.N_cams):
            self.is_camera_in_medium[k] = self.volume.grid.is_in_bbox(self.cameras[k].t)

        N_cams = self.N_cams
        pixels = self.cameras[0].pixels

        # gpu array
        max_voxels = int(5e8)
        max_scatter = int(1e6)
        self.dbetas = cuda.device_array_like(self.volume.betas)
        self.dI_total = cuda.device_array((N_cams, pixels[0], pixels[1]), dtype=np.float32)
        self.dtotal_grad = cuda.device_array(self.volume.betas.shape, dtype=np.float32)
        # self.all_lengths_inds =cuda.device_array((max_voxels,5), dtype=np.uint8)
        # self.all_lengths = cuda.device_array(max_voxels, dtype=np.float32)
        # self.all_ISs_mat = cuda.device_array() #TODO: predefine  all cuda arrays!!!
        # self.all_scatter_tensor = cuda.device_array(
        # self.all_camera_pixels = cuda.device_array(
        # self.scatter_inds = cuda.device_array(
        # self.voxel_inds = cuda.device_array(
        @cuda.jit()
        def render_cuda(all_lengths_inds, all_lengths, all_ISs_mat, all_scatter_tensor, \
                        all_camera_pixels, scatter_inds, voxel_inds, betas, beta_air, w0_cloud, w0_air, I_total):
            tid = cuda.grid(1)
            if tid < voxel_inds.shape[0] - 1:
                # reading thread indices
                scatter_start = scatter_inds[tid]
                scatter_end = scatter_inds[tid + 1]
                voxel_start = voxel_inds[tid]
                voxel_end = voxel_inds[tid + 1]

                # reading thread data
                length_inds = all_lengths_inds[voxel_start:voxel_end]
                lengths = all_lengths[voxel_start:voxel_end]
                ISs_mat = all_ISs_mat[:, scatter_start:scatter_end]
                scatter_tensor = all_scatter_tensor[:, scatter_start:scatter_end]
                camera_pixels = all_camera_pixels[:, :, scatter_start:scatter_end]

                # rendering
                N_seg = scatter_tensor.shape[1]
                path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=np.float32)
                # optical_lengths = optical_lengths[:,:N_seg]

                for row_ind in range(lengths.shape[0]):
                    i, j, k, cam_ind, seg = length_inds[row_ind]
                    L = lengths[row_ind]
                    if cam_ind == 255:
                        for cam_j in range(N_cams):
                            for seg_j in range(N_seg - seg):
                                path_contrib[cam_j, seg + seg_j] += betas[i, j, k] * L
                    else:
                        path_contrib[cam_ind, seg] += betas[i, j, k] * L

                si = scatter_tensor
                prod = 1
                for seg in range(N_seg):
                    prod *= (w0_cloud * (betas[si[0, seg], si[1, seg], si[2, seg]] - beta_air) + w0_air * beta_air)
                    for cam_j in range(N_cams):
                        pc = ISs_mat[cam_j, seg] * math.exp(-path_contrib[cam_j, seg]) * prod
                        pixel = camera_pixels[:, cam_j, seg]
                        cuda.atomic.add(I_total, (cam_j, pixel[0], pixel[1]), pc)

        @cuda.jit()
        def render_differentiable_cuda(all_lengths_inds, all_lengths, all_ISs_mat, all_scatter_tensor, \
                            all_camera_pixels, scatter_inds, voxel_inds, betas, beta_air, w0_cloud, w0_air, I_dif,
                            total_grad,):
            tid = cuda.grid(1)
            if tid < voxel_inds.shape[0] - 1:
                # reading thread indices
                scatter_start = scatter_inds[tid]
                scatter_end = scatter_inds[tid + 1]
                voxel_start = voxel_inds[tid]
                voxel_end = voxel_inds[tid + 1]

                # reading thread data
                length_inds = all_lengths_inds[voxel_start:voxel_end]
                lengths = all_lengths[voxel_start:voxel_end]
                ISs_mat = all_ISs_mat[:, scatter_start:scatter_end]
                scatter_tensor = all_scatter_tensor[:, scatter_start:scatter_end]
                camera_pixels = all_camera_pixels[:, :, scatter_start:scatter_end]

                # rendering
                N_seg = scatter_tensor.shape[1]
                path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=np.float32)
                # optical_lengths = optical_lengths[:,:N_seg]
                # rendering

                for row_ind in range(lengths.shape[0]):
                    i, j, k, cam_ind, seg = length_inds[row_ind]
                    L = lengths[row_ind]
                    if cam_ind == 255:
                        for cam_j in range(N_cams):
                            for seg_j in range(N_seg - seg):
                                path_contrib[cam_j, seg + seg_j] += betas[i, j, k] * L
                    else:
                        path_contrib[cam_ind, seg] += betas[i, j, k] * L

                si = scatter_tensor
                prod = 1
                for seg in range(N_seg):
                    prod *= (w0_cloud * (betas[si[0, seg], si[1, seg], si[2, seg]] - beta_air) + w0_air * beta_air)
                    for cam_j in range(N_cams):
                        path_contrib[cam_j, seg] = ISs_mat[cam_j, seg] * math.exp(-path_contrib[cam_j, seg]) * prod


                for row_ind in range(lengths.shape[0]):
                    i, j, k, cam_ind, seg = length_inds[row_ind]
                    L = lengths[row_ind]
                    if cam_ind == 255:
                        pixel = camera_pixels[:, :, seg:]
                        for pj in range(pixel.shape[2]):
                            for cam_j in range(N_cams):
                                grad_contrib = -L * path_contrib[cam_j, seg + pj]*\
                                               I_dif[cam_j, pixel[0, cam_j, pj], pixel[1, cam_j, pj]]
                                cuda.atomic.add(total_grad, (i, j, k), grad_contrib)

                    else:
                        pixel = camera_pixels[:, cam_ind, seg]
                        grad_contrib = -L * path_contrib[cam_ind, seg] * I_dif[cam_ind, pixel[0], pixel[1]]
                        cuda.atomic.add(total_grad, (i, j, k), grad_contrib)

                for seg in range(N_seg):
                    beta_scatter = w0_cloud * (betas[si[0, seg], si[1, seg], si[2, seg]] - beta_air) + w0_air * beta_air
                    pixel = camera_pixels[:, :, seg:]
                    for pj in range(pixel.shape[2]):
                        for cam_ind in range(N_cams):
                            grad_contrib = (w0_cloud / beta_scatter) * path_contrib[cam_ind, seg + pj] *\
                                           I_dif[cam_ind, pixel[0, cam_ind, pj], pixel[1, cam_ind, pj]]
                            cuda.atomic.add(total_grad, (si[0, seg], si[1, seg], si[2, seg]), grad_contrib)



        self.render_cuda = render_cuda
        self.render_differentiable_cuda = render_differentiable_cuda

    def build_paths_list(self, Np, Ns, workers=1):
        paths = []
        betas = self.volume.betas
        bbox = self.volume.grid.bbox
        bbox_size = self.volume.grid.bbox_size
        voxel_size = self.volume.grid.voxel_size
        sun_direction = self.sun_direction
        N_cams = self.N_cams
        pixels_shape = self.cameras[0].pixels
        ts = np.array([cam.t for cam in self.cameras])
        Ps = np.array([cam.P for cam in self.cameras])
        is_in_medium = self.is_camera_in_medium
        g = self.g
        if workers == 1:
            for _ in tqdm(range(Np)):
                path = generate_path(Ns, betas, bbox, bbox_size, voxel_size, sun_direction, N_cams, pixels_shape, ts, Ps, is_in_medium, g)
                paths.append(path)
        else:
            pass

        print(f"none: {len([path for path in paths if path is None])/Np}")
        cuda_paths = CudaPaths(paths)
        cuda_paths.compress()
        return cuda_paths



    def render(self, cuda_paths, I_gt=None):
        # east declerations
        Np = cuda_paths.Np
        N_cams = len(self.cameras)
        pixels_shape = self.cameras[0].pixels
        betas = self.volume.betas
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air

        threadsperblock = 256
        blockspergrid = (cuda_paths.Np_nonan + (threadsperblock - 1)) // threadsperblock

        self.dbetas.copy_to_device(self.volume.betas)
        self.dI_total.copy_to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=np.float32))
        self.dtotal_grad.copy_to_device(np.zeros_like(betas, dtype=np.float32))

        if not cuda_paths.in_device:
            cuda_paths.to_device()
            print("paths moved to device")
        args = cuda_paths.get_args()
        self.render_cuda[blockspergrid, threadsperblock](*args, self.dbetas, beta_air, w0_cloud, w0_air, self.dI_total)
        I_total = self.dI_total.copy_to_host()
        I_total /= Np
        if I_gt is None:
            return I_total
        I_dif = (I_total - I_gt).astype(np.float32)
        self.dI_total.copy_to_device(I_dif)
        self.render_differentiable_cuda[blockspergrid, threadsperblock](*args, self.dbetas, beta_air, w0_cloud, w0_air,
                                                                        self.dI_total, self.dtotal_grad)
        total_grad = self.dtotal_grad.copy_to_host()

        total_grad /= (Np * N_cams)
        return I_total, total_grad

    def __str__(self):
        text = ""
        text += "Grid:  \n"
        text += str(self.volume.grid) + "  \n"
        text += f"Sun Diretion: theta={self.sun_angles[0]}, phi={self.sun_angles[1]}  \n\n"
        text += f"{self.N_cams} Cameras:" +"  \n"
        for i, cam in enumerate(self.cameras):
            text += f"camera {i}: {str(cam)}  \n"
        text += "  \n"
        text += "Phase_function:  \n"
        text += str(self.g) +"  \n\n"
        return text




##########################################################
####   Numba Function   ##################################
##########################################################


#### SCENE FUNCTIONS ####
@njit()
def generate_path(Ns, betas, bbox, bbox_size, voxel_size, sun_direction, N_cams, pixels_shape, ts, Ps, is_in_medium, g):
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


@njit()
def local_estimation_save(source_point, source_voxel, camera_direction, dest, betas, voxel_size):
    grid_shape = betas.shape
    current_voxel = np.copy(source_voxel)
    current_point = np.copy(source_point)
    seg_voxels = []
    seg_lengths = []
    distance = np.linalg.norm(source_point - dest)
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
@njit()
def project_point(point, P, pixels_shape):
    point = point.reshape(3,1)
    points_hom = np.vstack((point, np.ones((1, 1))))
    points_2d_h = P @ points_hom
    points_2d = points_2d_h[:2] / points_2d_h[2]
    points_2d = points_2d.reshape(-1)
    condition = (points_2d >= 0) * (points_2d <= pixels_shape)
    points_2d[np.logical_not(condition)] = -1
    points_2d = np.floor(points_2d).astype(np.int32)
    return points_2d


#### PHASE FUNCTION FUNCTIONS ####
@njit()
def pdf(cos_theta, g):
    theta_pdf = 0.5*(1 - g**2)/(1 + g**2 - 2*g * cos_theta) ** 1.5
    phi_pdf = 1 / (2*np.pi)
    return theta_pdf * phi_pdf

@njit()
def sample_direction(old_direction, g):
    new_direction = np.empty(3)
    p1, p2 = np.random.rand(2)
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
@njit()
def vertical_concat(arrays):
    res = np.empty((len(arrays),arrays[0].shape[0]), dtype=np.int32)
    for i in range(len(arrays)):
        res[i,:] = arrays[i]
    return res

@njit()
def horizontal_concat(arrays):
    res = np.empty((arrays[0].shape[0], len(arrays)))
    for i in range(len(arrays)):
        res[i] = arrays[i]
    return res