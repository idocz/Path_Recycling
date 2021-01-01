from numba import cuda
import numpy as np
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
#### SCENE FUNCTIONS ####

cuda.select_device(0)


@cuda.jit()
def render_cuda(all_lengths_inds, all_lengths, all_ISs_mat, all_scatter_tensor, \
                all_camera_pixels, scatter_sizes, voxel_inds, betas, beta_air, w0_cloud, w0_air, I_total):
    tid = cuda.grid(1)
    if tid < voxel_inds.shape[0] - 1:
        # if tid == 0:
        # reading thread indices
        scatter_start = tid * Ns
        scatter_end = (tid + 1) * Ns
        voxel_start = voxel_inds[tid]
        voxel_end = voxel_inds[tid + 1]
        # reading thread data
        length_inds = all_lengths_inds[voxel_start:voxel_end]
        lengths = all_lengths[voxel_start:voxel_end]
        ISs_mat = all_ISs_mat[:, scatter_start:scatter_end]
        si = all_scatter_tensor[:, scatter_start:scatter_end] # scatter_voxels
        camera_pixels = all_camera_pixels[:, :, scatter_start:scatter_end]

        # rendering
        N_seg = scatter_sizes[tid]
        path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=np.float32)
        for row_ind in range(lengths.shape[0]):

            i, j, k, cam_ind, seg = length_inds[row_ind]
            if i == 255:
                break
            L = lengths[row_ind]
            if cam_ind == 255:
                for cam_j in range(N_cams):

                    for seg_j in range(N_seg - seg):
                        path_contrib[cam_j, seg + seg_j] += betas[i, j, k] * L
            else:
                path_contrib[cam_ind, seg] += betas[i, j, k] * L

        prod = 1
        for seg in range(N_seg):
            prod *= (w0_cloud * (betas[si[0, seg], si[1, seg], si[2, seg]] - beta_air) + w0_air * beta_air)
            for cam_j in range(N_cams):
                pc = ISs_mat[cam_j, seg] * math.exp(-path_contrib[cam_j, seg]) * prod
                pixel = camera_pixels[:, cam_j, seg]
                cuda.atomic.add(I_total, (cam_j, pixel[0], pixel[1]), pc)


@cuda.jit()
def calculate_paths_matrix(Np, Ns, betas, bbox, bbox_size, voxel_size, N_cams, ts, is_in_medium, starting_points,
                           scatter_points, scatter_sizes, voxel_inds, camera_pixels, voxels_mat, lengths):
    tid = cuda.grid(1)
    if tid < Np and scatter_sizes[tid] != 0:
        N_seg = scatter_sizes[tid]
        scatter_ind = tid * Ns
        voxel_ind = voxel_inds[tid]
        grid_shape = betas.shape

        # local memory
        current_voxel = cuda.local.array(3, dtype=np.uint8)
        next_voxel = cuda.local.array(3, dtype=np.uint8)
        camera_voxel = cuda.local.array(3, dtype=np.uint8)
        current_point = cuda.local.array(3, dtype=np.float32)
        camera_point = cuda.local.array(3, dtype=np.float32)
        next_point = cuda.local.array(3, dtype=np.float32)
        direction = cuda.local.array(3, dtype=np.float32)
        cam_direction = cuda.local.array(3, dtype=np.float32)
        dest = cuda.local.array(3, dtype=np.float32)
        assign_3d(current_point, starting_points[:, tid])
        get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
        for seg in range(N_seg):
            seg_ind = seg + scatter_ind
            assign_3d(next_point, scatter_points[:, seg_ind])
            ###########################################################
            ############## voxel_fixed traversal_algorithm_save #############
            current_length = 0
            distance = distance_and_direction(current_point, next_point, direction)
            while True:
                length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel)
                current_length += length
                # update row
                voxels_mat[voxel_ind, 0] = current_voxel[0] # voxels
                voxels_mat[voxel_ind, 1] = current_voxel[1] # voxels
                voxels_mat[voxel_ind, 2] = current_voxel[2] # voxels
                voxels_mat[voxel_ind, 3] = 255 # cam (255 for all cams)
                voxels_mat[voxel_ind, 4] = seg # segment
                lengths[voxel_ind] = length
                voxel_ind += 1

                if current_length >= distance - 1e-6:
                    step_back = current_length - distance
                    lengths[voxel_ind-1] -= step_back
                    current_point[0] = current_point[0] - step_back * direction[0]
                    current_point[1] = current_point[1] - step_back * direction[1]
                    current_point[2] = current_point[2] - step_back * direction[2]
                    break

                assign_3d(current_voxel, next_voxel)
            ######################## voxel_fixed_traversal_algorithm_save ###################
            ###########################################################################

            for k in range(N_cams):
                pixel = camera_pixels[:,k,seg_ind]
                if not is_pixel_valid(pixel):
                    continue
                else:
                    assign_3d(camera_voxel, current_voxel)
                    assign_3d(camera_point, current_point)
                    distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                    if is_in_medium[k]:
                        assign_3d(dest, ts[k])
                    else:
                        distance_to_camera = get_intersection_with_borders(camera_point, cam_direction, bbox, dest)

                    ###########################################################################
                    ######################## local estimation save ###################
                    current_length = 0
                    while True:
                        length = travel_to_voxels_border(camera_point, camera_voxel, cam_direction, voxel_size,
                                                         next_voxel)
                        current_length += length
                        # print2_3d(camera_voxel, camera_point)
                        # update row
                        voxels_mat[voxel_ind,0] = camera_voxel[0]  # voxels
                        voxels_mat[voxel_ind,1] = camera_voxel[1]  # voxels
                        voxels_mat[voxel_ind,2] = camera_voxel[2]  # voxels
                        voxels_mat[voxel_ind,3] = k  # cam (255 for all cams)
                        voxels_mat[voxel_ind,4] = seg  # segment
                        lengths[voxel_ind] = length
                        voxel_ind += 1
                        # print(distance_to_camera - current_length)
                        if current_length >= distance_to_camera - 1e-6:
                            # print("end", camera_voxel[0], camera_voxel[1], camera_voxel[2], current_length)
                            step_back = current_length - distance_to_camera
                            lengths[voxel_ind-1] -= step_back
                            camera_point[0] = camera_point[0] - step_back * cam_direction[0]
                            camera_point[1] = camera_point[1] - step_back * cam_direction[1]
                            camera_point[2] = camera_point[2] - step_back * cam_direction[2]
                            break
                        assign_3d(camera_voxel, next_voxel)
                    ######################## local estimation save ###################
                    ###########################################################################

@cuda.jit()
def generate_paths(Np, Ns, betas, bbox, bbox_size, voxel_size, sun_direction, N_cams, pixels_shape, ts, Ps, is_in_medium, g,
                  scatter_voxels, starting_points, scatter_points, camera_pixels, ISs_mat, scatter_sizes, voxel_sizes, rng_states):

    tid = cuda.grid(1)
    if tid < Np:
        start_ind = tid * Ns
        grid_shape = betas.shape

        # local memory
        current_voxel = cuda.local.array(3, dtype=np.uint8)
        next_voxel = cuda.local.array(3, dtype=np.uint8)
        camera_voxel = cuda.local.array(3, dtype=np.uint8)
        current_point = cuda.local.array(3, dtype=np.float32)
        direction = cuda.local.array(3, dtype=np.float32)
        new_direction = cuda.local.array(3, dtype=np.float32)
        cam_direction = cuda.local.array(3, dtype=np.float32)
        dest = cuda.local.array(3, dtype=np.float32)
        pixel = cuda.local.array(2, dtype=np.uint8)
        assign_3d(direction, sun_direction)

        # sample entering point
        p = xoroshiro128p_uniform_float32(rng_states, tid)
        current_point[0] = bbox_size[0] * p + bbox[0, 0]

        p = xoroshiro128p_uniform_float32(rng_states, tid)
        current_point[1] = bbox_size[1] * p + bbox[1, 0]

        current_point[2] = bbox[2,1]
        starting_points[0, tid] = current_point[0]
        starting_points[1, tid] = current_point[1]
        starting_points[2, tid] = current_point[2]
        get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
        total_voxels_size = 0
        IS = 1
        for seg in range(Ns):
            temp_voxels_count = 0
            seg_ind = seg + start_ind
            p = xoroshiro128p_uniform_float32(rng_states, tid)
            tau_rand = -math.log(1 - p)
            ###########################################################
            ############## voxel_traversal_algorithm_save #############
            current_tau = 0.0
            beta = 0
            while True:
                if  current_voxel[0] >= grid_shape[0] or current_voxel[1] == 255 or current_voxel[1] >= grid_shape[1] \
                        or current_voxel[2] >= grid_shape[2]:

                    in_medium = False
                    break
                beta = betas[current_voxel[0], current_voxel[1], current_voxel[2]]
                length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel)
                current_tau += length * beta
                temp_voxels_count += 1

                if current_tau >= tau_rand:
                    step_back = (current_tau - tau_rand) / beta
                    current_point[0] = current_point[0] - step_back * direction[0]
                    current_point[1] = current_point[1] - step_back * direction[1]
                    current_point[2] = current_point[2] - step_back * direction[2]
                    in_medium = True
                    break
                assign_3d(current_voxel, next_voxel)

            ######################## voxel_traversal_algorithm_save ###################
            ###########################################################################
            if in_medium == False:
                break
            total_voxels_size += temp_voxels_count
            # keeping track of scatter points
            scatter_points[0,seg_ind] = current_point[0]
            scatter_points[1,seg_ind] = current_point[1]
            scatter_points[2,seg_ind] = current_point[2]
            scatter_voxels[0,seg_ind] = current_voxel[0]
            scatter_voxels[1,seg_ind] = current_voxel[1]
            scatter_voxels[2,seg_ind] = current_voxel[2]
            # Scatter IS in main trajectory
            IS *= 1 / (beta * (1 - p))

            # calculating ISs_mat and total_voxels_size

            for k in range(N_cams):
                project_point(current_point, Ps[k], pixels_shape, pixel)
                camera_pixels[0,k,seg_ind] = pixel[0]
                camera_pixels[1,k,seg_ind] = pixel[1]
                if not is_pixel_valid(pixel):
                    continue
                else:
                    distance_to_camera = distance_and_direction(current_point, ts[k], cam_direction)
                    if is_in_medium[k]:
                        assign_3d(dest,ts[k])
                    else:
                        get_intersection_with_borders(current_point, cam_direction, bbox, dest)

                    get_voxel_of_point(dest, grid_shape, bbox, bbox_size, camera_voxel)
                    total_voxels_size += estimate_voxels_size(current_voxel, camera_voxel)
                    # print(a)
                    cos_theta = dot_3d(direction, cam_direction)
                    ISs_mat[k,seg_ind] = (1 / (distance_to_camera**2)) * pdf(cos_theta, g) * IS

            sample_direction(direction, g, new_direction, rng_states, tid)
            assign_3d(direction, new_direction)


        N_seg = seg  + int(in_medium)
        scatter_sizes[tid] = N_seg
        voxel_sizes[tid] = total_voxels_size




#### GRID FUNCTIONS #####
@cuda.jit(device=True)
def get_voxel_of_point(point, grid_shape, bbox, bbox_size, res):
    if point[0] == bbox[0, 1]:
        res[0] = grid_shape[0] - 1
    else:
        res[0] = int(((point[0] - bbox[0, 0]) / bbox_size[0]) * grid_shape[0])

    if point[1] == bbox[1, 1]:
        res[1] = grid_shape[1] - 1
    else:
        res[1] = int(((point[1] - bbox[1, 0]) / bbox_size[1]) * grid_shape[1])

    if point[2] == bbox[2, 1]:
        res[2] = grid_shape[2] - 1
    else:
        res[2] = int(((point[2] - bbox[2, 0]) / bbox_size[2]) * grid_shape[2])



@cuda.jit(device=True)
def travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel):
    # print(current_point[0],current_point[1], current_point[2])
    # print(current_point[0],current_point[1], current_point[2])
    inc_x = sign(direction[0])
    inc_y = sign(direction[1])
    inc_z = sign(direction[2])
    # print(inc_x, inc_y, inc_z)
    voxel_fix_x = (inc_x - 1) / 2
    voxel_fix_y = (inc_y - 1) / 2
    voxel_fix_z = (inc_z - 1) / 2
    t_x = 2
    t_y = 2
    t_z = 2
    if direction[0] != 0:
        t_x = (((current_voxel[0] + 1 + voxel_fix_x) * voxel_size[0]) - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (((current_voxel[1] + 1 + voxel_fix_y) * voxel_size[1]) - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (((current_voxel[2] + 1 + voxel_fix_z) * voxel_size[2]) - current_point[2]) / direction[2]
    t_min = min_3d(t_x, t_y, t_z)
    # print(t_x, t_y, t_z, t_min)
    assign_3d(next_voxel, current_voxel)
    if t_min == t_x:
        next_voxel[0] += inc_x
    elif t_min == t_y:
        next_voxel[1] += inc_y
    elif t_min == t_z:
        next_voxel[2] += inc_z
    current_point[0] = current_point[0] + t_min*direction[0]
    current_point[1] = current_point[1] + t_min*direction[1]
    current_point[2] = current_point[2] + t_min*direction[2]
    # if t_min < 0:
    #     print("bug:",t_min)
    return abs(t_min)


@cuda.jit(device=True)
def get_intersection_with_borders(point, direction, bbox, res):
    t_x = 1
    t_y = 1
    t_z = 1
    if direction[0] > 0:
        tx = (bbox[0, 1] - point[0]) / direction[0]
    elif direction[0] < 0:
        tx = (bbox[0, 0] - point[0]) / direction[0]

    if direction[1] > 0:
        ty = (bbox[1, 1] - point[1]) / direction[1]
    elif direction[1] < 0:
        ty = (bbox[1, 0] - point[1]) / direction[1]

    if direction[2] > 0:
        tz = (bbox[2, 1] - point[2]) / direction[2]
    elif direction[2] < 0:
        tz = (bbox[2, 0] - point[2]) / direction[2]

    t_min = min_3d(tx, ty, tz)
    res[0] = point[0] + t_min*direction[0]
    res[1] = point[1] + t_min*direction[1]
    res[2] = point[2] + t_min*direction[2]
    if t_min < 0:
        print("bug")
    return t_min

@cuda.jit(device=True)
def estimate_voxels_size(voxel_a, voxel_b):
    res = 0
    if voxel_a[0] >= voxel_b[0]:
        res += voxel_a[0] - voxel_b[0]
    else:
        res += voxel_b[0] - voxel_a[0]
    if voxel_a[1] >= voxel_b[1]:
        res += voxel_a[1] - voxel_b[1]
    else:
        res += voxel_b[1] - voxel_a[1]

    if voxel_a[2] >= voxel_b[2]:
        res += voxel_a[2] - voxel_b[2]
    else:
        res += voxel_b[2] - voxel_a[2]

    return res + 1


#### CAMERA FUNCTIONS ####
@cuda.jit(device=True)
def project_point(point, P, pixels_shape, res):
    z = point[0]*P[2,0] + point[1]*P[2,1] + point[2]*P[2,2] + P[2,3]
    x = (point[0]*P[0,0] + point[1]*P[0,1] + point[2]*P[0,2] + P[0,3]) / z
    y = (point[0]*P[1,0] + point[1]*P[1,1] + point[2]*P[1,2] + P[1,3]) / z
    if x < 0 or x > pixels_shape[0]:
        x = 255
    if y < 0 or y > pixels_shape[1]:
        y = 255
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
    phi = p2 * 2 * math.pi
    sin_theta = math.sqrt(1 - cos_theta**2)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    if abs(old_direction[2]) > 0.99999:#|z| ~ 1
        z_sign = sign(old_direction[2])
        new_direction[0] = sin_theta * cos_phi
        new_direction[1] = z_sign * sin_theta * sin_phi
        new_direction[2] = z_sign * cos_theta
    else:
        denom = math.sqrt(1 - old_direction[2]**2)
        z_cos_phi = old_direction[2] * cos_phi
        new_direction[0] = (sin_theta * (old_direction[0] * z_cos_phi - old_direction[1] * sin_phi) / denom) + old_direction[0] * cos_theta
        new_direction[1] = (sin_theta * (old_direction[1] * z_cos_phi + old_direction[0] * sin_phi) / denom) + old_direction[1] * cos_theta
        new_direction[2] = old_direction[2] * cos_theta - denom * sin_theta * cos_phi

    if abs(new_direction[0]) < 1e-6:
        new_direction[0] = 0
    if abs(new_direction[1]) < 1e-6:
        new_direction[1] = 0
    if abs(new_direction[2]) < 1e-6:
        new_direction[2] = 0
    return new_direction



#### UTILS FUNCTIONS ###

@cuda.jit(device=True)
def assign_3d(a, b):
    a[0] = b[0]
    a[1] = b[1]
    a[2] = b[2]

@cuda.jit(device=True)
def compare_3d(a, b):
    if a[0] == b[0] and a[1] == b[1] and a[2] == b[2]:
        return True
    else:
        return False

@cuda.jit(device=True)
def min_3d(x, y, z):
    if x <= y:
        if x <= z:
            return x
        else:
            return z
    else:
        if y <= z:
            return y
        else:
            return z



@cuda.jit(device=True)
def dot_3d(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cuda.jit(device=True)
def print_3d(a):
    return print(a[0],a[1],a[2])
@cuda.jit(device=True)
def print2_3d(a,b):
    return print(a[0],a[1],a[2], b[0],b[1],b[2])

@cuda.jit(device=True)
def is_pixel_valid(pixel):
    if pixel[0] != 255 and pixel[1] != 255:
        return True
    else:
        return False

@cuda.jit(device=True)
def distance_and_direction(source, dest, direction):
    direction[0] = dest[0] - source[0]
    direction[1] = dest[1] - source[1]
    direction[2] = dest[2] - source[2]
    distance = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
    direction[0] = direction[0] / distance
    direction[1] = direction[1] / distance
    direction[2] = direction[2] / distance
    return distance

@cuda.jit(device=True)
def sign(a):
    if a >= 0:
        return 1
    else:
        return -1



import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
import pickle
from utils import *
from classes.path import *
import math
from time import time
from classes.camera import *
checkpoint_id = "2212-1250-03"
beta_gt = loadmat(join("data", "rico.mat"))["beta"]
cp = pickle.load(open(join("checkpoints",checkpoint_id,"data",f"checkpoint_loader"), "rb" ))
print("Loading the following Scence:")
print(cp)
scene = cp.recreate_scene()
scene.volume.set_beta_cloud(beta_gt)

Ns = 15
bbox = scene.volume.grid.bbox
N_cams = 5

focal_length = 60e-3
sensor_size = np.array((40e-3, 40e-3))
ps = 128
pixels = np.array((ps, ps))

N_cams = 5
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
R = 1.5 * 1.04
for cam_ind in range(N_cams):
    phi = 0
    theta = (-(N_cams // 2) + cam_ind) * 40
    theta_rad = theta * (np.pi / 180)
    t = R * theta_phi_to_direction(theta_rad, phi) + volume_center
    euler_angles = np.array((180, theta, 0))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)

scene.cameras = cameras
beta_air = scene.volume.beta_air
w0_cloud = scene.volume.w0_cloud
w0_air = scene.volume.w0_air
g = scene.g
pixels_shape = scene.cameras[0].pixels
ts = np.vstack([cam.t.reshape(1,-1) for cam in scene.cameras])
Ps = np.concatenate([cam.P.reshape(1, 3, 4) for cam in scene.cameras], axis=0)
sun_direction = scene.sun_direction
sun_direction[np.abs(sun_direction)<1e-6] = 0
print(sun_direction)
Np = int(7e6)
# inputs
dbetas = cuda.to_device(scene.volume.betas)
dbbox = cuda.to_device(scene.volume.grid.bbox)
dbbox_size = cuda.to_device(scene.volume.grid.bbox_size)
dvoxel_size = cuda.to_device(scene.volume.grid.voxel_size)
dsun_direction = cuda.to_device(sun_direction)
dpixels_shape = cuda.to_device(scene.cameras[0].pixels)
dts = cuda.to_device(ts)
dPs = cuda.to_device(Ps)
dis_in_medium = cuda.to_device(scene.is_camera_in_medium)

for i in range(1):
    start = time()
    # outputs
    dstarting_points = cuda.to_device(np.zeros((3, Np), dtype=np.float32))
    dscatter_points = cuda.to_device(np.zeros((3,Ns*Np), dtype=np.float32))
    dscatter_voxels = cuda.to_device(np.zeros((3,Ns*Np), dtype=np.uint8))
    dcamera_pixels = cuda.to_device(np.zeros((2,N_cams,Ns*Np), dtype=np.uint8))
    dISs_mat = cuda.to_device(np.zeros((N_cams,Ns*Np), dtype=np.float32))
    dscatter_sizes = cuda.to_device(np.zeros(Np, dtype=np.uint8))
    dvoxel_sizes = cuda.to_device(np.zeros(Np, dtype=np.uint32))

    # cuda parameters
    threadsperblock = 256
    blockspergrid = (Np + (threadsperblock - 1)) // threadsperblock
    seed = np.random.randint(1, int(1e10))
    # seed = 12
    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=seed)

    start = time()
    # generating paths
    generate_paths[blockspergrid, threadsperblock](Np, Ns, dbetas, dbbox, dbbox_size, dvoxel_size, dsun_direction, N_cams,
                                                  dpixels_shape, dts, dPs, dis_in_medium, g, dscatter_voxels, dstarting_points,
                                                 dscatter_points, dcamera_pixels, dISs_mat, dscatter_sizes, dvoxel_sizes,
                                                   rng_states)



    cuda.synchronize()
    end = time()
    print(end-start)

    scatter_points = dscatter_points.copy_to_host()
    camera_pixels = dcamera_pixels.copy_to_host()
    ISs_mat = dISs_mat.copy_to_host()
    scatter_sizes = dscatter_sizes.copy_to_host()
    voxel_sizes = dvoxel_sizes.copy_to_host()
    voxel_inds = np.concatenate([np.array([0]), voxel_sizes])
    voxel_inds = np.cumsum(voxel_inds)
    print(voxel_inds[-5:])
    print(voxel_inds[:5])
    total_num_of_voxels = np.sum(voxel_sizes)


    dvoxel_inds = cuda.to_device(voxel_inds)
    dvoxels_mat = cuda.to_device(np.ones((total_num_of_voxels,5), dtype=np.uint8)*255)
    dlengths = cuda.to_device(np.zeros(total_num_of_voxels, dtype=np.float32))
    print("calculate_paths_matrix")
    calculate_paths_matrix[blockspergrid, threadsperblock](Np, Ns, dbetas, dbbox, dbbox_size, dvoxel_size, N_cams, ts,
                                                          dis_in_medium, dstarting_points, dscatter_points,dscatter_sizes, dvoxel_inds,
                                                           dcamera_pixels, dvoxels_mat, dlengths)

    cuda.synchronize()
    voxels_mat = dvoxels_mat.copy_to_host()
    lengths = dlengths.copy_to_host()
    end = time()
    print(f"{i} took: {end-start} ")


    dI_total = cuda.to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=np.float32))

    print("render_cuda")
    render_cuda[blockspergrid, threadsperblock](dvoxels_mat, dlengths, dISs_mat, dscatter_voxels,\
                   dcamera_pixels, dscatter_sizes, dvoxel_inds, dbetas, beta_air, w0_cloud, w0_air, dI_total)



cuda.synchronize()
I_total = dI_total.copy_to_host()
I_total /= Np
plt.imshow(I_total[1], cmap="gray")
plt.show()

paths = scene.build_paths_list(Np, Ns)
I_total_cpu = scene.render(paths)
I_gt = np.load(join("checkpoints", "2212-1250-03", "data", "gt.npz"))["images"]
print(f"max:{I_total.max()}, {I_gt.max()}")
abs_im = np.abs(I_total - I_gt)


print("I_gt vs I_total")

# plt.imshow(abs_im[0], cmap="gray")
# plt.show()

res = abs_im.mean() / np.abs(I_gt).mean()
print(res.mean())
print(res.max())
print(res.min())


print("I_gt vs I_total_cpu")
abs_im = np.abs(I_total_cpu - I_gt)

# plt.imshow(abs_im[0], cmap="gray")
# plt.show()


res = abs_im.mean() / np.abs(I_gt).mean()
print(res.mean())
print(res.max())
print(res.min())


paths = scene.build_paths_list(Np, Ns)
I_total_cpu2 = scene.render(paths)


print("I_total_cpu vs I_total_cpu2")
abs_im = np.abs(I_total_cpu - I_total_cpu2)



plt.imshow(abs_im[0], cmap="gray")
plt.show()


res = abs_im.mean() / np.abs(I_total_cpu2).mean()
print(res.mean())
print(res.max())
print(res.min())