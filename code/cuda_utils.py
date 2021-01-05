from numba import cuda
import math
import numpy as np
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64

float_eff = np.float32
float_reg = np.float32
float_precis = np.float64
if float_precis == np.float64:
    sample_uniform = xoroshiro128p_uniform_float64
else:
    sample_uniform = xoroshiro128p_uniform_float32

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
def is_voxel_valid(voxel, grid_shape,):
    return not (voxel[0] >= grid_shape[0] or voxel[1] >= grid_shape[1] or voxel[2]>= grid_shape[2])


@cuda.jit(device=True)
def travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel):
    inc_x = sign(direction[0])
    inc_y = sign(direction[1])
    inc_z = sign(direction[2])
    voxel_fix_x = (inc_x - 1) / 2
    voxel_fix_y = (inc_y - 1) / 2
    voxel_fix_z = (inc_z - 1) / 2
    t_x = 2.1
    t_y = 2.2
    t_z = 2.3
    if direction[0] != 0:
        t_x = (((current_voxel[0] + 1 + voxel_fix_x) * voxel_size[0]) - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (((current_voxel[1] + 1 + voxel_fix_y) * voxel_size[1]) - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (((current_voxel[2] + 1 + voxel_fix_z) * voxel_size[2]) - current_point[2]) / direction[2]

    t_min = min_3d(t_x, t_y, t_z)
    assign_3d(next_voxel, current_voxel)
    if t_min == t_x:
        next_voxel[0] += inc_x
    elif t_min == t_y:
        next_voxel[1] += inc_y
    elif t_min == t_z:
        next_voxel[2] += inc_z
    if t_min < 0:
        print("bugg in t_min", t_min)
    current_point[0] = current_point[0] + t_min*direction[0]
    current_point[1] = current_point[1] + t_min*direction[1]
    current_point[2] = current_point[2] + t_min*direction[2]
    return t_min


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
    p1 = xoroshiro128p_uniform_float64(rng_states, tid)
    p2 = xoroshiro128p_uniform_float64(rng_states, tid)
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


    return cos_theta



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
def compare_eps_3d(a, b, eps):
    if abs(a[0] - b[0]) > eps or abs(a[1] - b[1]) > eps or abs(a[2] - b[2]) > eps:
        return False
    else:
        return True

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
def calc_distance(source, dest):
    return math.sqrt((dest[0] - source[0])**2 + (dest[1] - source[1])**2 + (dest[2] - source[2])**2)

@cuda.jit(device=True)
def step_in_direction(current_point, direction, step_size):
    current_point[0] += step_size * direction[0]
    current_point[1] += step_size * direction[1]
    current_point[2] += step_size * direction[2]

@cuda.jit(device=True)
def sign(a):
    if a >= 0:
        return 1
    else:
        return -1