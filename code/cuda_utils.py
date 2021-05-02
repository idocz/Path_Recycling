from numba import cuda, njit
import math
import numpy as np
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64

float_eff = np.float32
float_reg = np.float32
float_precis = np.float64
eff_size = 32 / 8
reg_size = 32 / 8
precis_size = 32 / 8
divide_beta_eps = 1.25e-1
# divide_beta_eps = 0
b  = 0.0
e_ddis = 0.05

if float_eff == np.float64:
    eff_size *= 2
if float_reg == np.float64:
    reg_size *= 2
if float_precis == np.float64:
    sample_uniform = xoroshiro128p_uniform_float64
    precis_size *= 2
else:
    sample_uniform = xoroshiro128p_uniform_float32

#### GRID FUNCTIONS #####
@cuda.jit(device=True)
def get_voxel_of_point(point, grid_shape, bbox, bbox_size, res):
    if point[0] >= bbox[0, 1]:
        res[0] = grid_shape[0] - 1
    else:
        res[0] = int(((point[0] - bbox[0, 0]) / bbox_size[0]) * grid_shape[0])

    if point[1] >= bbox[1, 1]:
        res[1] = grid_shape[1] - 1
    else:
        res[1] = int(((point[1] - bbox[1, 0]) / bbox_size[1]) * grid_shape[1])

    if point[2] >= bbox[2, 1]:
        res[2] = grid_shape[2] - 1
    else:
        res[2] = int(((point[2] - bbox[2, 0]) / bbox_size[2]) * grid_shape[2])


@cuda.jit(device=True)
def is_voxel_valid(voxel, grid_shape):
    return not (voxel[0] >= grid_shape[0] or voxel[1] >= grid_shape[1] or voxel[2]>= grid_shape[2])


@cuda.jit(device=True)
def travel_to_voxels_border_fast(current_point, current_voxel, direction, voxel_size):
    t_x = 20.1
    t_y = 20.2
    t_z = 20.3
    border_x = (current_voxel[0] + (direction[0] > 0)) * voxel_size[0]
    border_y = (current_voxel[1] + (direction[1] > 0)) * voxel_size[1]
    border_z = (current_voxel[2] + (direction[2] > 0)) * voxel_size[2]
    if direction[0] != 0:
        t_x = (border_x - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (border_y - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (border_z - current_point[2]) / direction[2]

    if t_x <= t_y and t_x <= t_z:
        # collision with x
        # current_point[0] = border_x
        # current_point[1] = current_point[1] + t_x * direction[1]
        # current_point[2] = current_point[2] + t_x * direction[2]
        return t_x, border_x, 0
    elif t_y <= t_z:
        # collision with y
        # current_point[0] = current_point[0] + t_y * direction[0]
        # current_point[1] = border_y
        # current_point[2] = current_point[2] + t_y * direction[2]
        return t_y, border_y, 1
    else:
        # collision with z
        # current_point[0] = current_point[0] + t_z * direction[0]
        # current_point[1] = current_point[1] + t_z * direction[1]
        # current_point[2] = border_z
        return t_z, border_z, 2


@cuda.jit(device=True)
def travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel):
    t_x = 20.1
    t_y = 20.2
    t_z = 20.3
    border_x = (current_voxel[0] + (direction[0] > 0)) * voxel_size[0]
    border_y = (current_voxel[1] + (direction[1] > 0)) * voxel_size[1]
    border_z = (current_voxel[2] + (direction[2] > 0)) * voxel_size[2]
    if direction[0] != 0:
        t_x = (border_x - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (border_y - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (border_z - current_point[2]) / direction[2]

    assign_3d(next_voxel, current_voxel)
    if t_x <= t_y and t_x <= t_z:
        # collision with x
        next_voxel[0] += sign(direction[0])
        current_point[0] = border_x
        current_point[1] = current_point[1] + t_x * direction[1]
        current_point[2] = current_point[2] + t_x * direction[2]
        return t_x
    elif t_y <= t_z:
        # collision with y
        next_voxel[1] += sign(direction[1])
        current_point[0] = current_point[0] + t_y * direction[0]
        current_point[1] = border_y
        current_point[2] = current_point[2] + t_y * direction[2]
        return t_y
    else:
        # collision with z
        next_voxel[2] += sign(direction[2])
        current_point[0] = current_point[0] + t_z * direction[0]
        current_point[1] = current_point[1] + t_z * direction[1]
        current_point[2] = border_z
        return t_z


@cuda.jit(device=True)
def travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel):
    t_x = 20.1
    t_y = 20.2
    t_z = 20.3
    border_x = (current_voxel[0] + (direction[0]>0)) * voxel_size[0]
    border_y = (current_voxel[1] + (direction[1]>0)) * voxel_size[1]
    border_z = (current_voxel[2] + (direction[2]>0)) * voxel_size[2]
    if direction[0] != 0:
        t_x = (border_x - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (border_y - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (border_z - current_point[2]) / direction[2]

    assign_3d(next_voxel, current_voxel)
    if t_x <= t_y and t_x <= t_z:
        # collision with x
        next_voxel[0] += sign(direction[0])
        current_point[0] = border_x
        current_point[1] = current_point[1] + t_x * direction[1]
        current_point[2] = current_point[2] + t_x * direction[2]
        return t_x
    elif t_y <= t_z:
        # collision with y
        next_voxel[1] += sign(direction[1])
        current_point[0] = current_point[0] + t_y * direction[0]
        current_point[1] = border_y
        current_point[2] = current_point[2] + t_y * direction[2]
        return t_y
    else:
        # collision with z
        next_voxel[2] += sign(direction[2])
        current_point[0] = current_point[0] + t_z * direction[0]
        current_point[1] = current_point[1] + t_z * direction[1]
        current_point[2] = border_z
        return t_z




@cuda.jit(device=True)
def travel_to_voxels_border_old(current_point, current_voxel, direction, voxel_size, next_voxel):
    inc_x = sign(direction[0])
    inc_y = sign(direction[1])
    inc_z = sign(direction[2])
    t_x = 2.1
    t_y = 2.2
    t_z = 2.3
    if direction[0] != 0:
        t_x = (((current_voxel[0] + 1 + (inc_x - 1) / 2) * voxel_size[0]) - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (((current_voxel[1] + 1 + (inc_y - 1) / 2) * voxel_size[1]) - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (((current_voxel[2] + 1 + (inc_z - 1) / 2) * voxel_size[2]) - current_point[2]) / direction[2]

    assign_3d(next_voxel, current_voxel)
    if t_x <= t_y and t_x <= t_z:
        # collision with x
        next_voxel[0] += inc_x
        current_point[0] = (current_voxel[0] + 1 + (inc_x - 1) / 2) * voxel_size[0]
        current_point[1] = current_point[1] + t_x * direction[1]
        current_point[2] = current_point[2] + t_x * direction[2]
        return t_x
        # t_min = t_x
            # t_min = t_x
        # else:
        #     # collision with z
        #     next_voxel[2] += inc_z
        #     current_point[0] = current_point[0] + t_z * direction[0]
        #     current_point[1] = current_point[1] + t_z * direction[1]
        #     current_point[2] = (current_voxel[2] + 1 + (inc_z - 1) / 2) * voxel_size[2]
        #     # t_min = t_z
    elif t_y <= t_z:
        # collision with y
        next_voxel[1] += inc_y
        current_point[0] = current_point[0] + t_y * direction[0]
        current_point[1] = (current_voxel[1] + 1 + (inc_y - 1) / 2) * voxel_size[1]
        current_point[2] = current_point[2] + t_y * direction[2]
        return t_y
        # t_min = t_y
    else:
        # collision with z
        next_voxel[2] += inc_z
        current_point[0] = current_point[0] + t_z * direction[0]
        current_point[1] = current_point[1] + t_z * direction[1]
        current_point[2] = (current_voxel[2] + 1 + (inc_z - 1) / 2) * voxel_size[2]
        return t_z
        # t_min = t_z
    # current_point[0] = current_point[0] + t_min * direction[0]
    # current_point[1] = current_point[1] + t_min * direction[1]
    # current_point[2] = current_point[2] + t_min * direction[2]
    # return t_min

    # if t_min < 0:
    #     print("bugg in t_min", t_min)


    # return t_min


@cuda.jit(device=True)
def travel_to_voxels_border_old(current_point, current_voxel, direction, voxel_size, next_voxel):
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
        current_point[0] = (current_voxel[0] + 1 + voxel_fix_x) * voxel_size[0]
        current_point[1] = current_point[1] + t_min * direction[1]
        current_point[2] = current_point[2] + t_min * direction[2]
    elif t_min == t_y:
        next_voxel[1] += inc_y
        current_point[1] = (current_voxel[1] + 1 + voxel_fix_y) * voxel_size[1]
        current_point[0] = current_point[0] + t_min * direction[0]
        current_point[2] = current_point[2] + t_min * direction[2]


    elif t_min == t_z:
        next_voxel[2] += inc_z
        current_point[2] = (current_voxel[2] + 1 + voxel_fix_z) * voxel_size[2]
        current_point[0] = current_point[0] + t_min * direction[0]
        current_point[1] = current_point[1] + t_min * direction[1]

    if t_min < 0:
        print("bugg in t_min", t_min)


    return t_min


@cuda.jit(device=True)
def get_intersection_with_borders(point, direction, bbox, res):
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
def get_intersection_with_bbox(point, direction, bbox):
    # print_3d(point)
    # print_3d(direction)
    # print_3d(bbox[1])
    t1 = (bbox[0, 0] - point[0]) / direction[0]
    t2 = (bbox[1, 0] - point[0]) / direction[0]
    t3 = (bbox[0, 1] - point[1]) / direction[1]
    t4 = (bbox[1, 1] - point[1]) / direction[1]
    t5 = (bbox[0, 2] - point[2]) / direction[2]
    t6 = (bbox[1, 2] - point[2]) / direction[2]
    # print(t1,t2,t3,t4,t5,t6)
    tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
    tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))

    # ray (line) is intersecting AABB, but the whole AABB is behind us
    if tmax < 0 or tmin>tmax:
      return False
    tmin += 1e-6
    step_in_direction(point, direction, tmin)
    return True


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
    return res


#### CAMERA FUNCTIONS ####
@cuda.jit(device=True)
def project_point(point, P, pixels_shape, res):
    res[0] = 255
    # res[1] = 255
    z = point[0]*P[2,0] + point[1]*P[2,1] + point[2]*P[2,2] + P[2,3]
    x = (point[0]*P[0,0] + point[1]*P[0,1] + point[2]*P[0,2] + P[0,3]) / z
    y = (point[0]*P[1,0] + point[1]*P[1,1] + point[2]*P[1,2] + P[1,3]) / z
    if x >= 0 and x <= pixels_shape[0] and y >= 0 and y <= pixels_shape[1]:
        res[0] = np.uint8(x)
        res[1] = np.uint8(y)


#### PHASE FUNCTION FUNCTIONS ####
@cuda.jit(device=True)
def rayleigh_pdf(cos_theta):
    return (3 * (1 + cos_theta**2))/(16*math.pi)

@cuda.jit(device=True)
def rayleigh_sample_direction(old_direction, new_direction, rng_states, tid):
    p1 = sample_uniform(rng_states, tid)
    p2 = sample_uniform(rng_states, tid)
    u = -(2*(2* p1 - 1) + (4 * ((2 * p1 - 1) ** 2) + 1) ** (1/2))**(1/3)
    cos_theta = u - (1/ u)
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



@cuda.jit(device=True)
def HG_pdf(cos_theta, g):
    return (1 / (4*np.pi))*(1 - g**2)/(1 + g**2 - 2*g * cos_theta) ** 1.5

@cuda.jit(device=True)
def HG_sample_direction(old_direction, g, new_direction, rng_states, tid):
    p1 = sample_uniform(rng_states, tid)
    p2 = sample_uniform(rng_states, tid)
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
    return pixel[0] != 255


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

@cuda.jit(device=True)
def argmin(x,y,z):
    if x<=y and x<=z:
        return x,0
    elif y<=z:
        return y,1
    else:
        return z,2

@cuda.jit(device=True)
def mat_dot_vec(mat, vec, res):
    res[0] = mat[0,0] * vec[0] + mat[0,1] * vec[1] + mat[0,2] * vec[2]
    res[1] = mat[1,0] * vec[0] + mat[1,1] * vec[1] + mat[1,2] * vec[2]
    res[2] = mat[2,0] * vec[0] + mat[2,1] * vec[1] + mat[2,2] * vec[2]
