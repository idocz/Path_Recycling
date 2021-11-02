from classes.volume import *
from utils import  theta_phi_to_direction
from numba.cuda.random import create_xoroshiro128p_states
from cuda_utils import *

class SceneAirMSPI(object):
    def __init__(self, volume: Volume, camera_array_list, sun_direction, sun_intensity, g_cloud, rr_depth, rr_stop_prob):
        self.rr_depth = rr_depth
        self.rr_stop_prob = rr_stop_prob
        rr_factor = 1.0 / (1.0 - self.rr_stop_prob)
        self.rr_factor = rr_factor
        self.volume = volume
        self.sun_direction = sun_direction
        self.sun_direction /= np.linalg.norm(self.sun_direction)
        self.sun_direction *= -1
        self.sun_intensity = sun_intensity
        self.camera_array_list = camera_array_list
        self.g_cloud = g_cloud
        self.N_cams = len(camera_array_list)

        N_cams = self.N_cams
        max_pix_i = np.max([camera_array_list[cam_ind].shape[0] for cam_ind in range(N_cams)])
        max_pix_j = np.max([camera_array_list[cam_ind].shape[1] for cam_ind in range(N_cams)])
        self.pixels_shape = np.array([max_pix_i, max_pix_j])
        self.pixels_shapes = [camera_array_list[cam_ind].shape[:2] for cam_ind in range(N_cams)]

        # gpu array
        self.dbeta_cloud = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dtotal_grad = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dcloud_mask = cuda.device_array(self.volume.beta_cloud.shape, dtype=bool)
        self.dcloud_mask.copy_to_device(self.volume.cloud_mask)
        self.dbbox = cuda.to_device(self.volume.grid.bbox)
        self.dbbox_size = cuda.to_device(self.volume.grid.bbox_size)
        self.dvoxel_size = cuda.to_device(self.volume.grid.voxel_size)
        self.dsun_direction = cuda.to_device(self.sun_direction)
        self.dcamera_array_list = [cuda.to_device(camera_array_list[cam_ind]) for cam_ind in range(N_cams)]
        self.spp_map = None
        self.dpath_contrib = None
        self.dgrad_contrib = None

        @cuda.jit()
        def calc_scatter_sizes(job_list, camera_array, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size,
                               voxel_size, scatter_sizes, rng_states):

            tid = cuda.grid(1)
            # tid = camera_array.shape[0] * camera_array.shape[1] * s + camera_array.shape[1] * i + j
            if tid < job_list.shape[0]:
                i, j = job_list[tid]
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
                assign_3d(current_point, camera_array[i, j, :3])
                assign_3d(direction, camera_array[i, j, 3:6])
                intersected = get_intersection_with_bbox(current_point, direction, bbox)
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                scatter_sizes[tid] = 0

                beta_c = 1.0
                seg = 0
                while intersected:
                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break
                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)

                        current_tau += length * beta
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
                    seg += 1

                    cloud_prob_scat = w0_cloud * beta_c / (w0_cloud * beta_c + w0_air * beta_air)

                    if sample_uniform(rng_states, tid) <= cloud_prob_scat:
                        HG_sample_direction(direction, g_cloud, new_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, new_direction, rng_states, tid)

                    assign_3d(direction, new_direction)

                # voxels and scatter sizes for this path (this is not in a loop)
                scatter_sizes[tid] = seg


        @cuda.jit()
        def render_cuda(job_list, camera_array, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, sun_direction
                               , rng_states, scatter_inds, path_contrib, grad_contrib, I_total):
            tid = cuda.grid(1)
            # tid = camera_array.shape[0] * camera_array.shape[1] * s + camera_array.shape[1] * i + j
            if tid < job_list.shape[0]-1:
                i, j = job_list[tid]
                grid_shape = beta_cloud.shape
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_ind
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                le_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                le_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
                assign_3d(current_point, camera_array[i,j, :3])
                assign_3d(direction, camera_array[i,j, 3:6])
                intersected = get_intersection_with_bbox(current_point, direction, bbox)
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                attenuation = 1.0
                beta_c = 1.0
                beta = 1.0
                pc_sum = 0.0
                seg = 0
                while intersected:
                    seg_ind = seg + scatter_ind
                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break
                        else:
                            attenuation *= rr_factor
                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)

                        current_tau += length * beta
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


                    cloud_prob_scat = w0_cloud * beta_c / (w0_cloud * beta_c + w0_air * beta_air)

                    if sample_uniform(rng_states, tid) <= cloud_prob_scat:
                        HG_sample_direction(direction, g_cloud, new_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, new_direction, rng_states, tid)


                    cloud_prob = 1 - (beta_air / beta)
                    cloud_prob_scat = (w0_cloud * beta_c) / (w0_cloud * beta_c + w0_air * beta_air)
                    attenuation *= cloud_prob * w0_cloud + (1 - cloud_prob) * w0_air

                    # Local estimation to the sun
                    assign_3d(le_voxel, current_voxel)
                    assign_3d(le_point, current_point)
                    cos_theta = dot_3d(direction, sun_direction)
                    le_pdf = cloud_prob_scat * HG_pdf(cos_theta, g_cloud) \
                             + (1 - cloud_prob_scat) * rayleigh_pdf(cos_theta)

                    pc = le_pdf * attenuation
                    get_intersection_with_borders(le_point, sun_direction, bbox, next_point)
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    ###########################################################################
                    ######################## local estimation save ############################
                    tau = 0
                    for pi in range(path_size):
                        ##### border traversal #####
                        border_x = (le_voxel[0] + (sun_direction[0] > 0)) * voxel_size[0]
                        border_y = (le_voxel[1] + (sun_direction[1] > 0)) * voxel_size[1]
                        border_z = (le_voxel[2] + (sun_direction[2] > 0)) * voxel_size[2]
                        t_x = (border_x - le_point[0]) / (sun_direction)[0]
                        t_y = (border_y - le_point[1]) / (sun_direction)[1]
                        t_z = (border_z - le_point[2]) / (sun_direction)[2]
                        length, border_ind = argmin(t_x, t_y, t_z)
                        ###############################

                        beta_le = beta_cloud[le_voxel[0], le_voxel[1], le_voxel[2]] + beta_air
                        tau += beta_le * length
                        # assign_3d(camera_voxel, next_voxel)
                        step_in_direction(le_point, sun_direction, length)
                        # camera_point[border_ind] = border
                        le_voxel[border_ind] += sign(sun_direction[border_ind])
                    # Last Step
                    length = calc_distance(le_point, next_point)
                    beta_le = beta_cloud[le_voxel[0], le_voxel[1], le_voxel[2]] + beta_air
                    tau += beta_le * length
                    ######################## local estimation save ############################
                    ###########################################################################
                    pc *= math.exp(-tau)
                    path_contrib[seg_ind] = pc
                    cuda.atomic.add(I_total, (i, j), pc)
                    pc_sum += pc
                    seg += 1

                    assign_3d(direction, new_direction)

                grad_contrib[tid] = pc_sum
                if seg!= N_seg:
                    print("reg", seg, N_seg)


        @cuda.jit()
        def render_differentiable_cuda(job_list, camera_array, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size,
                        voxel_size, sun_direction
                        , rng_states, scatter_inds, path_contrib, grad_contrib, cloud_mask, I_dif, total_grad):
            tid = cuda.grid(1)
            # tid = camera_array.shape[0] * camera_array.shape[1] * s + camera_array.shape[1] * i + j
            if tid < job_list.shape[0] - 1:
                i, j = job_list[tid]
                grid_shape = beta_cloud.shape
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                dif_val = I_dif[i,j]
                pc_sum = grad_contrib[tid]
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                le_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                le_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
                assign_3d(current_point, camera_array[i, j, :3])
                assign_3d(direction, camera_array[i, j, 3:6])
                intersected = get_intersection_with_bbox(current_point, direction, bbox)
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                beta_c = 1.0
                beta = 1.0
                seg = 0
                cos_theta_scatter = 0.0
                while intersected:
                    seg_ind = seg + scatter_ind
                    pc = path_contrib[seg_ind]
                    # Russian Roulette

                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break
                    if seg > 0:
                        if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                            omega_fp_c = w0_cloud * HG_pdf(cos_theta_scatter, g_cloud)
                            seg_contrib = omega_fp_c / \
                                          (beta_c * omega_fp_c + beta_air * w0_air * rayleigh_pdf(cos_theta_scatter))  # scatter fix
                            cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]),
                                            seg_contrib * pc_sum * dif_val)

                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)
                        if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                            cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]),
                                            -length * pc_sum * dif_val)
                        current_tau += length * beta
                        if current_tau >= tau_rand:
                            step_back = (current_tau - tau_rand) / beta
                            current_point[0] = current_point[0] - step_back * direction[0]
                            current_point[1] = current_point[1] - step_back * direction[1]
                            current_point[2] = current_point[2] - step_back * direction[2]
                            in_medium = True
                            if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                                cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]),
                                                step_back * pc_sum * dif_val)
                            break
                        assign_3d(current_voxel, next_voxel)

                    ######################## voxel_traversal_algorithm_save ###################
                    ###########################################################################
                    if in_medium == False:
                        break
                    cloud_prob_scat = (w0_cloud * beta_c) / (w0_cloud * beta_c + w0_air * beta_air)

                    if sample_uniform(rng_states, tid) <= cloud_prob_scat:
                        HG_sample_direction(direction, g_cloud, new_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, new_direction, rng_states, tid)


                    # Local estimation to the sun
                    assign_3d(le_voxel, current_voxel)
                    assign_3d(le_point, current_point)
                    cos_theta = dot_3d(direction, sun_direction)
                    get_intersection_with_borders(le_point, sun_direction, bbox, next_point)
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                        # seg_contrib = le_contrib + (beta_c + (rayleigh_pdf(cos_theta) / HG_pdf(cos_theta, g_cloud)) * beta_air) ** (-1)
                        omega_fp_c = w0_cloud * HG_pdf(cos_theta, g_cloud)
                        seg_contrib = omega_fp_c / \
                                      (beta_c * omega_fp_c + beta_air * w0_air * rayleigh_pdf(cos_theta))
                        cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]),
                                        seg_contrib * pc * dif_val)
                    ###########################################################################
                    ######################## local estimation save ############################
                    for pi in range(path_size):
                        ##### border traversal #####
                        border_x = (le_voxel[0] + (sun_direction[0] > 0)) * voxel_size[0]
                        border_y = (le_voxel[1] + (sun_direction[1] > 0)) * voxel_size[1]
                        border_z = (le_voxel[2] + (sun_direction[2] > 0)) * voxel_size[2]
                        t_x = (border_x - le_point[0]) / (sun_direction)[0]
                        t_y = (border_y - le_point[1]) / (sun_direction)[1]
                        t_z = (border_z - le_point[2]) / (sun_direction)[2]
                        length, border_ind = argmin(t_x, t_y, t_z)
                        if cloud_mask[le_voxel[0], le_voxel[1], le_voxel[2]]:
                            cuda.atomic.add(total_grad, (le_voxel[0], le_voxel[1], le_voxel[2]), -length * pc * dif_val)
                        ###############################
                        step_in_direction(le_point, sun_direction, length)
                        le_voxel[border_ind] += sign(sun_direction[border_ind])
                    # Last Step
                    length = calc_distance(le_point, next_point)
                    if cloud_mask[le_voxel[0], le_voxel[1], le_voxel[2]]:
                        cuda.atomic.add(total_grad, (le_voxel[0], le_voxel[1], le_voxel[2]), -length * pc * dif_val)
                    ######################## local estimation save ############################
                    ###########################################################################

                    cos_theta_scatter = dot_3d(direction, new_direction)
                    assign_3d(direction, new_direction)
                    seg += 1
                if seg!= N_seg:
                    print("differ", seg, N_seg)

        @cuda.jit()
        def render_cuda_old(job_list, camera_array, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, sun_direction
                               , rng_states, I_total):
            tid = cuda.grid(1)
            # tid = camera_array.shape[0] * camera_array.shape[1] * s + camera_array.shape[1] * i + j
            if tid < job_list.shape[0]-1:
                i, j = job_list[tid]
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                le_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                le_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
                assign_3d(current_point, camera_array[i,j, :3])
                assign_3d(direction, camera_array[i,j, 3:6])
                intersected = get_intersection_with_bbox(current_point, direction, bbox)
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                attenuation = 1.0
                beta_c = 1.0
                beta = 1.0
                seg = 0
                while intersected:
                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break
                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)

                        current_tau += length * beta
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
                    seg += 1
                    cloud_prob_scat = w0_cloud * beta_c / (w0_cloud * beta_c + w0_air * beta_air)

                    if sample_uniform(rng_states, tid) <= cloud_prob_scat:
                        HG_sample_direction(direction, g_cloud, new_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, new_direction, rng_states, tid)


                    cloud_prob = 1 - (beta_air / beta)
                    cloud_prob_scat = (w0_cloud * beta_c) / (w0_cloud * beta_c + w0_air * beta_air)
                    attenuation *= cloud_prob * w0_cloud + (1 - cloud_prob) * w0_air
                    if seg >= rr_depth:
                        attenuation *= rr_factor

                    # Local estimation to the sun
                    assign_3d(le_voxel, current_voxel)
                    assign_3d(le_point, current_point)
                    cos_theta = dot_3d(direction, sun_direction)
                    le_pdf = cloud_prob_scat * HG_pdf(cos_theta, g_cloud) \
                             + (1 - cloud_prob_scat) * rayleigh_pdf(cos_theta)

                    pc = le_pdf * attenuation
                    get_intersection_with_borders(le_point, sun_direction, bbox, next_point)
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    ###########################################################################
                    ######################## local estimation save ############################
                    tau = 0
                    for pi in range(path_size):
                        ##### border traversal #####
                        border_x = (le_voxel[0] + (sun_direction[0] > 0)) * voxel_size[0]
                        border_y = (le_voxel[1] + (sun_direction[1] > 0)) * voxel_size[1]
                        border_z = (le_voxel[2] + (sun_direction[2] > 0)) * voxel_size[2]
                        t_x = (border_x - le_point[0]) / (sun_direction)[0]
                        t_y = (border_y - le_point[1]) / (sun_direction)[1]
                        t_z = (border_z - le_point[2]) / (sun_direction)[2]
                        length, border_ind = argmin(t_x, t_y, t_z)
                        ###############################

                        beta_le = beta_cloud[le_voxel[0], le_voxel[1], le_voxel[2]] + beta_air
                        tau += beta_le * length
                        # assign_3d(camera_voxel, next_voxel)
                        step_in_direction(le_point, sun_direction, length)
                        # camera_point[border_ind] = border
                        le_voxel[border_ind] += sign(sun_direction[border_ind])
                    # Last Step
                    length = calc_distance(le_point, next_point)
                    beta_le = beta_cloud[le_voxel[0], le_voxel[1], le_voxel[2]] + beta_air
                    tau += beta_le * length
                    ######################## local estimation save ############################
                    ###########################################################################
                    pc *= math.exp(-tau)
                    cuda.atomic.add(I_total, (i, j), pc)

                    assign_3d(direction, new_direction)

        self.render_cuda = render_cuda
        self.calc_scatter_sizes = calc_scatter_sizes
        self.render_cuda_old = render_cuda_old
        self.render_differentiable_cuda = render_differentiable_cuda

    def init_cuda_param(self, Np, init=False, seed=None):
        # width, height = self.camera_array_list[cam_ind].shape[:2]
        # grid_shape = (width, height, spp)
        # Np = int(np.prod(grid_shape))
        # print(f"Np = {Np:.2e}")
        self.threadsperblock = 256
        self.blockspergrid = (Np + (self.threadsperblock - 1)) // self.threadsperblock
        # blockspergrid_x = (width + self.threadsperblock[0] - 1) // self.threadsperblock[0]
        # blockspergrid_y = (height + self.threadsperblock[1] - 1) // self.threadsperblock[1]
        # blockspergrid_z = (spp + self.threadsperblock[2] - 1) // self.threadsperblock[2]
        # self.blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        if init:
            if seed is None:
                self.seed = np.random.randint(1, int(1e9))
            else:
                self.seed = seed
            self.rng_states = create_xoroshiro128p_states(Np, seed=self.seed).copy_to_host()
            self.Np = Np


    def build_path_list(self, Np, cam_ind, spp_map=None,init_cuda=True):
        if self.dpath_contrib is not None:
            del(self.dpath_contrib)
            del(self.dgrad_contrib)
            del(self.dscatter_inds)
            del(self.djob_list)
        self.cam_ind = cam_ind
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud
        width, height = self.pixels_shapes[cam_ind]
        if spp_map is None:
            spp_map = np.ones((width, height), dtype=np.uint32) * (Np // (width * height))

        self.spp_map = spp_map
        self.create_job_list(spp_map)
        self.init_cuda_param(self.Np, init_cuda)
        blockspergrid, threadsperblock = self.blockspergrid, self.threadsperblock
        dscatter_sizes = cuda.to_device(np.empty(self.Np, dtype=np.uint16))
        djob_list = cuda.to_device(self.job_list)
        dcamera_array = self.dcamera_array_list[cam_ind]
        drng_states = cuda.to_device(self.rng_states)
        self.calc_scatter_sizes[blockspergrid, threadsperblock](djob_list, dcamera_array, self.dbeta_cloud, beta_air,
                                                         g_cloud, w0_cloud, w0_air, self.dbbox,
                                                         self.dbbox_size, self.dvoxel_size, dscatter_sizes,
                                                         drng_states)
        cuda.synchronize()
        scatter_sizes = dscatter_sizes.copy_to_host()
        del(dscatter_sizes)
        del(djob_list)
        cond = scatter_sizes != 0
        scatter_sizes = scatter_sizes[cond]
        self.job_list = self.job_list[cond]
        self.rng_states_mod = self.rng_states[:self.Np][cond]
        self.Np_nonan = np.sum(cond)
        scatter_inds = np.concatenate([np.array([0]), scatter_sizes])
        scatter_inds = np.cumsum(scatter_inds)
        self.total_scatter_num = scatter_inds[-1]
        self.dscatter_inds = cuda.to_device(scatter_inds)
        self.djob_list = cuda.to_device(self.job_list)
        self.dpath_contrib = cuda.to_device(np.empty((self.total_scatter_num), dtype=float_reg))
        self.dgrad_contrib = cuda.to_device(np.empty(self.Np_nonan, dtype=float_reg))
        # return scatter_sizes


    def render(self, I_gt=None):
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud
        width, height = self.pixels_shapes[self.cam_ind]

        self.init_cuda_param(self.Np_nonan, False)
        blockspergrid, threadsperblock = self.blockspergrid, self.threadsperblock
        dI_total = cuda.to_device(np.zeros((width,height), dtype=float_reg))
        dcamera_array = self.dcamera_array_list[self.cam_ind]
        drng_states = cuda.to_device(self.rng_states_mod)
        self.render_cuda[blockspergrid, threadsperblock](self.djob_list, dcamera_array, self.dbeta_cloud, beta_air,
                                                        g_cloud, w0_cloud, w0_air, self.dbbox,
                                                        self.dbbox_size, self.dvoxel_size, self.dsun_direction,
                                                        drng_states, self.dscatter_inds, self.dpath_contrib,
                                                         self.dgrad_contrib, dI_total)
        cuda.synchronize()

        I_total = dI_total.copy_to_host() * (self.sun_intensity/ self.spp_map)
        if I_gt is None:
            return I_total

        I_dif = I_total - I_gt
        dI_dif = cuda.to_device(I_dif)
        self.dtotal_grad.copy_to_device(np.zeros_like(self.volume.beta_cloud, dtype=float_reg))
        drng_states = cuda.to_device(self.rng_states_mod)
        self.render_differentiable_cuda[blockspergrid, threadsperblock](self.djob_list, dcamera_array, self.dbeta_cloud, beta_air,
                                                         g_cloud, w0_cloud, w0_air, self.dbbox,
                                                         self.dbbox_size, self.dvoxel_size, self.dsun_direction,
                                                         drng_states, self.dscatter_inds, self.dpath_contrib,
                                                         self.dgrad_contrib, self.dcloud_mask, dI_dif, self.dtotal_grad)
        total_grad = self.dtotal_grad.copy_to_host()/(width*height)
        return I_total, total_grad



    def render_all(self, I_gt, Np, spp_map=None,init_cuda=True):
        total_grad = 0
        for cam_ind in range(self.N_cams):
            pass

    def create_job_list(self, spp_map):
        self.Np = int(np.sum(spp_map))
        job_list = np.zeros((self.Np, 2), dtype=np.uint16)
        counter = 0
        for i in range(spp_map.shape[0]):
            for j in range(spp_map.shape[1]):
                job_list[counter:counter+spp_map[i,j], 0] = i
                job_list[counter:counter+spp_map[i,j], 1] = j
                counter += spp_map[i,j]
        self.job_list = job_list


    def set_cloud_mask(self, cloud_mask):
        self.volume.set_mask(cloud_mask)
        self.dcloud_mask.copy_to_device(cloud_mask)


    def render_old(self,  Np, cam_ind, spp_map=None,init_cuda=True):
        self.cam_ind = cam_ind
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud
        width, height = self.camera_array_list[cam_ind].shape[:2]
        if spp_map is None:
            spp_map = np.ones((width, height), dtype=np.uint32) * (Np // (width * height))

        self.spp_map = spp_map
        self.create_job_list(spp_map)
        self.init_cuda_param(self.Np, init_cuda)
        blockspergrid, threadsperblock = self.blockspergrid, self.threadsperblock
        djob_list = cuda.to_device(self.job_list)
        dcamera_array = self.dcamera_array_list[cam_ind]
        drng_states = cuda.to_device(self.rng_states)
        dI_total = cuda.to_device(np.zeros((width, height), dtype=float_reg))
        self.render_cuda_old[blockspergrid, threadsperblock](djob_list, dcamera_array, self.dbeta_cloud, beta_air,
                                                                g_cloud, w0_cloud, w0_air, self.dbbox,
                                                                self.dbbox_size, self.dvoxel_size, self.dsun_direction,
                                                                drng_states, dI_total)
        return dI_total.copy_to_host()/self.spp_map


