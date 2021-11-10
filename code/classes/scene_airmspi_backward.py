from classes.volume import *
from utils import  theta_phi_to_direction
from numba.cuda.random import create_xoroshiro128p_states
from cuda_utils import *
import matplotlib.pyplot as plt
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
        self.total_cam_num = len(camera_array_list)
        cam_num = self.total_cam_num
        max_pix_i = np.max([camera_array_list[cam_ind].shape[0] for cam_ind in range(cam_num)])
        max_pix_j = np.max([camera_array_list[cam_ind].shape[1] for cam_ind in range(cam_num)])
        self.pixels_shape = np.array([max_pix_i, max_pix_j])
        self.pixels_shapes = [camera_array_list[cam_ind].shape[:2] for cam_ind in range(cam_num)]
        self.total_pix_num = np.sum([np.prod(shape) for shape in self.pixels_shapes])
        # gpu array
        self.dbeta_cloud = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dtotal_grad = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dcloud_mask = cuda.device_array(self.volume.beta_cloud.shape, dtype=bool)
        self.dcloud_mask.copy_to_device(self.volume.cloud_mask)
        self.dbbox = cuda.to_device(self.volume.grid.bbox)
        self.dbbox_size = cuda.to_device(self.volume.grid.bbox_size)
        self.dvoxel_size = cuda.to_device(self.volume.grid.voxel_size)
        self.dsun_direction = cuda.to_device(self.sun_direction)
        camera_array_pad = np.zeros((cam_num, *self.pixels_shape, 6), dtype=float_reg)
        for cam_ind in range(cam_num):
            pix_shape = self.pixels_shapes[cam_ind]
            camera_array_pad[cam_ind, :pix_shape[0], :pix_shape[1], :] = camera_array_list[cam_ind]
        # self.dcamera_array_list = [cuda.to_device(camera_array_list[cam_ind]) for cam_ind in range(N_cams)]
        self.dcamera_array_list = cuda.to_device(camera_array_pad)
        self.spp_map = None
        self.dpath_contrib = None
        self.dgrad_contrib = None

        @cuda.jit()
        def calc_scatter_sizes(job_list, camera_array_list, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size,
                               voxel_size, scatter_sizes, rng_states):

            tid = cuda.grid(1)
            # tid = camera_array.shape[0] * camera_array.shape[1] * s + camera_array.shape[1] * i + j
            if tid < job_list.shape[0]:
                cam_ind, i, j = job_list[tid]
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
                assign_3d(current_point, camera_array_list[cam_ind, i, j, :3])
                assign_3d(direction, camera_array_list[cam_ind, i, j, 3:6])
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
        def render_cuda(job_list, camera_array_list, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, sun_direction
                               , rng_states, scatter_inds, path_contrib, grad_contrib, I_total):
            tid = cuda.grid(1)
            # tid = camera_array.shape[0] * camera_array.shape[1] * s + camera_array.shape[1] * i + j
            if tid < job_list.shape[0]-1:
                cam_ind, i, j = job_list[tid]
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
                assign_3d(current_point, camera_array_list[cam_ind, i, j, :3])
                assign_3d(direction, camera_array_list[cam_ind, i, j, 3:6])
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
                    cuda.atomic.add(I_total, (cam_ind, i, j), pc)
                    pc_sum += pc
                    seg += 1

                    assign_3d(direction, new_direction)

                grad_contrib[tid] = pc_sum
                if seg!= N_seg:
                    print("reg", seg, N_seg)


        @cuda.jit()
        def render_differentiable_cuda(job_list, camera_array_list, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size,
                        voxel_size, sun_direction
                        , rng_states, scatter_inds, path_contrib, grad_contrib, cloud_mask, I_dif, total_grad):
            tid = cuda.grid(1)
            # tid = camera_array.shape[0] * camera_array.shape[1] * s + camera_array.shape[1] * i + j
            if tid < job_list.shape[0] - 1:
                cam_ind, i, j = job_list[tid]
                grid_shape = beta_cloud.shape
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                dif_val = I_dif[cam_ind, i,j]
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
                assign_3d(current_point, camera_array_list[cam_ind, i, j, :3])
                assign_3d(direction, camera_array_list[cam_ind, i, j, 3:6])
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
        def space_curving_cuda(pixels_mat, spp, camera_array_list, bbox, bbox_size, voxel_size, grid_shape, rng_states,
                               grid_counter):
            tid = cuda.grid(1)
            if tid < pixels_mat.shape[1]:
                cam_ind, i, j = pixels_mat[:, tid]
                current_point = cuda.local.array(3, dtype=float_precis)
                dest_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                direction_temp = cuda.local.array(3, dtype=float_precis)
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                if i != 0 and j !=0 and camera_array_list[cam_ind, i, j, 0] != 0:
                    for s in range(spp):
                        coin = sample_uniform(rng_states, tid)
                        alpha = 2 * (sample_uniform(rng_states, tid) - 0.5)
                        delta = sign(alpha)
                        alpha = abs(alpha)
                        assign_3d(direction, camera_array_list[cam_ind, i, j, 3:6])
                        if coin < 0.5:
                            assign_3d(direction_temp, camera_array_list[cam_ind, i + delta, j , 3:6])
                        else:
                            assign_3d(direction_temp, camera_array_list[cam_ind, i , j + delta, 3:6])

                        direction[0] += alpha * (direction_temp[0] - direction[0])
                        direction[1] += alpha * (direction_temp[1] - direction[1])
                        direction[2] += alpha * (direction_temp[2] - direction[2])



                        assign_3d(current_point, camera_array_list[cam_ind, i, j, :3])

                        intersected = get_intersection_with_bbox(current_point, direction, bbox)
                        if intersected:
                            get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                            get_intersection_with_borders(current_point, direction, bbox, dest_point)
                            # print2_3d(current_point, dest_point)
                            get_voxel_of_point(dest_point, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)
                            for ps in range(path_size):
                                if not is_voxel_valid(current_voxel, grid_shape):
                                    print("bug")
                                # cuda.atomic.add(grid_counter,
                                #                 (current_voxel[0], current_voxel[1], current_voxel[2], cam_ind), True)
                                grid_counter[current_voxel[0], current_voxel[1], current_voxel[2], cam_ind] = True
                                border_x = (current_voxel[0] + (direction[0] > 0)) * voxel_size[0]
                                border_y = (current_voxel[1] + (direction[1] > 0)) * voxel_size[1]
                                border_z = (current_voxel[2] + (direction[2] > 0)) * voxel_size[2]
                                t_x = (border_x - current_point[0]) / direction[0]
                                t_y = (border_y - current_point[1]) / direction[1]
                                t_z = (border_z - current_point[2]) / direction[2]
                                length, border_ind = argmin(t_x, t_y, t_z)
                                # if ps < path_size - 1:
                                current_voxel[border_ind] += sign(direction[border_ind])
                                step_in_direction(current_point, direction, length)

        @cuda.jit()
        def compute_std_map_cuda(job_list, I_total, scatter_inds, path_contrib, std_map):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                cam_ind, i, j = job_list[tid]
                pc = 0
                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    pc += path_contrib[seg_ind]

                temp = pc - I_total[cam_ind, i, j]
                cuda.atomic.add(std_map, (cam_ind, i, j), temp*temp)

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
        self.space_curving_cuda = space_curving_cuda
        self.compute_std_map_cuda = compute_std_map_cuda

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
        self.init_Np = Np

    def build_path_list(self, Np, cam_inds=None, init_cuda=True):
        if self.dpath_contrib is not None:
            del(self.dpath_contrib)
            del(self.dgrad_contrib)
            del(self.dscatter_inds)
            del(self.djob_list)

        if cam_inds is None:
            cam_inds = np.arange(self.total_cam_num)
        self.cam_inds = cam_inds
        self.N_cams = len(cam_inds)
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud
        if self.spp_map is None:
            spp_map = np.zeros((self.total_cam_num, *self.pixels_shape), dtype=np.uint32)
            for cam_ind in self.cam_inds:
                width, height = self.pixels_shapes[cam_ind]
                spp_map[cam_ind, :width, :height] = 1
            spp_map = spp_map * (Np//np.sum(spp_map))
            self.create_job_list(spp_map)

        self.init_cuda_param(self.Np, init_cuda)
        blockspergrid, threadsperblock = self.blockspergrid, self.threadsperblock
        dscatter_sizes = cuda.to_device(np.empty(self.Np, dtype=np.uint16))
        djob_list = cuda.to_device(self.job_list)
        drng_states = cuda.to_device(self.rng_states)
        self.calc_scatter_sizes[blockspergrid, threadsperblock](djob_list, self.dcamera_array_list, self.dbeta_cloud, beta_air,
                                                         g_cloud, w0_cloud, w0_air, self.dbbox,
                                                         self.dbbox_size, self.dvoxel_size, dscatter_sizes,
                                                         drng_states)
        cuda.synchronize()
        scatter_sizes = dscatter_sizes.copy_to_host()
        del(dscatter_sizes)
        del(djob_list)
        cond = scatter_sizes != 0
        scatter_sizes = scatter_sizes[cond]
        job_list_mod = self.job_list[cond]
        self.rng_states_mod = self.rng_states[:self.Np][cond]
        self.Np_nonan = np.sum(cond)
        scatter_inds = np.concatenate([np.array([0]), scatter_sizes])
        scatter_inds = np.cumsum(scatter_inds)
        self.total_scatter_num = scatter_inds[-1]
        self.dscatter_inds = cuda.to_device(scatter_inds)
        self.djob_list = cuda.to_device(job_list_mod)
        self.dpath_contrib = cuda.to_device(np.empty((self.total_scatter_num), dtype=float_reg))
        self.dgrad_contrib = cuda.to_device(np.empty(self.Np_nonan, dtype=float_reg))
        # return scatter_sizes


    def render(self, I_gt=None, compute_spp_map=False):
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud

        self.init_cuda_param(self.Np_nonan, False)
        blockspergrid, threadsperblock = self.blockspergrid, self.threadsperblock
        dI_total = cuda.to_device(np.zeros((self.total_cam_num, *self.pixels_shape), dtype=float_reg))
        drng_states = cuda.to_device(self.rng_states_mod)
        self.render_cuda[blockspergrid, threadsperblock](self.djob_list, self.dcamera_array_list, self.dbeta_cloud, beta_air,
                                                        g_cloud, w0_cloud, w0_air, self.dbbox,
                                                        self.dbbox_size, self.dvoxel_size, self.dsun_direction,
                                                        drng_states, self.dscatter_inds, self.dpath_contrib,
                                                         self.dgrad_contrib, dI_total)
        cuda.synchronize()

        I_total = dI_total.copy_to_host()
        cond = self.spp_map!=0
        I_total[cond] *= (self.sun_intensity / self.spp_map[cond])
        if I_gt is None:
            return I_total

        I_dif = I_total - I_gt
        dI_dif = cuda.to_device(I_dif)
        self.dtotal_grad.copy_to_device(np.zeros_like(self.volume.beta_cloud, dtype=float_reg))
        drng_states = cuda.to_device(self.rng_states_mod)
        self.render_differentiable_cuda[blockspergrid, threadsperblock](self.djob_list, self.dcamera_array_list, self.dbeta_cloud, beta_air,
                                                         g_cloud, w0_cloud, w0_air, self.dbbox,
                                                         self.dbbox_size, self.dvoxel_size, self.dsun_direction,
                                                         drng_states, self.dscatter_inds, self.dpath_contrib,
                                                         self.dgrad_contrib, self.dcloud_mask, dI_dif, self.dtotal_grad)
        total_grad = self.dtotal_grad.copy_to_host()/self.total_pix_num


        if compute_spp_map:
            dI_total = cuda.to_device(I_total/self.sun_intensity)
            dstd_map = cuda.to_device(np.zeros_like(I_total))
            self.compute_std_map_cuda[blockspergrid, threadsperblock](self.djob_list, dI_total, self.dscatter_inds, self.dpath_contrib, dstd_map)
            std_map = dstd_map.copy_to_host()
            std_map /= np.sum(std_map)
            spp_map = (self.init_Np * std_map).astype(np.uint32)
            self.create_job_list(spp_map)

        return I_total, total_grad


    def space_curving(self, I_total, image_threshold, hit_threshold, spp):
        shape = self.volume.grid.shape
        pixels = self.pixels_shape
        I_mask = np.zeros(I_total.shape, dtype=bool)
        I_total_norm = (I_total - I_total.min()) / (I_total.max() - I_total.min())
        # I_mask[I_total/np.mean(I_total) > image_threshold] = True
        for k in range(self.total_cam_num):
            I_mask[k][I_total_norm[k] > image_threshold[k]] = True
        plt.figure(figsize=(I_total.shape[0], 2))
        for k in range(self.total_cam_num):
            img_and_mask = np.vstack([I_total[k] / np.max(I_total[k]), I_mask[k]])
            ax = plt.subplot(1, self.total_cam_num, k + 1)
            ax.imshow(img_and_mask, cmap="gray")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        dgrid_counter = cuda.to_device(np.zeros((*shape, self.total_cam_num), dtype=np.bool))

        grid_shape = self.volume.beta_cloud.shape
        dgrid_shape = cuda.to_device(grid_shape)

        # building pxiel_mat
        jj, kk, ii = np.meshgrid(np.arange(pixels[0]), np.arange(self.total_cam_num), np.arange(pixels[1]))
        ii = ii[I_mask == True]
        jj = jj[I_mask == True]
        kk = kk[I_mask == True]
        pixel_mat = np.vstack([kk, jj, ii])
        dpixel_mat = cuda.to_device(pixel_mat)
        print(pixel_mat.shape[1])
        self.init_cuda_param(pixel_mat.shape[1], init=True)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.space_curving_cuda[blockspergrid, threadsperblock] \
            (dpixel_mat, spp, self.dcamera_array_list, self.dbbox, self.dbbox_size, self.dvoxel_size, dgrid_shape,
             self.rng_states, dgrid_counter)
        grid_counter = dgrid_counter.copy_to_host()
        grid_counter = grid_counter.astype(np.bool)
        grid_mean = np.mean(grid_counter, axis=-1)
        cloud_mask = np.zeros(grid_shape, dtype=bool)
        cloud_mask[grid_mean > hit_threshold] = True

        return cloud_mask


    def create_job_list(self, spp_map):
        self.spp_map = spp_map
        self.Np = int(np.sum(spp_map))
        job_list = np.zeros((self.Np, 3), dtype=np.uint16)
        counter = 0
        for cam_ind in range(self.cam_inds):
            for i in range(spp_map.shape[1]):
                for j in range(spp_map.shape[2]):
                    job_list[counter:counter+spp_map[cam_ind, i,j], 0] = cam_ind
                    job_list[counter:counter+spp_map[cam_ind, i,j], 1] = i
                    job_list[counter:counter+spp_map[cam_ind, i,j], 2] = j
                    counter += spp_map[cam_ind, i, j]
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


