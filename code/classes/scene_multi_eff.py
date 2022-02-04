from classes.volume import *
from utils import  theta_phi_to_direction
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_dtype,init_xoroshiro128p_states_cpu
from cuda_utils import *
from utils import relative_distance
from utils import cuda_weight
from time import time
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from tqdm import tqdm
threadsperblock = 256


class SceneMultiEff(object):
    def __init__(self, volume: Volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob):
        print("No NE grads")
        self.rr_depth = rr_depth
        self.rr_stop_prob = rr_stop_prob
        rr_factor = 1.0 / (1.0 - self.rr_stop_prob)
        self.rr_factor = rr_factor
        self.Np = 0
        self.volume = volume
        self.sun_angles = sun_angles
        self.sun_direction = theta_phi_to_direction(*sun_angles)
        self.sun_direction += 1e-6
        self.sun_direction /= np.linalg.norm(self.sun_direction)
        # self.sun_direction[np.abs(self.sun_direction) < 1e-6] = 0
        self.cameras = cameras
        self.g_cloud = g_cloud
        self.N_cams = len(cameras)
        self.N_pixels = cameras[0].pixels
        self.is_camera_in_medium = np.zeros(self.N_cams, dtype=np.bool)
        for k in range(self.N_cams):
            self.is_camera_in_medium[k] = self.volume.grid.is_in_bbox(self.cameras[k].t)

        N_cams = self.N_cams
        self.pixels_shape = self.cameras[0].pixels
        ts = np.vstack([cam.t.reshape(1, -1) for cam in self.cameras])
        self.Ps = np.concatenate([cam.P.reshape(1, 3, 4) for cam in self.cameras], axis=0)

        # gpu array
        self.dbeta_cloud = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dbeta_zero = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dI_total = cuda.device_array((N_cams, *self.pixels_shape ), dtype=float_reg)
        self.dI_diff = cuda.device_array((N_cams, *self.pixels_shape ), dtype=float_reg)
        self.dtotal_grad = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dbbox = cuda.to_device(self.volume.grid.bbox)
        self.dbbox_size = cuda.to_device(self.volume.grid.bbox_size)
        self.dvoxel_size = cuda.to_device(self.volume.grid.voxel_size)
        self.dsun_direction = cuda.to_device(self.sun_direction)
        self.dpixels_shape = cuda.to_device(self.cameras[0].pixels)
        self.dts = cuda.to_device(ts)
        self.dPs = cuda.to_device(self.Ps)
        self.dis_in_medium = cuda.to_device(self.is_camera_in_medium)
        self.dcloud_mask = None
        self.dpath_contrib = None
        self.dgrad_contrib = None

        @cuda.jit()
        def calc_scatter_sizes(Np, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, sun_direction,
                               scatter_sizes, rng_states):

            tid = cuda.grid(1)
            if tid < Np:
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                starting_point = cuda.local.array(3, dtype=float_precis)
                current_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
                assign_3d(direction, sun_direction)
                # sample entering point
                starting_point[0] = bbox_size[0] * sample_uniform(rng_states, tid) + bbox[0, 0]

                starting_point[1] = bbox_size[1] * sample_uniform(rng_states, tid) + bbox[1, 0]

                starting_point[2] = bbox[2, 1]

                assign_3d(current_point, starting_point)
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                # for seg in range(Ns):
                seg = 0
                while True:
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

                    seg+=1

                    # sampling new direction
                    cloud_prob_scat =  w0_cloud*beta_c / (w0_cloud*beta_c + w0_air*beta_air)
                    if sample_uniform(rng_states, tid) <= cloud_prob_scat:
                        HG_sample_direction(direction, g_cloud, new_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, new_direction, rng_states, tid)
                    assign_3d(direction, new_direction)

                # voxels and scatter sizes for this path (this is not in a loop)
                scatter_sizes[tid] = seg

        @cuda.jit()
        def generate_paths( beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, sun_direction, starting_points,
                           scatter_points, scatter_inds, rng_states):

            tid = cuda.grid(1)
            if tid < starting_points.shape[1]:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                starting_point = cuda.local.array(3, dtype=float_precis)
                current_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
                assign_3d(direction, sun_direction)
                # sample entering point
                starting_point[0] = bbox_size[0] * sample_uniform(rng_states, tid) + bbox[0, 0]

                starting_point[1] = bbox_size[1] * sample_uniform(rng_states, tid) + bbox[1, 0]

                starting_point[2] = bbox[2, 1]
                assign_3d(starting_points[:, tid], starting_point)

                assign_3d(current_point, starting_point)
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                seg = 0
                while True:
                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break
                    seg_ind = scatter_ind + seg
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
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel)
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
                    # keeping track of scatter points
                    seg += 1

                    scatter_points[0, seg_ind] = current_point[0]
                    scatter_points[1, seg_ind] = current_point[1]
                    scatter_points[2, seg_ind] = current_point[2]

                    # sampling new direction
                    cloud_prob_scat =  w0_cloud*beta_c / (w0_cloud*beta_c + w0_air*beta_air)

                    if sample_uniform(rng_states, tid) <= cloud_prob_scat:
                        HG_sample_direction(direction, g_cloud, new_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, new_direction, rng_states, tid)
                    assign_3d(direction, new_direction)
                if N_seg!=seg:
                    print(seg,N_seg)




        @cuda.jit()
        def sort_scatter_points(scatter_points, starting_points, sorted_inds, sorted_scatter_inds, scatter_inds,
                                sorted_scatter_points, sorted_starting_points):
            tid = cuda.grid(1)
            if tid < sorted_inds.shape[0]:
                assign_3d(sorted_starting_points[:,tid], starting_points[:, sorted_inds[tid]])

                start_ind = scatter_inds[sorted_inds[tid]]
                sorted_start_ind = sorted_scatter_inds[tid]
                N_seg = sorted_scatter_inds[tid+1] - sorted_start_ind

                for seg in range(N_seg):
                    assign_3d(sorted_scatter_points[:, sorted_start_ind+seg], scatter_points[:, start_ind+seg])



        @cuda.jit()
        def render_main_paths_cuda(beta_cloud, beta_zero, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size,
             starting_points, scatter_points, scatter_inds, path_contrib):

            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                # next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)

                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                cloud_prob_scat = 1.0
                beta0_c = 1.0 # for type decleration
                attenuation = 1.0
                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    assign_3d(next_point, scatter_points[:, seg_ind])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    distance_and_direction(current_point, next_point, direction)
                    if seg > 0:
                        cos_theta = dot_3d(cam_direction, direction)
                        cloud_prob_scat0 = (w0_cloud*beta0_c)/(w0_cloud*beta0_c + w0_air*beta_air)
                        # angle pdf
                        attenuation *=  cloud_prob_scat * HG_pdf(cos_theta, g_cloud) + (1-cloud_prob_scat) * rayleigh_pdf(cos_theta)
                        attenuation /=  cloud_prob_scat0 * HG_pdf(cos_theta, g_cloud) + (1-cloud_prob_scat0) * rayleigh_pdf(cos_theta)
                    ###########################################################
                    ############## voxel_fixed traversal_algorithm_save #############

                    path_size = estimate_voxels_size(dest_voxel, current_voxel)

                    tau = 0
                    for pi in range(path_size):
                        # length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel)
                        ##### border traversal #####
                        border_x = (current_voxel[0] + (direction[0] > 0)) * voxel_size[0]
                        border_y = (current_voxel[1] + (direction[1] > 0)) * voxel_size[1]
                        border_z = (current_voxel[2] + (direction[2] > 0)) * voxel_size[2]
                        t_x = (border_x - current_point[0]) / direction[0]
                        t_y = (border_y - current_point[1]) / direction[1]
                        t_z = (border_z - current_point[2]) / direction[2]
                        length, border_ind = argmin(t_x, t_y, t_z)
                        ###############################
                        beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                        beta0 = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                        tau += (beta-beta0)*length
                        # assign_3d(current_voxel, next_voxel)
                        current_voxel[border_ind] += sign(direction[border_ind])
                        step_in_direction(current_point, direction, length)
                    # last step
                    length = calc_distance(current_point, next_point)
                    beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                    beta = beta_c + beta_air
                    beta0_c = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]]
                    beta0 = beta0_c + beta_air
                    tau += (beta - beta0) * length
                    assign_3d(current_point, next_point)
                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################
                    attenuation *= ((beta/beta0) * math.exp(-tau)) # length pdf
                    cloud_prob = 1 - (beta_air / beta)
                    cloud_prob_scat = (w0_cloud*beta_c)/(w0_cloud*beta_c + w0_air*beta_air)
                    attenuation *= cloud_prob * w0_cloud + (1-cloud_prob) * w0_air
                    if seg >= rr_depth:
                        attenuation *= rr_factor
                    path_contrib[seg_ind] = attenuation


        @cuda.jit()
        def render_cuda(beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, sun_direction, bbox, bbox_size, voxel_size,
                        ts, Ps, pixel_shape, is_in_medium, scatter_points, scatter_inds, path_contrib, I_diff, I_total):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                cam_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)
                assign_3d(current_point, scatter_points[:, scatter_ind])
                # assign_3d(direction, sun_direction)

                # dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                grad_sum = 0
                for seg in range(N_seg):
                    seg_ind = N_seg - 1 - seg + scatter_ind
                    # seg_ind = seg + scatter_ind
                    assign_3d(current_point, scatter_points[:, seg_ind])
                    if seg == N_seg-1:
                        assign_3d(direction, sun_direction)
                    else:
                        assign_3d(cam_point, scatter_points[:, seg_ind - 1])
                        distance_and_direction(cam_point, current_point, direction)
                    get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                    beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                    cloud_prob_scat = (w0_cloud * beta_c) / (w0_cloud * beta_c + w0_air * beta_air)

                    pc_main = path_contrib[seg_ind]
                    # path_contrib[seg_ind] = 0

                    for k in range(N_cams):
                        get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                        assign_3d(cam_point, current_point)
                        project_point(cam_point, Ps[k], pixel_shape, pixel)
                        if pixel[0] != 255:
                            distance_to_camera = distance_and_direction(cam_point, ts[k], cam_direction)
                            cos_theta = dot_3d(direction, cam_direction)
                            le_pdf = cloud_prob_scat * HG_pdf(cos_theta, g_cloud) \
                                     + (1 - cloud_prob_scat) * rayleigh_pdf(cos_theta)
                            pc = (1 / (distance_to_camera * distance_to_camera)) * le_pdf * pc_main
                            if is_in_medium[k]:
                                assign_3d(next_point, ts[k])
                            else:
                                get_intersection_with_borders(cam_point, cam_direction, bbox, next_point)

                            get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)
                            ###########################################################################
                            ######################## local estimation save ############################
                            tau = 0
                            for pi in range(path_size):
                                ##### border traversal #####
                                border_x = (current_voxel[0] + (cam_direction[0] > 0)) * voxel_size[0]
                                border_y = (current_voxel[1] + (cam_direction[1] > 0)) * voxel_size[1]
                                border_z = (current_voxel[2] + (cam_direction[2] > 0)) * voxel_size[2]
                                t_x = (border_x - cam_point[0]) / cam_direction[0]
                                t_y = (border_y - cam_point[1]) / cam_direction[1]
                                t_z = (border_z - cam_point[2]) / cam_direction[2]
                                length, border_ind = argmin(t_x, t_y, t_z)
                                ###############################
                                # length, border, border_ind = travel_to_voxels_border_fast(camera_point, camera_voxel, cam_direction,
                                #                                  voxel_size)
                                beta_cam = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                                tau += beta_cam * length
                                # assign_3d(camera_voxel, next_voxel)
                                step_in_direction(cam_point, cam_direction, length)
                                # camera_point[border_ind] = border
                                current_voxel[border_ind] += sign(cam_direction[border_ind])
                            # Last Step
                            length = calc_distance(cam_point, next_point)
                            beta_cam = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                            tau += beta_cam * length
                            ######################## local estimation save ############################
                            ###########################################################################
                            pc *= math.exp(-tau)
                            cuda.atomic.add(I_total, (k, pixel[0], pixel[1]), pc)
                            grad_sum += pc * I_diff[k,pixel[0],pixel[1]]
                    path_contrib[seg_ind] = grad_sum
                    # assign_3d(current_point, scatter_points[:, seg_ind])
                    # distance_and_direction(current_point, scatter_points[:, seg_ind + 1], direction)
                    #
                    # assign_3d(cam_point, scatter_points[:, seg_ind+1])
                    # distance_and_direction(current_point, cam_point, direction)
                    # assign_3d(current_point, scatter_points[:, seg_ind])





        @cuda.jit()
        def render_differentiable_cuda(beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, starting_points, scatter_points, scatter_inds,
                                       path_contrib, cloud_mask, total_grad):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind

                grid_shape = beta_cloud.shape

                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                # camera_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                # camera_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)
                # pixel = cuda.local.array(2, dtype=np.uint8)
                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                beta_c = 1.0  # for type decleration

                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    assign_3d(next_point, scatter_points[:, seg_ind])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    distance_and_direction(current_point, next_point, direction)
                    # GRAD CALCULATION (SCATTERING)
                    grad_temp = path_contrib[seg_ind]
                    if seg > 0:
                        if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                            cos_theta_scatter = dot_3d(cam_direction, direction)
                            omega_fp_c = w0_cloud * HG_pdf(cos_theta_scatter,g_cloud)
                            # seg_contrib = (beta_c + (rayleigh_pdf(cos_theta_scatter)/HG_pdf(cos_theta_scatter,g_cloud)) * beta_air ) **(-1)
                            seg_contrib = omega_fp_c / \
                                          (beta_c * omega_fp_c + beta_air * w0_air * rayleigh_pdf(cos_theta_scatter)) # scatter fix
                            cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]), seg_contrib * grad_temp)
                    ###########################################################
                    ############## voxel_fixed traversal_algorithm_save #############
                    path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    for pi in range(path_size):
                        ##### border traversal #####
                        border_x = (current_voxel[0] + (direction[0] > 0)) * voxel_size[0]
                        border_y = (current_voxel[1] + (direction[1] > 0)) * voxel_size[1]
                        border_z = (current_voxel[2] + (direction[2] > 0)) * voxel_size[2]
                        t_x = (border_x - current_point[0]) / direction[0]
                        t_y = (border_y - current_point[1]) / direction[1]
                        t_z = (border_z - current_point[2]) / direction[2]
                        length, border_ind = argmin(t_x, t_y, t_z)
                        ###############################
                        if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                            grad = -length * grad_temp
                            cuda.atomic.add(total_grad, (current_voxel[0],current_voxel[1],current_voxel[2]), grad)
                        current_voxel[border_ind] += sign(direction[border_ind])
                        step_in_direction(current_point, direction, length)

                    # last step
                    length = calc_distance(current_point, next_point)
                    if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                        cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]), -length * grad_temp)
                    assign_3d(current_point, next_point)

                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################
                    beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + divide_beta_eps
                    # omega_beta_c = w0_cloud * beta_c
                    # le_contrib = (beta_c + (w0_air/w0_cloud) * beta_air) **(-1)
                    # le_contrib = omega_beta_c / (omega_beta_c + w0_air*beta_air) # scatter fix
                    # le_contrib -= w0_cloud / (omega_beta_c + w0_air * beta_air)
                    # le_contrib -= 1 / (beta_c + beta_air)
                    # for k in range(N_cams):
                    #     project_point(current_point, Ps[k], pixel_shape, pixel)
                    #     if pixel[0] != 255:
                    #         assign_3d(camera_voxel, current_voxel)
                    #         assign_3d(camera_point, current_point)
                    #         distance_and_direction(camera_point, ts[k], cam_direction)
                    #         cos_theta = dot_3d(direction, cam_direction)
                    #
                    #         if is_in_medium[k]:
                    #             assign_3d(next_point, ts[k])
                    #         else:
                    #             get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)
                    #         get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    #         path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    #
                    #         grad_temp = path_contrib[k, seg_ind] * I_diff[k, pixel[0], pixel[1]]
                    #         # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                    #         if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]]:
                    #             # seg_contrib = le_contrib + (beta_c + (rayleigh_pdf(cos_theta) / HG_pdf(cos_theta, g_cloud)) * beta_air) ** (-1)
                    #             omega_fp_c = w0_cloud * HG_pdf(cos_theta, g_cloud)
                    #             seg_contrib = omega_fp_c / \
                    #                           (beta_c * omega_fp_c + beta_air * w0_air * rayleigh_pdf(cos_theta))
                    #             cuda.atomic.add(total_grad, (camera_voxel[0],camera_voxel[1],camera_voxel[2]), seg_contrib * grad_temp)
                    #         ###########################################################################
                    #         ######################## local estimation save ############################
                    #         for pi in range(path_size):
                    #             ##### border traversal #####
                    #             border_x = (camera_voxel[0] + (cam_direction[0] > 0)) * voxel_size[0]
                    #             border_y = (camera_voxel[1] + (cam_direction[1] > 0)) * voxel_size[1]
                    #             border_z = (camera_voxel[2] + (cam_direction[2] > 0)) * voxel_size[2]
                    #             t_x = (border_x - camera_point[0]) / cam_direction[0]
                    #             t_y = (border_y - camera_point[1]) / cam_direction[1]
                    #             t_z = (border_z - camera_point[2]) / cam_direction[2]
                    #             length, border_ind = argmin(t_x, t_y, t_z)
                    #             ###############################
                    #             # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                    #             if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]] and False:
                    #                 grad = -length * grad_temp
                    #                 cuda.atomic.add(total_grad,(camera_voxel[0], camera_voxel[1], camera_voxel[2]), grad)
                    #             # next point and voxel
                    #             camera_voxel[border_ind] += sign(cam_direction[border_ind])
                    #             step_in_direction(camera_point, cam_direction, length)
                    #         # Last Step
                    #         length = calc_distance(camera_point, next_point)
                    #         # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                    #         if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]] and False:
                    #             grad = -length * grad_temp
                    #             cuda.atomic.add(total_grad, (camera_voxel[0], camera_voxel[1], camera_voxel[2]), grad)
                    #         ######################## local estimation save ############################
                    #         ###########################################################################

                    assign_3d(cam_direction, direction)  # using cam direction as temp direction


        @cuda.jit()
        def space_curving_cuda(pixels_mat, spp, RKs_inv, ts, bbox, bbox_size, voxel_size, grid_shape, rng_states,
                               grid_counter):
            tid = cuda.grid(1)
            if tid < pixels_mat.shape[1]:
                cam_ind, j, i = pixels_mat[:,tid]
                pixel = cuda.local.array(3, dtype=float_precis)
                current_point = cuda.local.array(3, dtype=float_precis)
                dest_point = cuda.local.array(3, dtype=float_precis)
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                direction = cuda.local.array(3, dtype=float_precis)
                pixel[2] = 1

                for s in range(spp):
                    assign_3d(current_point, ts[cam_ind])
                    pixel[0] = j + sample_uniform(rng_states, tid)
                    pixel[1] = i + sample_uniform(rng_states, tid)
                    mat_dot_vec(RKs_inv[cam_ind], pixel, direction)
                    norm_3d(direction)
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






        self.calc_scatter_sizes = calc_scatter_sizes
        self.generate_paths = generate_paths
        self.sort_scatter_points = sort_scatter_points
        self.render_cuda = render_cuda
        self.render_main_paths_cuda = render_main_paths_cuda
        self.render_differentiable_cuda = render_differentiable_cuda
        self.space_curving_cuda = space_curving_cuda

    def init_cuda_param(self, Np, init=False, seed=None):
        self.threadsperblock = threadsperblock
        self.blockspergrid = (Np + (threadsperblock - 1)) // threadsperblock
        if init:
            if seed is None:
                self.seed = np.random.randint(1, int(1e9))
            else:
                self.seed = seed
            self.rng_states = np.empty(shape=Np, dtype=xoroshiro128p_dtype)
            init_xoroshiro128p_states_cpu(self.rng_states, self.seed, 0)
            self.Np = Np

    def build_paths_list(self, Np, to_sort=True, to_print=False):
        # inputs
        del(self.dpath_contrib)
        # del(self.dgrad_contrib)
        self.init_cuda_param(Np, self.Np!=Np)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.dbeta_zero.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        # outputs
        start = time()
        dscatter_sizes = cuda.to_device(np.empty(Np, dtype=np.uint8))
        drng_states = cuda.to_device(self.rng_states)
        self.calc_scatter_sizes[blockspergrid, threadsperblock] \
            (Np, self.dbeta_zero, beta_air, self.g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size,
             self.dvoxel_size,self.dsun_direction, dscatter_sizes, drng_states)

        cuda.synchronize()
        if to_print:
            print("calc_scatter_sizes:",time()-start)
        scatter_sizes = dscatter_sizes.copy_to_host()
        del(dscatter_sizes)
        rng_states_temp = drng_states.copy_to_host()
        del(drng_states)

        cond = scatter_sizes != 0
        Np_nonan = np.sum(cond)
        scatter_sizes = scatter_sizes[cond]
        rng_states_mod = self.rng_states[cond]
        if to_sort:
            sorted_inds = np.argsort(scatter_sizes)
            scatter_sizes = scatter_sizes[sorted_inds]
            rng_states_mod = rng_states_mod[sorted_inds]
        # rng_states_mod = np.concatenate([rng_states_mod, np.zeros(Np-Np_nonan, dtype=rng_states_mod.dtype)])
        drng_states_sorted = cuda.to_device(rng_states_mod)
        self.rng_states = rng_states_temp
        scatter_inds = np.concatenate([np.array([0]), scatter_sizes])
        scatter_inds = np.cumsum(scatter_inds)
        total_scatter_num = scatter_inds[-1]
        dscatter_inds = cuda.to_device(scatter_inds)
        dstarting_points = cuda.to_device(np.zeros((3,Np_nonan), dtype=float_precis))
        dscatter_points = cuda.to_device(np.zeros((3,total_scatter_num), dtype=float_precis))

        self.init_cuda_param(Np_nonan)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        start = time()

        self.generate_paths[blockspergrid, threadsperblock] \
            (self.dbeta_zero, beta_air, self.g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size, self.dvoxel_size,
             self.dsun_direction, dstarting_points, dscatter_points, dscatter_inds, drng_states_sorted)

        cuda.synchronize()
        del(drng_states_sorted)
        if to_print:
            print("generate_paths:", time() - start)
        # del(dscatter_inds)
        # del(dstarting_inds)

        self.total_num_of_scatter = total_scatter_num
        self.Np_nonan = Np_nonan
        self.dpath_contrib = cuda.to_device(np.empty(self.total_num_of_scatter, dtype=float_reg))
        # self.dgrad_contrib = cuda.to_device(np.empty(self.total_num_of_scatter, dtype=float_reg))
        return dstarting_points, dscatter_points, dscatter_inds




    def render(self, cuda_paths, I_diff=None, to_print=False):
        # east declerations
        require_grads = not I_diff is None
        Np_nonan = self.Np_nonan
        N_cams = len(self.cameras)
        pixels_shape = self.cameras[0].pixels
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud
        self.init_cuda_param(Np_nonan)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)

        start = time()

        # dpath_contrib = cuda.to_device(np.empty((self.N_cams, self.total_num_of_scatter), dtype=float_reg))

        # dcounter = cuda.to_device(np.zeros(1, dtype=int))
        self.render_main_paths_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size,
             self.dvoxel_size,  *cuda_paths, self.dpath_contrib)
        cuda.synchronize()
        if to_print:
            print("render_main took:", time() - start)

        start = time()
        self.dI_total.copy_to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=float_reg))
        if I_diff is None:
            I_diff = np.zeros((N_cams, *pixels_shape), dtype=float_reg)


        self.dI_diff.copy_to_device(I_diff)
        self.render_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, beta_air, g_cloud, w0_cloud, w0_air, self.dsun_direction, self.dbbox, self.dbbox_size,
             self.dvoxel_size, self.dts, self.dPs, self.dpixels_shape, self.dis_in_medium, *cuda_paths[1:], self.dpath_contrib,
             self.dI_diff, self.dI_total)
        cuda.synchronize()

        # print("voxel counter",dcounter.copy_to_host())
        I_total = self.dI_total.copy_to_host()
        I_total /= self.Np
        if to_print:
            print("render_cuda took:",time() - start)
        # return I_total
        if not require_grads:
            # del dpath_contrib
            return I_total

        ##### differentiable part ####
        self.dtotal_grad.copy_to_device(np.zeros_like(self.volume.beta_cloud, dtype=float_reg))


        start = time()
        self.render_differentiable_cuda[blockspergrid, threadsperblock]\
            (self.dbeta_cloud, beta_air, g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size, self.dvoxel_size,
                                   *cuda_paths, self.dpath_contrib, self.dcloud_mask, self.dtotal_grad)

        cuda.synchronize()
        if to_print:
            print("render_differentiable_cuda took:", time() - start)
        total_grad = self.dtotal_grad.copy_to_host()

        total_grad /= (self.Np * N_cams)
        return I_total, total_grad

    def space_curving(self, I_total, image_threshold, hit_threshold, spp):
        shape = self.volume.grid.shape
        pixels = self.cameras[0].pixels
        I_mask = np.zeros(I_total.shape, dtype=bool)
        I_total_norm = (I_total-I_total.min())/(I_total.max()-I_total.min())
        # I_mask[I_total/np.mean(I_total) > image_threshold] = True
        I_mask[I_total_norm > image_threshold] = True
        # plt.figure(figsize=(I_total.shape[0],2))
        # for k in range(self.N_cams):
        #     img_and_mask = np.vstack([I_total[k]/np.max(I_total[k]), I_mask[k]])
        #     ax = plt.subplot(1, self.N_cams, k+1)
        #     ax.imshow(img_and_mask, cmap="gray")
        #     plt.axis('off')

        # plt.tight_layout()
        # plt.show()
        dgrid_counter = cuda.to_device(np.zeros((*shape, self.N_cams), dtype=np.bool))
        RKs_inv = np.concatenate([(cam.R @ cam.K_inv).reshape(1,3,3) for cam in self.cameras])
        dRKs_inv = cuda.to_device(RKs_inv)

        grid_shape = self.volume.beta_cloud.shape
        dgrid_shape = cuda.to_device(grid_shape)

        #building pxiel_mat
        jj, kk, ii = np.meshgrid(np.arange(pixels[0]), np.arange(self.N_cams), np.arange(pixels[1]))
        ii = ii[I_mask == True]
        jj = jj[I_mask == True]
        kk = kk[I_mask == True]
        pixel_mat = np.vstack([kk, jj, ii])
        dpixel_mat = cuda.to_device(pixel_mat)
        print(pixel_mat.shape[1])
        self.init_cuda_param(pixel_mat.shape[1])
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.space_curving_cuda[blockspergrid, threadsperblock] \
        (dpixel_mat, spp, dRKs_inv, self.dts,self.dbbox, self.dbbox_size, self.dvoxel_size, dgrid_shape, self.rng_states, dgrid_counter)
        grid_counter = dgrid_counter.copy_to_host()
        grid_counter = grid_counter.astype(np.bool)
        grid_mean = np.mean(grid_counter, axis=-1)
        cloud_mask = np.zeros(grid_shape, dtype=bool)
        cloud_mask[grid_mean>hit_threshold] = True

        return cloud_mask
        # pixel_mat, N_cams, width, height, spp, I_mask, RKs_inv, ts, bbox, bbox_size, voxel_size, grid_shape, grid_counter):

    def find_best_initialization(self, beta_gt, I_gt, beta_low, beta_high, N_samples, Np, to_plot=False):
        betas = np.linspace(beta_low, beta_high, N_samples)
        losses = []
        # rel_dists = []
        if to_plot:
            plt.figure()
        for beta in tqdm(betas):
            self.volume.beta_cloud[self.volume.cloud_mask] = beta
            cuda_path = self.build_paths_list(Np)
            I = self.render(cuda_path)
            del(cuda_path)
            dif = I-I_gt
            loss = 0.5 * np.sum(dif * dif)
            rel_dist = relative_distance(beta_gt, beta)
            if to_plot:
                plt.scatter(loss, rel_dist, color="b")
            losses.append(loss)

        if to_plot:
            plt.grid()
            plt.xlabel("loss")
            plt.ylabel("rel_dist")
            plt.show()
        min_ind = np.argmin(losses)
        print(relative_distance(beta_gt,betas[min_ind]))
        return betas[min_ind]





    def set_cloud_mask(self, cloud_mask):
        self.volume.set_mask(cloud_mask)
        self.dcloud_mask = cuda.to_device(cloud_mask)

    def set_cameras(self, cameras):
        self.cameras = cameras
        self.is_camera_in_medium = np.zeros(self.N_cams, dtype=np.bool)
        for k in range(self.N_cams):
            self.is_camera_in_medium[k] = self.volume.grid.is_in_bbox(self.cameras[k].t)
        self.pixels_shape = self.cameras[0].pixels
        ts = np.vstack([cam.t.reshape(1, -1) for cam in self.cameras])
        Ps = np.concatenate([cam.P.reshape(1, 3, 4) for cam in self.cameras], axis=0)
        self.dts.copy_to_device(ts)
        self.dPs.copy_to_device(Ps)

    def reset_cameras(self, cameras):
        self.cameras = cameras
        self.N_cams = len(cameras)
        self.N_pixels = cameras[0].pixels
        self.is_camera_in_medium = np.zeros(self.N_cams, dtype=np.bool)
        for k in range(self.N_cams):
            self.is_camera_in_medium[k] = self.volume.grid.is_in_bbox(self.cameras[k].t)
        self.pixels_shape = self.cameras[0].pixels
        ts = np.vstack([cam.t.reshape(1, -1) for cam in self.cameras])
        self.Ps = np.concatenate([cam.P.reshape(1, 3, 4) for cam in self.cameras], axis=0)
        self.dpixels_shape = cuda.to_device(self.cameras[0].pixels)
        self.dts = cuda.to_device(ts)
        self.dPs = cuda.to_device(self.Ps)
        self.dis_in_medium = cuda.to_device(self.is_camera_in_medium)
        self.dI_total = cuda.device_array((self.N_cams, *self.pixels_shape), dtype=float_reg)

    def upscale_cameras(self, ps_new):
        del(self.dI_total)
        del(self.dpixels_shape)
        for cam in self.cameras:
            cam.update_pixels(np.array([ps_new, ps_new]))
        self.pixels_shape = np.array([ps_new, ps_new])
        self.dI_total = cuda.device_array((self.N_cams, *self.pixels_shape), dtype=float_reg)
        self.dpixels_shape = cuda.to_device(self.pixels_shape)
        self.set_cameras(self.cameras)

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
        text += f"g_cloud={self.g_cloud} \n\n"
        return text




