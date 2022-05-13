from classes.volume import *
from utils import  theta_phi_to_direction
from numba.cuda.random import create_xoroshiro128p_states, init_xoroshiro128p_states_cpu, xoroshiro128p_dtype
from cuda_utils import *
from utils import relative_distance
from utils import cuda_weight
from time import time
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

threadsperblock = 256
def torch_lexsort(a, dim=-1):
    assert dim == -1  # Transpose if you want differently
    assert a.ndim == 2  # Not sure what is numpy behaviour with > 2 dim
    # To be consistent with numpy, we flip the keys (sort by last row first)
    a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
    return torch.argsort(inv)

class SceneSeed(object):
    def __init__(self, volume: Volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob, N_batches=1):
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

        self.N_batches = N_batches
        # gpu array
        self.dbeta_cloud = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dbeta_zero = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dI_total = cuda.device_array((N_cams, *self.pixels_shape ), dtype=float_reg)
        self.dI_diff = cuda.device_array((N_cams, *self.pixels_shape ), dtype=float_reg)
        self.dtotal_grad = cuda.device_array((self.N_batches,*self.volume.beta_cloud.shape), dtype=float_reg)
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
        self.dlengths = None
        self.dscatter_inds = None
        self.init = False
        self.rng_states_updated = None
        self.drng_states = None
        @cuda.jit()
        def calc_scatter_sizes(Np, beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, sun_direction,
                               scatter_sizes, rng_states):

            tid = cuda.grid(1)
            if tid < Np:
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                pixel = cuda.local.array(2, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                starting_point = cuda.local.array(3, dtype=float_precis)
                current_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                camera_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)
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
                total_length = 0
                total_voxel_sizes = 0
                total_cam_voxel_sizes = 0
                while True:
                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break
                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    current_length = 0
                    current_voxel_sizes = 0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)
                        current_tau += length * beta
                        total_length += length
                        current_voxel_sizes += 1
                        if current_tau >= tau_rand:
                            step_back = (current_tau - tau_rand) / beta
                            current_length -= step_back
                            current_point[0] = current_point[0] - step_back * direction[0]
                            current_point[1] = current_point[1] - step_back * direction[1]
                            current_point[2] = current_point[2] - step_back * direction[2]
                            in_medium = True
                            total_length += current_length
                            total_voxel_sizes += current_voxel_sizes
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
        def render_gradcontrib_cuda(beta_cloud, beta_zero, beta_air, g_cloud, w0_cloud, w0_air, sun_direction, bbox, bbox_size, voxel_size, N_cams,
                        ts, Ps, pixel_shape, is_in_medium, rng_states, scatter_inds, I_total, I_diff, grad_contrib):

            tid = cuda.grid(1)
            if tid < rng_states.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                # next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                camera_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)

                # dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                # sample entering point
                current_point[0] = bbox_size[0] * sample_uniform(rng_states, tid) + bbox[0, 0]
                current_point[1] = bbox_size[1] * sample_uniform(rng_states, tid) + bbox[1, 0]
                current_point[2] = bbox[2, 1]
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                assign_3d(direction, sun_direction)
                cloud_prob_scat = 1.0
                beta0_c = 1.0 # for type decleration
                attenuation = 1.0


                seg = 0
                cos_theta_scatter = 0.0
                cloud_prob_scat0 = 0.0


                while True:
                    seg_ind = seg + scatter_ind


                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break
                        else:
                            attenuation *= rr_factor

                    if seg > 0:
                        # angle pdf
                        attenuation *= cloud_prob_scat * HG_pdf(cos_theta_scatter, g_cloud) + (
                                    1 - cloud_prob_scat) * rayleigh_pdf(cos_theta_scatter)
                        attenuation /= cloud_prob_scat0 * HG_pdf(cos_theta_scatter, g_cloud) + (
                                    1 - cloud_prob_scat0) * rayleigh_pdf(cos_theta_scatter)
                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    current_tau0 = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0_c = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0 = beta0_c + beta_air
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)

                        current_tau += length * beta
                        current_tau0 += length * beta0
                        if current_tau0 >= tau_rand:
                            step_back = (current_tau0 - tau_rand) / beta0
                            current_tau -= step_back * beta
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

                    cloud_prob_scat0 = (w0_cloud * beta0_c) / (w0_cloud * beta0_c + w0_air * beta_air)
                    cloud_prob_scat = (w0_cloud * beta_c) / (w0_cloud * beta_c + w0_air * beta_air)
                    attenuation *= (beta / beta0) * math.exp(tau_rand - current_tau)
                    cloud_prob = 1 - (beta_air / beta)
                    attenuation *= cloud_prob * w0_cloud + (1 - cloud_prob) * w0_air

                    # LOCAL ESTIMATION TO CAMERAS
                    grad_contrib[seg_ind] = 0
                    for k in range(N_cams):
                        project_point(current_point, Ps[k], pixel_shape, pixel)
                        if pixel[0] != 255:
                            assign_3d(camera_voxel, current_voxel)
                            assign_3d(camera_point, current_point)
                            distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                            cos_theta = dot_3d(direction, cam_direction)
                            le_pdf = cloud_prob_scat * HG_pdf(cos_theta, g_cloud) \
                                     + (1-cloud_prob_scat) * rayleigh_pdf(cos_theta)
                            pc = (1 / (distance_to_camera * distance_to_camera)) * le_pdf * attenuation
                            if is_in_medium[k]:
                                assign_3d(next_point, ts[k])
                            else:
                                get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)

                            get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)
                            ###########################################################################
                            ######################## local estimation save ############################
                            tau = 0
                            for pi in range(path_size):
                                ##### border traversal #####
                                border_x = (camera_voxel[0] + (cam_direction[0] > 0)) * voxel_size[0]
                                border_y = (camera_voxel[1] + (cam_direction[1] > 0)) * voxel_size[1]
                                border_z = (camera_voxel[2] + (cam_direction[2] > 0)) * voxel_size[2]
                                t_x = (border_x - camera_point[0]) / cam_direction[0]
                                t_y = (border_y - camera_point[1]) / cam_direction[1]
                                t_z = (border_z - camera_point[2]) / cam_direction[2]
                                length, border_ind = argmin(t_x, t_y, t_z)
                                ###############################
                                # length, border, border_ind = travel_to_voxels_border_fast(camera_point, camera_voxel, cam_direction,
                                #                                  voxel_size)
                                beta_cam = beta_cloud[camera_voxel[0],camera_voxel[1], camera_voxel[2]] + beta_air
                                tau += beta_cam * length
                                # assign_3d(camera_voxel, next_voxel)
                                step_in_direction(camera_point, cam_direction, length)
                                # camera_point[border_ind] = border
                                camera_voxel[border_ind] += sign(cam_direction[border_ind])
                            # Last Step
                            length = calc_distance(camera_point, next_point)
                            beta_cam = beta_cloud[camera_voxel[0], camera_voxel[1], camera_voxel[2]] + beta_air
                            tau += beta_cam * length
                            ######################## local estimation save ############################
                            ###########################################################################
                            pc *= math.exp(-tau)
                            # path_contrib[k,seg_ind] = pc
                            grad_contrib[seg_ind] += pc * I_diff[k,pixel[0],pixel[1]]
                            # if seg != 0:
                            cuda.atomic.add(I_total, (k, pixel[0], pixel[1]), pc)


                    # New Direction
                    if sample_uniform(rng_states, tid) <= cloud_prob_scat0:
                        HG_sample_direction(direction, g_cloud, cam_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, cam_direction, rng_states, tid)

                    cos_theta_scatter = dot_3d(cam_direction, direction)
                    assign_3d(direction, cam_direction)
                    seg += 1
                # if seg!= N_seg:
                # print("reg", seg, N_seg)

                # gradient preprocess
                grad_sum = 0
                for seg in range(N_seg):
                    seg_ind = N_seg-1-seg + scatter_ind
                    grad_sum += grad_contrib[seg_ind]
                    grad_contrib[seg_ind] = grad_sum

        @cuda.jit()
        def render_cuda(beta_cloud, beta_zero, beta_air, g_cloud, w0_cloud, w0_air, sun_direction, bbox, bbox_size,
                        voxel_size, N_cams,
                        ts, Ps, pixel_shape, is_in_medium, rng_states, scatter_inds, I_total, grad_contrib):

            tid = cuda.grid(1)
            if tid < rng_states.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                # next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                camera_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)

                # dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                # sample entering point
                current_point[0] = bbox_size[0] * sample_uniform(rng_states, tid) + bbox[0, 0]
                current_point[1] = bbox_size[1] * sample_uniform(rng_states, tid) + bbox[1, 0]
                current_point[2] = bbox[2, 1]
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                assign_3d(direction, sun_direction)
                cloud_prob_scat = 1.0
                beta0_c = 1.0  # for type decleration
                attenuation = 1.0

                seg = 0
                cos_theta_scatter = 0.0
                cloud_prob_scat0 = 0.0

                while True:
                    seg_ind = seg + scatter_ind

                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break
                        else:
                            attenuation *= rr_factor

                    if seg > 0:
                        # angle pdf
                        attenuation *= cloud_prob_scat * HG_pdf(cos_theta_scatter, g_cloud) + (
                                1 - cloud_prob_scat) * rayleigh_pdf(cos_theta_scatter)
                        attenuation /= cloud_prob_scat0 * HG_pdf(cos_theta_scatter, g_cloud) + (
                                1 - cloud_prob_scat0) * rayleigh_pdf(cos_theta_scatter)
                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    current_tau0 = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0_c = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0 = beta0_c + beta_air
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)

                        current_tau += length * beta
                        current_tau0 += length * beta0
                        if current_tau0 >= tau_rand:
                            step_back = (current_tau0 - tau_rand) / beta0
                            current_tau -= step_back * beta
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

                    cloud_prob_scat0 = (w0_cloud * beta0_c) / (w0_cloud * beta0_c + w0_air * beta_air)
                    cloud_prob_scat = (w0_cloud * beta_c) / (w0_cloud * beta_c + w0_air * beta_air)
                    attenuation *= (beta / beta0) * math.exp(tau_rand - current_tau)
                    cloud_prob = 1 - (beta_air / beta)
                    attenuation *= cloud_prob * w0_cloud + (1 - cloud_prob) * w0_air

                    # LOCAL ESTIMATION TO CAMERAS
                    # grad_contrib[seg_ind] = 0
                    grad_contrib[seg_ind] = attenuation
                    for k in range(N_cams):
                        project_point(current_point, Ps[k], pixel_shape, pixel)
                        if pixel[0] != 255:
                            assign_3d(camera_voxel, current_voxel)
                            assign_3d(camera_point, current_point)
                            distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                            cos_theta = dot_3d(direction, cam_direction)
                            le_pdf = cloud_prob_scat * HG_pdf(cos_theta, g_cloud) \
                                     + (1 - cloud_prob_scat) * rayleigh_pdf(cos_theta)
                            pc = (1 / (distance_to_camera * distance_to_camera)) * le_pdf * attenuation
                            if is_in_medium[k]:
                                assign_3d(next_point, ts[k])
                            else:
                                get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)

                            get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)
                            ###########################################################################
                            ######################## local estimation save ############################
                            tau = 0
                            for pi in range(path_size):
                                ##### border traversal #####
                                border_x = (camera_voxel[0] + (cam_direction[0] > 0)) * voxel_size[0]
                                border_y = (camera_voxel[1] + (cam_direction[1] > 0)) * voxel_size[1]
                                border_z = (camera_voxel[2] + (cam_direction[2] > 0)) * voxel_size[2]
                                t_x = (border_x - camera_point[0]) / cam_direction[0]
                                t_y = (border_y - camera_point[1]) / cam_direction[1]
                                t_z = (border_z - camera_point[2]) / cam_direction[2]
                                length, border_ind = argmin(t_x, t_y, t_z)
                                ###############################
                                # length, border, border_ind = travel_to_voxels_border_fast(camera_point, camera_voxel, cam_direction,
                                #                                  voxel_size)
                                beta_cam = beta_cloud[camera_voxel[0], camera_voxel[1], camera_voxel[2]] + beta_air
                                tau += beta_cam * length
                                # assign_3d(camera_voxel, next_voxel)
                                step_in_direction(camera_point, cam_direction, length)
                                # camera_point[border_ind] = border
                                camera_voxel[border_ind] += sign(cam_direction[border_ind])
                            # Last Step
                            length = calc_distance(camera_point, next_point)
                            beta_cam = beta_cloud[camera_voxel[0], camera_voxel[1], camera_voxel[2]] + beta_air
                            tau += beta_cam * length
                            ######################## local estimation save ############################
                            ###########################################################################
                            pc *= math.exp(-tau)
                            # path_contrib[k,seg_ind] = pc

                            # if seg != 0:
                            cuda.atomic.add(I_total, (k, pixel[0], pixel[1]), pc)

                    # New Direction
                    if sample_uniform(rng_states, tid) <= cloud_prob_scat0:
                        HG_sample_direction(direction, g_cloud, cam_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, cam_direction, rng_states, tid)

                    cos_theta_scatter = dot_3d(cam_direction, direction)
                    assign_3d(direction, cam_direction)
                    seg += 1

        @cuda.jit()
        def gradcontrib_cuda(beta_cloud, beta_zero, beta_air, g_cloud,  w0_cloud, w0_air, sun_direction, bbox,
                                    bbox_size, voxel_size, N_cams,
                                    ts, Ps, pixel_shape, is_in_medium, rng_states, scatter_inds, I_diff,
                                    grad_contrib):
            tid = cuda.grid(1)
            if tid < rng_states.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                # next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                camera_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)

                # dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                # sample entering point
                current_point[0] = bbox_size[0] * sample_uniform(rng_states, tid) + bbox[0, 0]
                current_point[1] = bbox_size[1] * sample_uniform(rng_states, tid) + bbox[1, 0]
                current_point[2] = bbox[2, 1]
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                assign_3d(direction, sun_direction)
                seg = 0
                while True:
                    seg_ind = seg + scatter_ind

                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break

                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    current_tau0 = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0_c = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0 = beta0_c + beta_air
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)

                        current_tau += length * beta
                        current_tau0 += length * beta0
                        if current_tau0 >= tau_rand:
                            step_back = (current_tau0 - tau_rand) / beta0
                            current_tau -= step_back * beta
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


                    # LOCAL ESTIMATION TO CAMERAS

                    attenuation = grad_contrib[seg_ind]
                    grad_contrib[seg_ind] = 0
                    cloud_prob_scat0 = (w0_cloud * beta0_c) / (w0_cloud * beta0_c + w0_air * beta_air)
                    cloud_prob_scat = (w0_cloud * beta_c) / (w0_cloud * beta_c + w0_air * beta_air)
                    for k in range(N_cams):
                        project_point(current_point, Ps[k], pixel_shape, pixel)
                        if pixel[0] != 255:
                            assign_3d(camera_voxel, current_voxel)
                            assign_3d(camera_point, current_point)
                            distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                            cos_theta = dot_3d(direction, cam_direction)
                            le_pdf = cloud_prob_scat * HG_pdf(cos_theta, g_cloud) \
                                     + (1 - cloud_prob_scat) * rayleigh_pdf(cos_theta)
                            pc = (1 / (distance_to_camera * distance_to_camera)) * le_pdf * attenuation
                            if is_in_medium[k]:
                                assign_3d(next_point, ts[k])
                            else:
                                get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)

                            get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)
                            ###########################################################################
                            ######################## local estimation save ############################
                            tau = 0
                            for pi in range(path_size):
                                ##### border traversal #####
                                border_x = (camera_voxel[0] + (cam_direction[0] > 0)) * voxel_size[0]
                                border_y = (camera_voxel[1] + (cam_direction[1] > 0)) * voxel_size[1]
                                border_z = (camera_voxel[2] + (cam_direction[2] > 0)) * voxel_size[2]
                                t_x = (border_x - camera_point[0]) / cam_direction[0]
                                t_y = (border_y - camera_point[1]) / cam_direction[1]
                                t_z = (border_z - camera_point[2]) / cam_direction[2]
                                length, border_ind = argmin(t_x, t_y, t_z)
                                ###############################
                                # length, border, border_ind = travel_to_voxels_border_fast(camera_point, camera_voxel, cam_direction,
                                #                                  voxel_size)
                                beta_cam = beta_cloud[camera_voxel[0], camera_voxel[1], camera_voxel[2]] + beta_air
                                tau += beta_cam * length
                                # assign_3d(camera_voxel, next_voxel)
                                step_in_direction(camera_point, cam_direction, length)
                                # camera_point[border_ind] = border
                                camera_voxel[border_ind] += sign(cam_direction[border_ind])
                            # Last Step
                            length = calc_distance(camera_point, next_point)
                            beta_cam = beta_cloud[camera_voxel[0], camera_voxel[1], camera_voxel[2]] + beta_air
                            tau += beta_cam * length
                            ######################## local estimation save ############################
                            ###########################################################################
                            pc *= math.exp(-tau)
                            # path_contrib[k,seg_ind] = pc
                            grad_contrib[seg_ind] += pc * I_diff[k, pixel[0], pixel[1]]
                            # if seg != 0:

                    # New Direction
                    if sample_uniform(rng_states, tid) <= cloud_prob_scat0:
                        HG_sample_direction(direction, g_cloud, cam_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, cam_direction, rng_states, tid)

                    cos_theta_scatter = dot_3d(cam_direction, direction)
                    assign_3d(direction, cam_direction)
                    seg += 1


                # gradient preprocess
                grad_sum = 0
                for seg in range(N_seg):
                    seg_ind = N_seg - 1 - seg + scatter_ind
                    grad_sum += grad_contrib[seg_ind]
                    grad_contrib[seg_ind] = grad_sum


        @cuda.jit()
        def render_differentiable_cuda(beta_cloud, beta_zero, beta_air, g_cloud, w0_cloud, w0_air, sun_direction, bbox, bbox_size, voxel_size,
                         rng_states, scatter_inds ,grad_contrib, cloud_mask, total_grad, N_batches):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind

                grid_shape = beta_cloud.shape

                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                next_direction = cuda.local.array(3, dtype=float_precis)
                current_point[0] = bbox_size[0] * sample_uniform(rng_states, tid) + bbox[0, 0]
                current_point[1] = bbox_size[1] * sample_uniform(rng_states, tid) + bbox[1, 0]
                current_point[2] = bbox[2, 1]

                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                assign_3d(direction, sun_direction)

                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                beta_c = 1.0  # for type decleration
                seg = 0
                cos_theta_scatter = 0.0
                mask = False
                grad_batch = tid % N_batches
                while True:
                    seg_ind = seg + scatter_ind
                    # Russian Roulette
                    if seg >= rr_depth:
                        if sample_uniform(rng_states, tid) <= rr_stop_prob:
                            break


                    grad_temp = grad_contrib[seg_ind]
                    if seg > 0 and seg < N_seg and mask:

                        omega_fp_c = w0_cloud * HG_pdf(cos_theta_scatter, g_cloud)
                        seg_contrib = omega_fp_c / \
                                      (beta_c * omega_fp_c + beta_air * w0_air * rayleigh_pdf(
                                          cos_theta_scatter))  # scatter fix
                        cuda.atomic.add(total_grad, (grad_batch,current_voxel[0], current_voxel[1], current_voxel[2]),
                                        seg_contrib * grad_temp)

                    tau_rand = -math.log(1 - sample_uniform(rng_states, tid))
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    current_tau0 = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0_c = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0 = beta0_c + beta_air
                        beta = beta_c + beta_air
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)
                        current_tau += length * beta
                        current_tau0 += length * beta0
                        mask = cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]
                        if current_tau0 < tau_rand:
                            if mask and seg < N_seg:
                                cuda.atomic.add(total_grad, (grad_batch,current_voxel[0], current_voxel[1], current_voxel[2]), -length * grad_temp)

                        else:
                            step_back = (current_tau0 - tau_rand) / beta0

                            if mask:
                                cuda.atomic.add(total_grad, (grad_batch, current_voxel[0], current_voxel[1], current_voxel[2]),
                                                -(length-step_back) * grad_temp)
                            beta_c += divide_beta_eps
                            current_tau -= step_back * beta
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

                    cloud_prob_scat0 = (w0_cloud * beta0_c) / (w0_cloud * beta0_c + w0_air * beta_air)
                    # New Direction
                    if sample_uniform(rng_states, tid) <= cloud_prob_scat0:
                        HG_sample_direction(direction, g_cloud, next_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, next_direction, rng_states, tid)

                    cos_theta_scatter = dot_3d(next_direction, direction)
                    assign_3d(direction, next_direction)
                    seg += 1



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
        # self.generate_paths = generate_paths
        # self.sort_scatter_points = sort_scatter_points
        self.render_gradconrtib_cuda = render_gradcontrib_cuda
        self.render_cuda = render_cuda
        self.gradcontrib_cuda = gradcontrib_cuda
        # self.calc_gradient_contribution = calc_gradient_contribution
        self.render_differentiable_cuda = render_differentiable_cuda
        self.space_curving_cuda = space_curving_cuda

    def init_cuda_param(self, Np, init=False, seed=None):
        self.threadsperblock = 256
        self.blockspergrid = (Np + (self.threadsperblock - 1)) // self.threadsperblock
        if init:
            if seed is None:
                self.seed = np.random.randint(1, int(1e9))
            else:
                self.seed = seed
            self.rng_states = np.empty(shape=Np, dtype=xoroshiro128p_dtype)
            init_xoroshiro128p_states_cpu(self.rng_states, self.seed, 0)
            self.init = True
            self.rng_states_updated = None


    def build_paths_list(self, Np, init=False, to_sort=True, to_print=False):
        assert self.init, "cuda rng was not initialized"
        if self.rng_states_updated is not None:
            self.rng_states = self.rng_states_updated
        self.Np = Np
        self.init_cuda_param(Np, init=False)
        # inputs
        # del(self.dpath_contrib)
        del(self.dgrad_contrib)
        del(self.dscatter_inds)
        del(self.drng_states)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.dbeta_zero.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        # outputs
        dscatter_sizes = cuda.to_device(np.empty(Np, dtype=np.uint8))
        drng_states = cuda.to_device(self.rng_states)
        start = time()
        self.calc_scatter_sizes[blockspergrid, threadsperblock]\
            (Np, self.dbeta_zero, beta_air, self.g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size,
             self.dvoxel_size,self.dsun_direction, dscatter_sizes, drng_states)

        cuda.synchronize()
        if to_print:
            print("calc scatter sizes took:",time()-start)
        scatter_sizes = dscatter_sizes.copy_to_host()
        self.rng_states_updated = drng_states.copy_to_host()
        del (dscatter_sizes)
        del(drng_states)
        cond = scatter_sizes != 0
        Np_nonan = np.sum(cond)

        scatter_sizes = scatter_sizes[cond]
        self.rng_states_mod = self.rng_states[cond]
        if to_sort:
            start = time()
            sorted_inds = np.argsort(scatter_sizes)
            print("lexsort took:", time()-start)
            # sorted_inds = np.lexsort([scatter_sizes,lengths])
            scatter_sizes = scatter_sizes[sorted_inds]
            self.rng_states_mod = self.rng_states_mod[sorted_inds]

        scatter_inds = np.concatenate([np.array([0]), scatter_sizes])
        scatter_inds = np.cumsum(scatter_inds)
        total_scatter_num = scatter_inds[-1]
        self.dscatter_inds = cuda.to_device(scatter_inds)



        self.total_num_of_scatter = total_scatter_num
        self.Np_nonan = Np_nonan
        # self.dpath_contrib = cuda.to_device(np.empty((self.N_cams, self.total_num_of_scatter), dtype=float_reg))
        self.dgrad_contrib = cuda.to_device(np.empty(self.total_num_of_scatter, dtype=float_reg))

        self.drng_states = cuda.device_array(self.rng_states_mod.shape, self.rng_states.dtype)
        self.init_cuda_param(self.Np_nonan, init=False)


    def render(self, I_gt=None, to_print=False):
        assert self.Np % self.N_batches==0, "N_batches must divide Np"
        # east declerations
        N_cams = len(self.cameras)
        pixels_shape = self.cameras[0].pixels
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        self.dI_total.copy_to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=float_reg))
        start = time()

        self.drng_states.copy_to_device(self.rng_states_mod)
        self.render_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dsun_direction, self.dbbox, self.dbbox_size,
             self.dvoxel_size, N_cams, self.dts, self.dPs, self.dpixels_shape, self.dis_in_medium, self.drng_states, self.dscatter_inds, self.dI_total,
            self.dgrad_contrib)
        cuda.synchronize()

        I_total = self.dI_total.copy_to_host()
        I_total /= self.Np
        if I_gt is None:
            if to_print:
                print("render_cuda took (once):", time() - start)
            # del dpath_contrib
            return I_total

        self.dI_total.copy_to_device(I_total - I_gt)
        self.drng_states.copy_to_device(self.rng_states_mod)
        self.gradcontrib_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dsun_direction, self.dbbox,
             self.dbbox_size,
             self.dvoxel_size, N_cams, self.dts, self.dPs, self.dpixels_shape, self.dis_in_medium, self.drng_states,
             self.dscatter_inds, self.dI_total, self.dgrad_contrib)
        cuda.synchronize()
        # print("voxel counter",dcounter.copy_to_host())

        if to_print:
            print("render_cuda took (twice):",time() - start)


        ##### differentiable part ####
        self.dtotal_grad.copy_to_device(np.zeros((self.N_batches, *self.volume.beta_cloud.shape),dtype=float_reg))
        # precalculating gradient contributions
        start = time()
        self.drng_states.copy_to_device(self.rng_states_mod)
        self.render_differentiable_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dsun_direction, self.dbbox,
             self.dbbox_size, self.dvoxel_size,
             self.drng_states, self.dscatter_inds, self.dgrad_contrib, self.dcloud_mask, self.dtotal_grad, self.N_batches)

        cuda.synchronize()
        if to_print:
            print("render_differentiable_cuda took:", time() - start)
        total_grad = self.dtotal_grad.copy_to_host()

        MC_frac = self.N_batches/(self.Np * N_cams)
        total_grad /= MC_frac

        return I_total, total_grad

    def render_given_I_diff(self, I_diff=None, to_print=False):
        # east declerations
        N_cams = len(self.cameras)
        pixels_shape = self.cameras[0].pixels
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        self.dI_total.copy_to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=float_reg))
        if I_diff is None:
            self.dI_diff.copy_to_device(np.zeros((len(self.cameras), *self.cameras[0].pixels),dtype=float_reg))
        else:
            self.dI_diff.copy_to_device(I_diff)
        start = time()

        self.drng_states.copy_to_device(self.rng_states_mod)
        self.render_gradconrtib_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dsun_direction, self.dbbox, self.dbbox_size,
             self.dvoxel_size, N_cams, self.dts, self.dPs, self.dpixels_shape, self.dis_in_medium, self.drng_states, self.dscatter_inds, self.dI_total,
             self.dI_diff, self.dgrad_contrib)
        cuda.synchronize()
        # print("voxel counter",dcounter.copy_to_host())
        I_total = self.dI_total.copy_to_host()
        I_total /= self.Np
        if to_print:
            print("render_cuda took:",time() - start)
        # return I_total
        if I_diff is None:
            # del dpath_contrib
            return I_total

        ##### differentiable part ####
        self.dtotal_grad.copy_to_device(np.zeros_like(self.volume.beta_cloud, dtype=float_reg))
        # precalculating gradient contributions
        start = time()
        self.drng_states.copy_to_device(self.rng_states_mod)
        self.render_differentiable_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dsun_direction, self.dbbox,
             self.dbbox_size, self.dvoxel_size,
             self.drng_states, self.dscatter_inds, self.dgrad_contrib, self.dcloud_mask, self.dtotal_grad)

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
        self.init_cuda_param(pixel_mat.shape[1], init=True)
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




