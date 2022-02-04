from classes.volume import *
from utils import theta_phi_to_direction
from numba.cuda.random import create_xoroshiro128p_states
from cuda_utils import *
from time import time
from scipy.ndimage import binary_dilation

threadsperblock = 256


class SceneDDIS(object):
    def __init__(self, volume: Volume, cameras, sun_angles, g_cloud, Ns):
        self.Ns = Ns
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
        self.dtotal_grad = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dbbox = cuda.to_device(self.volume.grid.bbox)
        self.dbbox_size = cuda.to_device(self.volume.grid.bbox_size)
        self.dvoxel_size = cuda.to_device(self.volume.grid.voxel_size)
        self.dsun_direction = cuda.to_device(self.sun_direction)
        self.dpixels_shape = cuda.to_device(self.cameras[0].pixels)
        self.dts = cuda.to_device(ts)
        self.dPs = cuda.to_device(self.Ps)
        self.dis_in_medium = cuda.to_device(self.is_camera_in_medium)
        self.dcloud_mask = cuda.to_device(self.volume.cloud_mask)
        self.dpath_contrib = None
        self.dgrad_contrib = None

        @cuda.jit()
        def scatter_cound(Np, Ns, beta_cloud, beta_air, g_cloud, bbox, bbox_size, voxel_size, sun_direction, starting_points,
                           scatter_points, scatter_sizes, rng_states, ticket):
            pass
        @cuda.jit()
        def generate_paths(Np, Ns, beta_cloud, beta_air, g_cloud, bbox, bbox_size, voxel_size, sun_direction, starting_points,
                           scatter_points, scatter_sizes, rng_states, ticket):

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
                p = sample_uniform(rng_states, tid)
                starting_point[0] = bbox_size[0] * p + bbox[0, 0]

                p = sample_uniform(rng_states, tid)
                starting_point[1] = bbox_size[1] * p + bbox[1, 0]

                starting_point[2] = bbox[2, 1]
                assign_3d(current_point, starting_point)
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                for seg in range(Ns):

                    p = sample_uniform(rng_states, tid)
                    tau_rand = -math.log(1 - p)
                    ###########################################################
                    ############## voxel_traversal_algorithm_save #############
                    current_tau = 0.0
                    while True:
                        if not is_voxel_valid(current_voxel, grid_shape):
                            in_medium = False
                            break
                        beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
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
                    if seg == 0:
                        ind = cuda.atomic.add(ticket,0,1)
                    scatter_points[0, seg, ind] = current_point[0]
                    scatter_points[1, seg, ind] = current_point[1]
                    scatter_points[2, seg, ind] = current_point[2]

                    # sampling new direction
                    cloud_prob = (beta - beta_air) / beta
                    p = sample_uniform(rng_states, tid)
                    if p <= cloud_prob:
                        HG_sample_direction(direction, g_cloud, new_direction, rng_states, tid)
                    else:
                        rayleigh_sample_direction(direction, new_direction, rng_states, tid)
                    assign_3d(direction, new_direction)

                # voxels and scatter sizes for this path (this is not in a loop)
                N_seg = seg + int(in_medium)
                if N_seg != 0:
                    assign_3d(starting_points[:,ind], starting_point)
                    scatter_sizes[ind] = N_seg

        @cuda.jit()
        def post_generation(scatter_points, scatter_inds, scatter_points_zipped):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:

                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_ind
                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    assign_3d(scatter_points_zipped[:, seg_ind], scatter_points[:,seg,tid])
                    # for k in range(N_cams):
                    #     project_point(scatter_points[:,seg,tid], Ps[k], pixel_shape, pixel_mat[:,k,seg_ind])

        @cuda.jit()
        def ddis_generation(beta_cloud, beta_air, g_cloud, bbox, bbox_size, voxel_size, N_cams, ts, Ps, pixel_shape,
                            scatter_inds, scatter_points, starting_points, ddis_starting_points, ddis_points, rng_states):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                pixel = cuda.local.array(2, dtype=np.uint8)
                starting_point = cuda.local.array(3, dtype=float_precis)
                current_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)

                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                for seg in range(-1, N_seg):
                    seg_ind = seg + scatter_ind
                    for k in range(N_cams):
                        project_point(current_point, Ps[k], pixel_shape, pixel)
                        if pixel[0] != 255:
                            p = sample_uniform(rng_states, tid)
                            if p <= e_ddis:
                                distance_to_camera = distance_and_direction(current_point, ts[k], direction)
                            else:
                                assign_3d(next_point, scatter_points[:, seg_ind])
                                distance_and_direction(current_point, next_point, direction)
                            HG_sample_direction(direction, g_cloud, new_direction, rng_states, tid)
                            p = sample_uniform(rng_states, tid)
                            tau_rand = -math.log(1 - p)
                            ###########################################################
                            ############## voxel_traversal_algorithm_save #############
                            current_tau = 0.0
                            while True:
                                if not is_voxel_valid(current_voxel, grid_shape):
                                    in_medium = False
                                    break
                                beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                                length = travel_to_voxels_border(current_point, current_voxel, new_direction, voxel_size,
                                                                 next_voxel)
                                current_tau += length * beta
                                if current_tau >= tau_rand:
                                    step_back = (current_tau - tau_rand) / beta
                                    current_point[0] = current_point[0] - step_back * new_direction[0]
                                    current_point[1] = current_point[1] - step_back * new_direction[1]
                                    current_point[2] = current_point[2] - step_back * new_direction[2]
                                    in_medium = True
                                    break
                                assign_3d(current_voxel, next_voxel)
                            ######################## voxel_traversal_algorithm_save ###################
                            ###########################################################################
                            if in_medium:
                                if seg == -1:
                                    assign_3d(ddis_starting_points[:,k,tid], current_point)
                                else:
                                    assign_3d(ddis_points[:, k, seg_ind], current_point)









        @cuda.jit()
        def render_cuda(beta_cloud, beta_zero, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, N_cams,
                        ts, Ps, pixel_shape, is_in_medium, starting_points, scatter_points, scatter_inds,I_total, path_contrib, rng_states):

            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                camera_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=float_precis)
                previous_direction = cuda.local.array(3, dtype=float_precis)
                temp_direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)
                ddis_direction = cuda.local.array(3, dtype=float_precis)

                # dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                beta = 1.0 # for type decleration
                cloud_prob = 1.0
                beta0 = 1.0 # for type decleration
                IS = 1.0
                attenuation = 1.0
                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind

                    # Propagating to new mother point
                    assign_3d(next_point, scatter_points[:, seg_ind])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    distance_and_direction(current_point, next_point, direction)
                    if seg > 0:
                        cos_theta = dot_3d(previous_direction, direction)
                        cloud_prob0 = (beta0 - beta_air) / beta0
                        # angle pdf
                        IS *= cloud_prob * HG_pdf(cos_theta, g_cloud) + (1 - cloud_prob) * rayleigh_pdf(cos_theta)
                        IS /= cloud_prob0 * HG_pdf(cos_theta, g_cloud) + (1 - cloud_prob0) * rayleigh_pdf(cos_theta)
                    ###########################################################
                    ############## voxel_fixed traversal_algorithm_save #############
                    path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    tau = 0
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
                        beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                        beta0 = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                        tau += (beta - beta0) * length
                        # assign_3d(current_voxel, next_voxel)
                        current_voxel[border_ind] += sign(direction[border_ind])
                        step_in_direction(current_point, direction, length)
                    # last step
                    length = calc_distance(current_point, next_point)
                    beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                    beta0 = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                    tau += (beta - beta0) * length
                    assign_3d(current_point, next_point)
                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################
                    IS *= ((beta / beta0) * math.exp(-tau))  # length pdf
                    # beta_c = beta - beta_air
                    cloud_prob = 1 - (beta_air / beta)
                    attenuation *= cloud_prob * w0_cloud + (1 - cloud_prob) * w0_air
                    assign_3d(previous_direction, direction)  # using cam direction as temp direction
                    ###################### DDIS LOCAL ESTIMATION #########################
                    for k in range(N_cams):
                        assign_3d(camera_point, current_point)
                        assign_3d(camera_voxel, current_voxel)
                        distance_and_direction(camera_point, ts[k], cam_direction)
                        if seg == 0:
                            project_point(camera_point, Ps[k], pixel_shape, pixel)
                            if pixel[0] != 255:
                                distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                                cos_theta = dot_3d(direction, cam_direction)
                                le_pdf = cloud_prob*HG_pdf(cos_theta, g_cloud) + (1 - cloud_prob)*rayleigh_pdf(cos_theta)
                                pc = (1 / (distance_to_camera ** 2)) * IS * le_pdf * attenuation
                                if is_in_medium[k]:
                                    assign_3d(next_point, ts[k])
                                else:
                                    get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)

                                get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                                path_size = estimate_voxels_size(dest_voxel, camera_voxel)
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
                                path_contrib[k, seg_ind] = pc
                                cuda.atomic.add(I_total, (k, pixel[0], pixel[1]), pc)
                                assign_3d(camera_point, current_point)
                                assign_3d(camera_voxel, current_voxel)
                                # cuda.atomic.add(I_scatter, 0, pc)

                        p = sample_uniform(rng_states, tid)
                        if p <= e_ddis:
                            assign_3d(temp_direction, cam_direction)
                        else:
                            assign_3d(temp_direction, direction)

                        p = sample_uniform(rng_states, tid)
                        if p <= cloud_prob:
                            HG_sample_direction(temp_direction, g_cloud, ddis_direction, rng_states, tid)
                        else:
                            rayleigh_sample_direction(temp_direction, ddis_direction, rng_states, tid)
                        cos_theta_regular = dot_3d(ddis_direction, direction)
                        cos_theta_ddis = dot_3d(ddis_direction, cam_direction)
                        true_phase_pdf = cloud_prob*HG_pdf(cos_theta_regular, g_cloud) + (1-cloud_prob)*rayleigh_pdf(cos_theta_regular)
                        ddis_phase_pdf = cloud_prob*HG_pdf(cos_theta_ddis, g_cloud) + (1-cloud_prob)*rayleigh_pdf(cos_theta_ddis)
                        IS_cam = IS * (true_phase_pdf/(e_ddis*ddis_phase_pdf + (1-e_ddis)*true_phase_pdf))
                        p = sample_uniform(rng_states, tid)
                        tau_rand = -math.log(1 - p)
                        ###########################################################
                        ############## voxel_traversal_algorithm_save #############
                        current_tau = 0.0
                        while True:
                            if not is_voxel_valid(camera_voxel, grid_shape):
                                in_medium = False
                                break
                            beta_cam = beta_cloud[camera_voxel[0], camera_voxel[1], camera_voxel[2]] + beta_air
                            length = travel_to_voxels_border(camera_point, camera_voxel, ddis_direction, voxel_size,
                                                             next_voxel)
                            current_tau += length * beta_cam
                            if current_tau >= tau_rand:
                                step_back = (current_tau - tau_rand) / beta_cam
                                camera_point[0] = camera_point[0] - step_back * ddis_direction[0]
                                camera_point[1] = camera_point[1] - step_back * ddis_direction[1]
                                camera_point[2] = camera_point[2] - step_back * ddis_direction[2]
                                in_medium = True
                                break
                            assign_3d(camera_voxel, next_voxel)

                        if in_medium:
                            project_point(camera_point, Ps[k], pixel_shape, pixel)
                            if pixel[0] != 255:
                                cloud_prob_cam = 1 - (beta_air / beta_cam)
                                attenuation_cam = attenuation * (
                                            cloud_prob_cam * w0_cloud + (1 - cloud_prob_cam) * w0_air)
                                distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                                cos_theta = dot_3d(ddis_direction, cam_direction)
                                le_pdf = cloud_prob_cam * HG_pdf(cos_theta, g_cloud) + (1-cloud_prob_cam) * rayleigh_pdf(cos_theta)
                                pc = (1 / (distance_to_camera ** 2)) * IS_cam * le_pdf * attenuation_cam
                                if is_in_medium[k]:
                                    assign_3d(next_point, ts[k])
                                else:
                                    get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)

                                get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                                path_size = estimate_voxels_size(dest_voxel, camera_voxel)
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
                                path_contrib[k,seg_ind] += pc
                                cuda.atomic.add(I_total, (k, pixel[0], pixel[1]), pc)
                                # cuda.atomic.add(I_scatter, seg+1, pc)
                        ###################### DDIS LOCAL ESTIMATION #########################



        @cuda.jit()
        def calc_gradient_contribution(scatter_points, path_contrib, I_diff, Ps, pixel_shape, scatter_inds, grad_contrib):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                pixel = cuda.local.array(2, dtype=np.uint8)
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    grad_contrib[seg_ind] = 0
                    for sub_seg in range(seg, N_seg):
                        sub_seg_ind = sub_seg + scatter_ind
                        for k in range(path_contrib.shape[0]):
                            # assign_3d(current_point, scatter_points[:,sub_seg_ind])
                            project_point(scatter_points[:,sub_seg_ind], Ps[k], pixel_shape, pixel)
                            if pixel[0] != 255:
                                grad_contrib[seg_ind] += path_contrib[k, sub_seg_ind] * I_diff[k,pixel[0],pixel[1]]


        @cuda.jit()
        def render_differentiable_cuda(beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, N_cams,
                        ts, Ps, pixel_shape, is_in_medium, starting_points, scatter_points, scatter_inds,
                                       I_diff, path_contrib, grad_contrib, cloud_mask, total_grad):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind

                grid_shape = beta_cloud.shape

                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                camera_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                beta_c = 1.0  # for type decleration
                le_contrib = 1.0  # for type decleration

                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    assign_3d(next_point, scatter_points[:, seg_ind])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    distance_and_direction(current_point, next_point, direction)
                    # GRAD CALCULATION (SCATTERING)
                    grad_temp = grad_contrib[seg_ind]
                    if seg > 0:
                        if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                            cos_theta_scatter = dot_3d(cam_direction, direction)
                            seg_contrib = (beta_c + (rayleigh_pdf(cos_theta_scatter)/HG_pdf(cos_theta_scatter,g_cloud)) * beta_air ) **(-1)
                            seg_contrib += le_contrib
                            grad = seg_contrib * grad_temp
                            cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]), grad)
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
                        grad = -length * grad_temp
                        cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]), grad)
                    assign_3d(current_point, next_point)

                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################
                    beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + divide_beta_eps
                    le_contrib = (beta_c + (w0_air/w0_cloud) * beta_air) **(-1)
                    le_contrib -= 1 / (beta_c + beta_air)
                    for k in range(N_cams):
                        project_point(current_point, Ps[k], pixel_shape, pixel)
                        if pixel[0] != 255:
                            assign_3d(camera_voxel, current_voxel)
                            assign_3d(camera_point, current_point)
                            distance_and_direction(camera_point, ts[k], cam_direction)
                            cos_theta = dot_3d(direction, cam_direction)

                            if is_in_medium[k]:
                                assign_3d(next_point, ts[k])
                            else:
                                get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)
                            get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)

                            grad_temp = path_contrib[k, seg_ind] * I_diff[k, pixel[0], pixel[1]]
                            # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                            if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]]:
                                seg_contrib = le_contrib + (beta_c + (rayleigh_pdf(cos_theta) / HG_pdf(cos_theta, g_cloud)) * beta_air) ** (-1)
                                grad = seg_contrib * grad_temp
                                cuda.atomic.add(total_grad, (camera_voxel[0],camera_voxel[1],camera_voxel[2]), grad)
                            ###########################################################################
                            ######################## local estimation save ############################
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
                                # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                                if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]]:
                                    grad = -length * grad_temp
                                    cuda.atomic.add(total_grad,(camera_voxel[0], camera_voxel[1], camera_voxel[2]), grad)
                                # next point and voxel
                                camera_voxel[border_ind] += sign(cam_direction[border_ind])
                                step_in_direction(camera_point, cam_direction, length)
                            # Last Step
                            length = calc_distance(camera_point, next_point)
                            # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                            if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]]:
                                grad = -length * grad_temp
                                cuda.atomic.add(total_grad, (camera_voxel[0], camera_voxel[1], camera_voxel[2]), grad)
                            ######################## local estimation save ############################
                            ###########################################################################

                    assign_3d(cam_direction, direction)  # using cam direction as temp direction

        @cuda.jit()
        def render_var(beta_cloud, beta_zero, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, N_cams,
                        ts, Ps, pixel_shape, is_in_medium, starting_points, scatter_points, scatter_inds, first_moment,
                       second_moment, rng_states):

            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid + 1] - scatter_ind
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                camera_point = cuda.local.array(3, dtype=float_precis)
                next_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=float_precis)
                previous_direction = cuda.local.array(3, dtype=float_precis)
                temp_direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)
                ddis_direction = cuda.local.array(3, dtype=float_precis)

                # dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                beta = 1.0  # for type decleration
                cloud_prob = 1.0
                beta0 = 1.0  # for type decleration
                IS = 1.0
                attenuation = 1.0
                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind

                    # Propagating to new mother point
                    assign_3d(next_point, scatter_points[:, seg_ind])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    distance_and_direction(current_point, next_point, direction)
                    if seg > 0:
                        cos_theta = dot_3d(previous_direction, direction)
                        cloud_prob0 = (beta0 - beta_air) / beta0
                        # angle pdf
                        IS *= cloud_prob * HG_pdf(cos_theta, g_cloud) + (1 - cloud_prob) * rayleigh_pdf(cos_theta)
                        IS /= cloud_prob0 * HG_pdf(cos_theta, g_cloud) + (1 - cloud_prob0) * rayleigh_pdf(cos_theta)
                    ###########################################################
                    ############## voxel_fixed traversal_algorithm_save #############
                    path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    tau = 0
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
                        beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                        beta0 = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                        tau += (beta - beta0) * length
                        # assign_3d(current_voxel, next_voxel)
                        current_voxel[border_ind] += sign(direction[border_ind])
                        step_in_direction(current_point, direction, length)
                    # last step
                    length = calc_distance(current_point, next_point)
                    beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                    beta0 = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                    tau += (beta - beta0) * length
                    assign_3d(current_point, next_point)
                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################
                    IS *= ((beta / beta0) * math.exp(-tau))  # length pdf
                    # beta_c = beta - beta_air
                    cloud_prob = 1 - (beta_air / beta)
                    attenuation *= cloud_prob * w0_cloud + (1 - cloud_prob) * w0_air
                    assign_3d(previous_direction, direction)  # using cam direction as temp direction
                    ###################### DDIS LOCAL ESTIMATION #########################
                    for k in range(N_cams):
                        assign_3d(camera_point, current_point)
                        assign_3d(camera_voxel, current_voxel)
                        distance_and_direction(camera_point, ts[k], cam_direction)
                        if seg == 0:
                            project_point(camera_point, Ps[k], pixel_shape, pixel)
                            if pixel[0] != 255:
                                distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                                cos_theta = dot_3d(direction, cam_direction)
                                le_pdf = cloud_prob * HG_pdf(cos_theta, g_cloud) + (1 - cloud_prob) * rayleigh_pdf(
                                    cos_theta)
                                pc = (1 / (distance_to_camera ** 2)) * IS * le_pdf * attenuation
                                if is_in_medium[k]:
                                    assign_3d(next_point, ts[k])
                                else:
                                    get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)

                                get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                                path_size = estimate_voxels_size(dest_voxel, camera_voxel)
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
                                cuda.atomic.add(first_moment, (k, pixel[0], pixel[1]), pc)
                                cuda.atomic.add(second_moment, (k, pixel[0], pixel[1]), pc**2)
                                assign_3d(camera_point, current_point)
                                assign_3d(camera_voxel, current_voxel)
                                # cuda.atomic.add(I_scatter, 0, pc)

                        p = sample_uniform(rng_states, tid)
                        if p <= e_ddis:
                            assign_3d(temp_direction, cam_direction)
                        else:
                            assign_3d(temp_direction, direction)

                        p = sample_uniform(rng_states, tid)
                        if p <= cloud_prob:
                            HG_sample_direction(temp_direction, g_cloud, ddis_direction, rng_states, tid)
                        else:
                            rayleigh_sample_direction(temp_direction, ddis_direction, rng_states, tid)
                        cos_theta_regular = dot_3d(ddis_direction, direction)
                        cos_theta_ddis = dot_3d(ddis_direction, cam_direction)
                        true_phase_pdf = cloud_prob * HG_pdf(cos_theta_regular, g_cloud) + (
                                    1 - cloud_prob) * rayleigh_pdf(cos_theta_regular)
                        ddis_phase_pdf = cloud_prob * HG_pdf(cos_theta_ddis, g_cloud) + (1 - cloud_prob) * rayleigh_pdf(
                            cos_theta_ddis)
                        IS_cam = IS * (true_phase_pdf / (e_ddis * ddis_phase_pdf + (1 - e_ddis) * true_phase_pdf))
                        p = sample_uniform(rng_states, tid)
                        tau_rand = -math.log(1 - p)
                        ###########################################################
                        ############## voxel_traversal_algorithm_save #############
                        current_tau = 0.0
                        while True:
                            if not is_voxel_valid(camera_voxel, grid_shape):
                                in_medium = False
                                break
                            beta_cam = beta_cloud[camera_voxel[0], camera_voxel[1], camera_voxel[2]] + beta_air
                            length = travel_to_voxels_border(camera_point, camera_voxel, ddis_direction, voxel_size,
                                                             next_voxel)
                            current_tau += length * beta_cam
                            if current_tau >= tau_rand:
                                step_back = (current_tau - tau_rand) / beta_cam
                                camera_point[0] = camera_point[0] - step_back * ddis_direction[0]
                                camera_point[1] = camera_point[1] - step_back * ddis_direction[1]
                                camera_point[2] = camera_point[2] - step_back * ddis_direction[2]
                                in_medium = True
                                break
                            assign_3d(camera_voxel, next_voxel)

                        if in_medium:
                            project_point(camera_point, Ps[k], pixel_shape, pixel)
                            if pixel[0] != 255:
                                cloud_prob_cam = 1 - (beta_air / beta_cam)
                                attenuation_cam = attenuation * (
                                        cloud_prob_cam * w0_cloud + (1 - cloud_prob_cam) * w0_air)
                                distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                                cos_theta = dot_3d(ddis_direction, cam_direction)
                                le_pdf = cloud_prob_cam * HG_pdf(cos_theta, g_cloud) + (
                                            1 - cloud_prob_cam) * rayleigh_pdf(cos_theta)
                                pc = (1 / (distance_to_camera ** 2)) * IS_cam * le_pdf * attenuation_cam
                                if is_in_medium[k]:
                                    assign_3d(next_point, ts[k])
                                else:
                                    get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)

                                get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                                path_size = estimate_voxels_size(dest_voxel, camera_voxel)
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
                                cuda.atomic.add(first_moment, (k, pixel[0], pixel[1]), pc)
                                cuda.atomic.add(second_moment, (k, pixel[0], pixel[1]), pc**2)
                                # cuda.atomic.add(I_scatter, seg+1, pc)
                        ###################### DDIS LOCAL ESTIMATION #########################

        self.generate_paths = generate_paths
        self.post_generation = post_generation
        self.ddis_generation = ddis_generation
        self.render_cuda = render_cuda
        self.calc_gradient_contribution = calc_gradient_contribution
        self.render_differentiable_cuda = render_differentiable_cuda
        self.render_var = render_var

    def init_cuda_param(self, Np, init=False, seed=None):
        self.threadsperblock = threadsperblock
        self.blockspergrid = (Np + (threadsperblock - 1)) // threadsperblock
        if init:
            if seed is None:
                self.seed = np.random.randint(1, int(1e10))
            else:
                self.seed = seed
            self.rng_states = create_xoroshiro128p_states(threadsperblock * self.blockspergrid, seed=self.seed)

    def build_paths_list(self, Np, Ns, to_print=False):
        # inputs
        del(self.dpath_contrib)
        del(self.dgrad_contrib)
        self.Np = Np
        self.dbeta_zero.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        # outputs
        if to_print:
            print(f"preallocation weights: {3*(Ns+1)*Np*precis_size/1e9: .2f} GB")
        dstarting_points = cuda.to_device(np.empty((3, Np), dtype=float_precis))
        dscatter_points = cuda.to_device(np.empty((3, Ns, Np), dtype=float_precis))
        dscatter_sizes = cuda.to_device(np.empty(Np, dtype=np.uint8))
        dticket = cuda.to_device(np.zeros(1, dtype=np.int32))

        # cuda parameters
        self.init_cuda_param(Np)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid

        start = time()
        self.generate_paths[blockspergrid, threadsperblock]\
            (Np, Ns, self.dbeta_zero, beta_air, self.g_cloud,  self.dbbox, self.dbbox_size, self.dvoxel_size,
            self.dsun_direction, dstarting_points, dscatter_points, dscatter_sizes, self.rng_states, dticket)

        cuda.synchronize()
        if to_print:
            print("generate_paths took:", time() - start)


        start = time()
        ticket = dticket.copy_to_host()
        Np_nonan = ticket[0]
        scatter_sizes = dscatter_sizes.copy_to_host()
        starting_points = dstarting_points.copy_to_host()
        del(dscatter_sizes)
        del(dstarting_points)
        scatter_sizes = scatter_sizes[:Np_nonan]
        starting_points = np.ascontiguousarray(starting_points[:,:Np_nonan])

        scatter_inds = np.concatenate([np.array([0]), scatter_sizes])
        scatter_inds = np.cumsum(scatter_inds)
        total_num_of_scatter = np.sum(scatter_sizes)
        if to_print:
            print(f"total_num_of_scatter={total_num_of_scatter}")
            print(f"not none paths={Np_nonan / Np}")

        dscatter_inds = cuda.to_device(scatter_inds)
        dscatter_points_zipped = cuda.to_device(np.empty((3, total_num_of_scatter), dtype=float_precis))
        # dpixel_mat = cuda.to_device(np.empty((2, self.N_cams, total_num_of_scatter), dtype=np.uint8))
        self.init_cuda_param(Np_nonan)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid

        middle_alloc = (3*(total_num_of_scatter+Np_nonan) + 3*(Ns+1)*Np_nonan)* precis_size +  \
                       2*self.N_cams*total_num_of_scatter + Np_nonan*4
        middle_alloc /= 1e9
        if to_print:
            print("middle took:", time() - start)
            print(f"middle allocation weights: {middle_alloc: .2f} GB")
        start = time()
        self.post_generation[blockspergrid, threadsperblock](dscatter_points, dscatter_inds, dscatter_points_zipped)
        cuda.synchronize()
        del(dscatter_points)

        if to_print:
            post_alloc = 3 * (total_num_of_scatter + Np_nonan) * precis_size + 2 * self.N_cams * total_num_of_scatter + Np_nonan * 4
            post_alloc /= 1e9
            print(f"post allocation weights: {post_alloc: .2f} GB")
            # print(f"ddis allocation weights: {(8*self.N_cams*3*(Np_nonan+total_num_of_scatter))/1e9: .2f} GB")
            print("post_generation took:", time() - start)

        dstarting_points = cuda.to_device(starting_points)
        # dddis_starting_points = cuda.to_device(np.empty((3,self.N_cams,Np_nonan),dtype=float_precis))
        # dddis_points = cuda.to_device(np.empty((3,self.N_cams,total_num_of_scatter),dtype=float_precis))
        # self.ddis_generation[blockspergrid, threadsperblock]\
        #     (self.dbeta_zero, beta_air, self.g_cloud,  self.dbbox, self.dbbox_size, self.dvoxel_size, self.N_cams, self.dts,
        #      self.dPs, self.dpixels_shape, dscatter_inds, dscatter_points_zipped, dstarting_points, dddis_starting_points, dddis_points,
        #      self.rng_states)






        self.total_num_of_scatter = total_num_of_scatter
        self.Np_nonan = Np_nonan
        self.dpath_contrib = cuda.to_device(np.empty((self.N_cams, self.total_num_of_scatter), dtype=float_reg))
        self.dgrad_contrib = cuda.to_device(np.empty(self.total_num_of_scatter, dtype=float_reg))
        return dstarting_points, dscatter_points_zipped, dscatter_inds



    def render(self, cuda_paths, I_gt=None, to_print=False):
        # east declerations
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
        self.dI_total.copy_to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=float_reg))
        start = time()

        # dpath_contrib = cuda.to_device(np.empty((self.N_cams, self.total_num_of_scatter), dtype=float_reg))

        # dI_scatter = cuda.to_device(np.zeros(self.Ns+1, dtype=float_precis))
        self.render_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size,
             self.dvoxel_size, N_cams, self.dts, self.dPs, self.dpixels_shape, self.dis_in_medium,
             *cuda_paths,self.dI_total, self.dpath_contrib, self.rng_states)

        cuda.synchronize()
        I_total = self.dI_total.copy_to_host()
        # I_scatter = dI_scatter.copy_to_host()
        I_total /= self.Np
        if to_print:
            print("render_cuda took:",time() - start)
        # return I_total
        if I_gt is None:
            # del dpath_contrib
            return I_total

        ##### differentiable part ####
        self.dtotal_grad.copy_to_device(np.zeros_like(self.volume.beta_cloud, dtype=float_reg))
        # precalculating gradient contributions
        I_dif = (I_total - I_gt).astype(float_reg)
        self.dI_total.copy_to_device(I_dif)
        # dgrad_contrib = cuda.to_device(np.zeros(self.total_num_of_scatter, dtype=float_reg))
        start = time()
        self.calc_gradient_contribution[blockspergrid, threadsperblock]\
            (cuda_paths[1], self.dpath_contrib, self.dI_total, self.dPs, self.dpixels_shape,cuda_paths[2], self.dgrad_contrib)

        cuda.synchronize()
        if to_print:
            print("calc_gradient_contribution took:",time()-start)
        start = time()
        self.render_differentiable_cuda[blockspergrid, threadsperblock]\
            (self.dbeta_cloud, beta_air, g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size, self.dvoxel_size,
                                       N_cams, self.dts, self.dPs, self.dpixels_shape, self.dis_in_medium, *cuda_paths, self.dI_total, self.dpath_contrib,
             self.dgrad_contrib, self.dcloud_mask, self.dtotal_grad)


        cuda.synchronize()
        if to_print:
            print("render_differentiable_cuda took:", time() - start)
        # del(dpath_contrib)
        # del(dgrad_contrib)
        total_grad = self.dtotal_grad.copy_to_host()
        total_grad /= (self.Np * N_cams)
        return I_total, total_grad


    def render_std(self, cuda_paths, I_mean=None, to_print=False):
        # east declerations
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
        dfirst_moment = cuda.to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=float_reg))
        dsecond_moment = cuda.to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=float_reg))
        start = time()

        # dpath_contrib = cuda.to_device(np.empty((self.N_cams, self.total_num_of_scatter), dtype=float_reg))

        # dI_scatter = cuda.to_device(np.zeros(self.Ns+1, dtype=float_precis))
        self.render_var[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size,
             self.dvoxel_size, N_cams, self.dts, self.dPs, self.dpixels_shape, self.dis_in_medium,
             *cuda_paths, dfirst_moment, dsecond_moment, self.rng_states)
        cuda.synchronize()
        if I_mean is None:
            first_moment = dfirst_moment.copy_to_host()
        else:
            first_moment = I_mean * self.Np
        second_moment = dsecond_moment.copy_to_host()
        I_var = second_moment - (1/self.Np)*first_moment**2
        I_var /= self.Np-1
        I_total = first_moment/self.Np
        if to_print:
            print("render_cuda took:", time() - start)
        return I_total, np.sqrt(np.abs(I_var))

    def space_curving(self, image_mask, to_print=True):
        shape = self.volume.grid.shape
        N_cams = image_mask.shape[0]
        cloud_mask = np.zeros(shape, dtype=np.bool)
        point = np.zeros(3, dtype=np.float)
        voxel_size = self.volume.grid.voxel_size
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    point[0] = voxel_size[0] * (i + 0.5)
                    point[1] = voxel_size[1] * (j + 0.5)
                    point[2] = voxel_size[2] * (k + 0.5)
                    counter = 0
                    for cam_ind in range(N_cams):
                        pixel = self.cameras[cam_ind].project_point(point)
                        if image_mask[cam_ind, pixel[0], pixel[1]]:
                            counter += 1
                    if counter >= 7:
                        cloud_mask[i, j, k] = True
        cloud_mask = binary_dilation(cloud_mask).astype(np)
        if to_print:
            beta_cloud = self.volume.beta_cloud
            cloud_mask_real = beta_cloud > 0
            print(f"accuracy:", np.mean(cloud_mask == cloud_mask_real))
            print(f"fp:", np.mean((cloud_mask == 1) * (cloud_mask_real == 0)))
            fn = (cloud_mask == 0) * (cloud_mask_real == 1)
            print(f"fn:", np.mean(fn))
            fn_exp = (fn * beta_cloud).reshape(-1)
            print(f"fn_exp mean:", np.mean(fn_exp))
            print(f"fn_exp max:", np.max(fn_exp))
            print(f"fn_exp min:", np.min(fn_exp[fn_exp != 0]))
            print("missed beta:", np.sum(fn_exp) / np.sum(beta_cloud))

        self.volume.set_mask(cloud_mask)
        self.dcloud_mask.copy_to_device(cloud_mask)

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




