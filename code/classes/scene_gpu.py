from classes.grid import *
from classes.volume import *
from classes.sparse_path import SparsePath
from classes.path import CudaPaths
from utils import  theta_phi_to_direction
from tqdm import tqdm
import math
from numba import njit
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64
from cuda_utils import *
threadsperblock = 256


class SceneGPU(object):
    def __init__(self, volume: Volume, cameras, sun_angles, g_cloud, g_air, Ns):
        self.Ns = Ns
        self.volume = volume
        self.sun_angles = sun_angles
        self.sun_direction = theta_phi_to_direction(*sun_angles)
        self.sun_direction[np.abs(self.sun_direction) < 1e-6] = 0
        self.cameras = cameras
        self.g_cloud = g_cloud
        self.g_air = g_air
        self.N_cams = len(cameras)
        self.N_pixels = cameras[0].pixels
        self.is_camera_in_medium = np.zeros(self.N_cams, dtype=np.bool)
        for k in range(self.N_cams):
            self.is_camera_in_medium[k] = self.volume.grid.is_in_bbox(self.cameras[k].t)

        N_cams = self.N_cams
        pixels = self.cameras[0].pixels
        ts = np.vstack([cam.t.reshape(1, -1) for cam in self.cameras])
        Ps = np.concatenate([cam.P.reshape(1, 3, 4) for cam in self.cameras], axis=0)

        # gpu array
        self.dbeta_cloud = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dI_total = cuda.device_array((N_cams, pixels[0], pixels[1]), dtype=float_reg)
        self.dtotal_grad = cuda.device_array(self.volume.beta_cloud.shape, dtype=float_reg)
        self.dbbox = cuda.to_device(self.volume.grid.bbox)
        self.dbbox_size = cuda.to_device(self.volume.grid.bbox_size)
        self.dvoxel_size = cuda.to_device(self.volume.grid.voxel_size)
        self.dsun_direction = cuda.to_device(self.sun_direction)
        self.dpixels_shape = cuda.to_device(self.cameras[0].pixels)
        self.dts = cuda.to_device(ts)
        self.dPs = cuda.to_device(Ps)
        self.dis_in_medium = cuda.to_device(self.is_camera_in_medium)
        self.dcloud_mask = cuda.to_device(self.volume.cloud_mask)
        @cuda.jit()
        def render_cuda(voxels_mat, lengths, ISs_mat, angles_mat, scatter_angles, sv, camera_pixels, scatter_inds, voxel_inds, beta_cloud,
                        beta_air, w0_cloud, w0_air, g_cloud, g_air, I_total): # sv for scatter_voxels
            tid = cuda.grid(1)
            if tid < voxel_inds.shape[0] - 1:
                # reading thread indices
                scatter_start = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_start
                voxel_start = voxel_inds[tid]
                voxels_size = voxel_inds[tid+1] - voxel_start
                # rendering
                path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=float_reg)

                for ind in range(voxels_size):
                    row_ind = voxel_start + ind
                    i, j, k, cam_ind, seg = voxels_mat[row_ind]
                    if i == 255:
                        print(i, j, k, cam_ind, seg)
                        break
                    L = lengths[row_ind]
                    beta = beta_cloud[i, j, k] + beta_air
                    if cam_ind == 255:
                        for cam_j in range(N_cams):
                            for seg_j in range(N_seg - seg):
                                path_contrib[cam_j, seg + seg_j] += beta * L
                    else:
                        path_contrib[cam_ind, seg] += beta * L

                prod = 1
                for seg in range(N_seg):
                    seg_ind = scatter_start + seg
                    beta = beta_cloud[sv[0, seg_ind], sv[1, seg_ind], sv[2, seg_ind]]
                    cloud_prob = beta / (beta + beta_air)
                    air_prob = 1 - cloud_prob
                    prod *= (w0_cloud * beta + w0_air * beta_air)
                    for cam_j in range(N_cams):
                        angle = angles_mat[cam_j, seg_ind]
                        angle_pdf = cloud_prob*pdf(angle, g_cloud) + air_prob*pdf(angle, g_air)
                        pc = ISs_mat[cam_j, seg_ind] * math.exp(-path_contrib[cam_j, seg]) * angle_pdf * prod
                        pixel = camera_pixels[:, cam_j, seg_ind]
                        cuda.atomic.add(I_total, (cam_j, pixel[0], pixel[1]), pc)
                    scatter_angle = scatter_angles[seg_ind]
                    scatter_angle_pdf = cloud_prob*pdf(scatter_angle, g_cloud) + air_prob*pdf(scatter_angle, g_air)
                    prod *= scatter_angle_pdf


        @cuda.jit()
        def render_differentiable_cuda(voxels_mat, lengths, ISs_mat,  angles_mat, scatter_angles, sv,camera_pixels,
                                       scatter_inds, voxel_inds, beta_cloud, beta_air, w0_cloud, w0_air, g_cloud, g_air,
                                       I_dif, total_grad, cloud_mask):
            tid = cuda.grid(1)
            if tid < voxel_inds.shape[0] - 1:

                # reading thread indices
                scatter_start = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_start
                voxel_start = voxel_inds[tid]
                voxels_size = voxel_inds[tid + 1] - voxel_start


                path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=float_reg)
                # rendering
                for ind in range(voxels_size):
                    row_ind = voxel_start + ind
                    i, j, k, cam_ind, seg = voxels_mat[row_ind]
                    if i == 255:
                        print("bug:", i)
                        break

                    L = lengths[row_ind]
                    beta = beta_cloud[i,j,k] + beta_air
                    if cam_ind == 255:
                        for cam_j in range(N_cams):
                            for seg_j in range(N_seg - seg):
                                path_contrib[cam_j, seg + seg_j] += beta * L
                    else:
                        path_contrib[cam_ind, seg] += beta * L

                prod = 1
                for seg in range(N_seg):
                    seg_ind = scatter_start + seg
                    beta = beta_cloud[sv[0, seg_ind], sv[1, seg_ind], sv[2, seg_ind]]
                    cloud_prob = beta / (beta + beta_air)
                    air_prob = 1 - cloud_prob
                    prod *= (w0_cloud * beta + w0_air * beta_air)
                    for cam_j in range(N_cams):
                        angle = angles_mat[cam_j, seg_ind]
                        angle_pdf = cloud_prob * pdf(angle, g_cloud) + air_prob * pdf(angle, g_air)
                        path_contrib[cam_j, seg] = ISs_mat[cam_j, seg_ind] * math.exp(-path_contrib[cam_j, seg]) * angle_pdf * prod
                    scatter_angle = scatter_angles[seg_ind]
                    scatter_angle_pdf = cloud_prob * pdf(scatter_angle, g_cloud) + air_prob * pdf(scatter_angle, g_air)
                    prod *= scatter_angle_pdf



                for ind in range(voxels_size):
                    if i == 255:
                        print("bug:",i)
                    row_ind = ind + voxel_start
                    i, j, k, cam_ind, seg = voxels_mat[row_ind]
                    if not cloud_mask[i,j,k]:# or beta_cloud[i,j,k] < 1e-3:
                        continue
                    L = lengths[row_ind]
                    seg_ind = seg + scatter_start
                    if cam_ind == 255:
                        for pj in range(Ns - seg):
                            for cam_j in range(N_cams):
                                pixel = camera_pixels[:, cam_j, seg_ind + pj]
                                grad_contrib = -L * path_contrib[cam_j, seg + pj] * I_dif[cam_j, pixel[0], pixel[1]]
                                # if beta_cloud[i,j,k] < 0.5:
                                #     print(grad_contrib, path_contrib[cam_j, seg + pj])
                                cuda.atomic.add(total_grad, (i, j, k), grad_contrib)

                    else:
                        pixel = camera_pixels[:, cam_ind, seg_ind]
                        grad_contrib = -L * path_contrib[cam_ind, seg] * I_dif[cam_ind, pixel[0], pixel[1]]
                        cuda.atomic.add(total_grad, (i, j, k), grad_contrib)


                for seg in range(N_seg):
                    seg_ind = seg + scatter_start
                    if not cloud_mask[sv[0,seg_ind], sv[1,seg_ind], sv[2, seg_ind]]:
                        continue
                    beta_scatter = w0_cloud * beta_cloud[sv[0, seg_ind], sv[1, seg_ind], sv[2, seg_ind]] + w0_air * beta_air
                    for pj in range(Ns - seg):
                        for cam_ind in range(N_cams):
                            pixel = camera_pixels[:, cam_ind, seg_ind + pj]
                            grad_contrib = (w0_cloud / beta_scatter) * path_contrib[cam_ind, seg + pj] * \
                                           I_dif[cam_ind, pixel[0], pixel[1]]
                            cuda.atomic.add(total_grad, (sv[0, seg_ind], sv[1, seg_ind], sv[2, seg_ind]), grad_contrib)

        @cuda.jit()
        def calculate_paths_matrix(Ns, beta_cloud, beta_air, bbox, bbox_size, voxel_size, N_cams, ts, is_in_medium, g_cloud, g_air, pixels_shape,
                                   starting_points, scatter_points, scatter_inds, voxel_inds, scatter_angles,
                                   angles_mat, ISs_mat, scatter_voxels, camera_pixels, voxels_mat, lengths):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:
                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_ind
                voxel_ind = voxel_inds[tid]
                voxels_size = voxel_inds[tid+1] - voxel_ind

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
                cam_direction = cuda.local.array(3, dtype=float_precis)
                dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                beta = 1.0 # for type decleration
                IS = 1.0

                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    # next_point = scatter_points[:, seg_ind]
                    # print_3d(next_point)
                    assign_3d(next_point, scatter_points[:, seg, tid])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    distance = distance_and_direction(current_point, next_point, direction)
                    if seg > 0:
                        cos_theta = dot_3d(cam_direction, direction)
                        scatter_angles[seg_ind-1] = cos_theta
                        cloud_prob = (beta - beta_air) / beta
                        air_prob = 1 - cloud_prob
                        IS *= (cloud_prob * pdf(cos_theta, g_cloud) + air_prob * pdf(cos_theta,
                                                                                     g_air)) ** -1  # angle pdf
                    ###########################################################
                    ############## voxel_fixed traversal_algorithm_save #############

                    path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    reach_dest = False
                    counter = 0
                    total_counter = 0
                    current_length = 0
                    tau = 0

                    # while not reach_dest:
                    for pi in range(path_size):
                        if pi == path_size - 1:
                            length = distance - current_length
                            assign_3d(current_point, next_point)
                            reach_dest = True
                        else:
                            length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel)

                        beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + beta_air
                        tau += beta*length

                        if not is_voxel_valid(current_voxel, grid_shape):
                            print("buggggg")
                        voxels_mat[voxel_ind, 0] = current_voxel[0]  # voxels
                        voxels_mat[voxel_ind, 1] = current_voxel[1]  # voxels
                        voxels_mat[voxel_ind, 2] = current_voxel[2]  # voxels
                        voxels_mat[voxel_ind, 3] = 255  # cam (255 for all cams)
                        voxels_mat[voxel_ind, 4] = seg  # segment
                        lengths[voxel_ind] = length
                        voxel_ind += 1
                        counter += 1
                        total_counter += 1
                        current_length += length
                        if not reach_dest:
                            assign_3d(current_voxel, next_voxel)


                    if counter != path_size:
                        print("path_size bug:",counter, path_size)


                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################
                    scatter_voxels[0, seg_ind] = current_voxel[0]
                    scatter_voxels[1, seg_ind] = current_voxel[1]
                    scatter_voxels[2, seg_ind] = current_voxel[2]

                    IS *= (beta * math.exp(-tau)) ** -1  # length pdf
                    for k in range(N_cams):
                        project_point(current_point, Ps[k], pixels_shape, pixel)
                        camera_pixels[0, k, seg_ind] = pixel[0]
                        camera_pixels[1, k, seg_ind] = pixel[1]
                        if not is_pixel_valid(pixel):
                            continue
                        else:
                            assign_3d(camera_voxel, current_voxel)
                            assign_3d(camera_point, current_point)
                            distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                            cos_theta = dot_3d(direction, cam_direction)
                            angles_mat[k, seg_ind] = cos_theta
                            ISs_mat[k, seg_ind] = (1 / (distance_to_camera ** 2)) * IS
                            if is_in_medium[k]:
                                assign_3d(dest, ts[k])
                            else:
                                get_intersection_with_borders(camera_point, cam_direction, bbox, dest)

                            get_voxel_of_point(dest, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)
                            counter = 0
                            ###########################################################################
                            ######################## local estimation save ############################
                            # for pi in range(path_size):
                            reach_dest = False
                            while not reach_dest:

                                if compare_3d(camera_voxel, dest_voxel):
                                    length = calc_distance(camera_point, dest)
                                    reach_dest = True
                                else:
                                    length = travel_to_voxels_border(camera_point, camera_voxel, cam_direction,
                                                                     voxel_size, next_voxel)
                                    # update row
                                if not is_voxel_valid(current_voxel, grid_shape):
                                    print("buggggg: cam", camera_voxel[0],camera_voxel[1], camera_voxel[2])
                                voxels_mat[voxel_ind, 0] = camera_voxel[0]  # voxels
                                voxels_mat[voxel_ind, 1] = camera_voxel[1]  # voxels
                                voxels_mat[voxel_ind, 2] = camera_voxel[2]  # voxels
                                voxels_mat[voxel_ind, 3] = k  # cam (255 for all cams)
                                voxels_mat[voxel_ind, 4] = seg  # segment
                                lengths[voxel_ind] = length
                                # increase global voxel index
                                voxel_ind += 1
                                total_counter += 1
                                if not reach_dest:
                                    assign_3d(camera_voxel, next_voxel)
                                counter += 1
                            ######################## local estimation save ############################
                            ###########################################################################
                            if counter != path_size:
                                print("path_size camera bug:", counter, path_size)

                    assign_3d(cam_direction, direction) #using cam direction as temp direction
                # if total_counter != voxels_size:
                #     print("total_counter bug",total_counter,voxels_size)
                # else:
                #     print(total_counter, voxels_size)


        @cuda.jit()
        def generate_paths(Np, Ns, beta_cloud, beta_air, bbox, bbox_size, voxel_size, sun_direction, N_cams,
                           pixels_shape, ts, Ps, is_in_medium, starting_points, scatter_points,
                           scatter_sizes, voxel_sizes, rng_states):

            tid = cuda.grid(1)
            if tid < Np:
                grid_shape = beta_cloud.shape

                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)
                dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                assign_3d(direction, sun_direction)

                # sample entering point
                p = sample_uniform(rng_states, tid)
                current_point[0] = bbox_size[0] * p + bbox[0, 0]

                p = sample_uniform(rng_states, tid)
                current_point[1] = bbox_size[1] * p + bbox[1, 0]

                current_point[2] = bbox[2, 1]
                starting_points[0, tid] = current_point[0]
                starting_points[1, tid] = current_point[1]
                starting_points[2, tid] = current_point[2]
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                total_voxels_size = 0
                for seg in range(Ns):
                    temp_voxels_count = 0
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
                    scatter_points[0, seg, tid] = current_point[0]
                    scatter_points[1, seg, tid] = current_point[1]
                    scatter_points[2, seg, tid] = current_point[2]

                    # calculating  total_voxels_size
                    for k in range(N_cams):
                        project_point(current_point, Ps[k], pixels_shape, pixel)
                        if is_pixel_valid(pixel):
                            distance_and_direction(current_point, ts[k], cam_direction)
                            if is_in_medium[k]:
                                assign_3d(dest, ts[k])
                            else:
                                get_intersection_with_borders(current_point, cam_direction, bbox, dest)
                            get_voxel_of_point(dest, grid_shape, bbox, bbox_size, camera_voxel)
                            total_voxels_size += estimate_voxels_size(current_voxel, camera_voxel)
                    # sampling new direction
                    cloud_prob = (beta - beta_air) / beta
                    p = sample_uniform(rng_states, tid)
                    if p <= cloud_prob:
                        g = g_cloud
                    else:
                        g = g_air
                    sample_direction(direction, g, new_direction, rng_states, tid)
                    assign_3d(direction, new_direction)

                # voxels and scatter sizes for this path (this is not in a loop)
                N_seg = seg + int(in_medium)
                scatter_sizes[tid] = N_seg
                voxel_sizes[tid] = total_voxels_size

        self.render_cuda = render_cuda
        self.render_differentiable_cuda = render_differentiable_cuda
        self.calculate_paths_matrix = calculate_paths_matrix
        self.generate_paths = generate_paths

    def init_cuda_param(self, Np, seed=None):
        self.threadsperblock = threadsperblock
        self.blockspergrid = (Np + (threadsperblock - 1)) // threadsperblock
        if seed is None:
            self.seed = np.random.randint(1, int(1e10))
        else:
            self.seed = seed
        self.rng_states = create_xoroshiro128p_states(threadsperblock * self.blockspergrid, seed=self.seed)

    def build_paths_list(self, Np, Ns):
        g_cloud = self.g_cloud
        g_air = self.g_air

        # inputs
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        # outputs
        dstarting_points = cuda.to_device(np.zeros((3, Np), dtype=float_precis))
        dscatter_points = cuda.to_device(np.zeros((3, Ns, Np), dtype=float_precis))
        dscatter_sizes = cuda.to_device(np.zeros(Np, dtype=np.uint8))
        dvoxel_sizes = cuda.to_device(np.zeros(Np, dtype=np.uint32))

        # cuda parameters
        self.init_cuda_param(Np)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid


        self.generate_paths[blockspergrid, threadsperblock]\
            (Np, Ns, self.dbeta_cloud, beta_air,  self.dbbox, self.dbbox_size, self.dvoxel_size,
            self.dsun_direction, self.N_cams, self.dpixels_shape, self.dts, self.dPs, self.dis_in_medium,
             dstarting_points, dscatter_points, dscatter_sizes, dvoxel_sizes, self.rng_states)

        cuda.synchronize()


        voxel_sizes = dvoxel_sizes.copy_to_host()
        scatter_sizes = dscatter_sizes.copy_to_host()
        starting_points = dstarting_points.copy_to_host()
        scatter_points = dscatter_points.copy_to_host()
        del(dvoxel_sizes)
        del(dscatter_sizes)
        del(dstarting_points)
        del(dscatter_points)
        print(((scatter_sizes!=0) == (voxel_sizes!=0)).all())
        active_paths = scatter_sizes != 0
        voxel_sizes = voxel_sizes[active_paths]
        scatter_sizes = scatter_sizes[active_paths]
        starting_points = starting_points[:, active_paths]
        scatter_points = np.ascontiguousarray(scatter_points[:, :, active_paths])

        total_num_of_voxels = np.sum(voxel_sizes)
        total_num_of_scatter = np.sum(scatter_sizes)

        voxel_inds = np.concatenate([np.array([0]), voxel_sizes])
        voxel_inds = np.cumsum(voxel_inds)
        scatter_inds = np.concatenate([np.array([0]), scatter_sizes])
        scatter_inds = np.cumsum(scatter_inds)

        Np_nonan = int(np.sum(active_paths))
        self.init_cuda_param(Np_nonan)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        print(f"not none paths={Np_nonan / Np}")
        voxels_GB = (total_num_of_voxels * 5 + total_num_of_voxels * eff_size) / 1e9
        scatters_GB = (3*total_num_of_scatter*precis_size + 3*total_num_of_scatter + 2*self.N_cams*total_num_of_scatter
                       + 2*self.N_cams*total_num_of_scatter*reg_size +total_num_of_scatter*reg_size + Np*5)/1e9
        print(f"voxels dataset weights: {voxels_GB: .2f} GB")
        print(f"scatters dataset weights: {scatters_GB: .2f} GB")
        print(f"total dataset weights: {voxels_GB+scatters_GB: .2f} GB")

        dstarting_points = cuda.to_device(starting_points)
        dscatter_points = cuda.to_device(scatter_points)
        dscatter_voxels = cuda.to_device(np.zeros((3, total_num_of_scatter), dtype=np.uint8))
        dcamera_pixels = cuda.to_device(np.zeros((2, self.N_cams, total_num_of_scatter), dtype=np.uint8))
        dISs_mat = cuda.to_device(np.zeros((self.N_cams, total_num_of_scatter), dtype=float_reg))
        dangles_mat = cuda.to_device(np.zeros((self.N_cams, total_num_of_scatter), dtype=float_reg))
        dscatter_angles = cuda.to_device(np.zeros(total_num_of_scatter, dtype=float_reg))

        dvoxel_inds = cuda.to_device(voxel_inds)
        dscatter_inds = cuda.to_device(scatter_inds)
        dvoxels_mat = cuda.to_device(np.ones((total_num_of_voxels, 5), dtype=np.uint8) * 255)
        dlengths = cuda.to_device(np.zeros(total_num_of_voxels, dtype=float_eff))

        # adding voxels meta
        self.calculate_paths_matrix[blockspergrid, threadsperblock]\
            (Ns, self.dbeta_cloud, beta_air, self.dbbox, self.dbbox_size, self.dvoxel_size, self.N_cams, self.dts,
             self.dis_in_medium, g_cloud, g_air, self.dpixels_shape, dstarting_points, dscatter_points, dscatter_inds,
             dvoxel_inds, dscatter_angles, dangles_mat, dISs_mat, dscatter_voxels, dcamera_pixels, dvoxels_mat, dlengths)


        voxels_mat = dvoxels_mat.copy_to_host()

        cuda.synchronize()
        del(dscatter_points)
        del(dstarting_points)
        return( dvoxels_mat, dlengths, dISs_mat, dangles_mat, dscatter_angles, dscatter_voxels, dcamera_pixels,\
                 dscatter_inds, dvoxel_inds), Np_nonan



    def render(self, cuda_paths, Np, Np_nonan, I_gt=None):
        # east declerations
        N_cams = len(self.cameras)
        pixels_shape = self.cameras[0].pixels
        beta_cloud = self.volume.beta_cloud
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air
        g_cloud = self.g_cloud
        g_air = self.g_air
        self.init_cuda_param(Np_nonan)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        self.dbeta_cloud.copy_to_device(self.volume.beta_cloud)
        self.dI_total.copy_to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=float_reg))



        self.render_cuda[blockspergrid, threadsperblock] \
            (*cuda_paths, self.dbeta_cloud, beta_air, w0_cloud, w0_air, g_cloud, g_air, self.dI_total)

        cuda.synchronize()
        I_total = self.dI_total.copy_to_host()
        I_total /= Np
        if I_gt is None:
            return I_total

        # differentiable part
        self.dtotal_grad.copy_to_device(np.zeros_like(beta_cloud, dtype=float_reg))
        I_dif = (I_total - I_gt).astype(float_reg)
        # print(f"I_dif={np.linalg.norm(I_dif)}, I_total={np.linalg.norm(I_total)}, I_gt={np.linalg.norm(I_gt)}")
        self.dI_total.copy_to_device(I_dif)

        self.render_differentiable_cuda[blockspergrid, threadsperblock] \
            (*cuda_paths, self.dbeta_cloud, beta_air, w0_cloud, w0_air, g_cloud, g_air, self.dI_total, self.dtotal_grad,
             self.dcloud_mask)

        cuda.synchronize()
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
        text += f"g_cloud={self.g_cloud}, g_air={self.g_air}  \n\n"
        return text




