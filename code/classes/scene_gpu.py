from classes.grid import *
from classes.volume import *
from classes.sparse_path import SparsePath
from classes.path import CudaPaths
from utils import  theta_phi_to_direction
from tqdm import tqdm
import math
from numba import njit
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from cuda_utils import *

class SceneGPU(object):
    def __init__(self, volume: Volume, cameras, sun_angles, g, Ns):
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
        def render_cuda(voxels_mat, lengths, ISs_mat, sv, camera_pixels, scatter_sizes, voxel_inds, betas,
                        beta_air, w0_cloud, w0_air, I_total): # sv for scatter_voxels
            tid = cuda.grid(1)
            if tid < voxel_inds.shape[0] - 1:
                # reading thread indices
                scatter_start = tid * Ns
                voxel_start = voxel_inds[tid]
                voxels_size = voxel_inds[tid+1] - voxel_start

                # rendering
                N_seg = scatter_sizes[tid]
                path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=np.float32)

                for ind in range(voxels_size):
                    row_ind = voxel_start + ind
                    i, j, k, cam_ind, seg = voxels_mat[row_ind]
                    if i == 255:
                        # print("error!!!!!!!!!!!!!!!")
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
                    seg_ind = scatter_start + seg
                    prod *= (w0_cloud * (betas[sv[0, seg_ind], sv[1, seg_ind], sv[2, seg_ind]] - beta_air) + w0_air * beta_air)
                    for cam_j in range(N_cams):
                        pc = ISs_mat[cam_j, seg_ind] * math.exp(-path_contrib[cam_j, seg]) * prod
                        pixel = camera_pixels[:, cam_j, seg_ind]
                        cuda.atomic.add(I_total, (cam_j, pixel[0], pixel[1]), pc)

        @cuda.jit()
        def render_differentiable_cuda(voxels_mat, lengths, ISs_mat, sv,camera_pixels, scatter_sizes, voxel_inds, betas,
                                       beta_air, w0_cloud, w0_air, I_dif, total_grad,):
            tid = cuda.grid(1)
            if tid < voxel_inds.shape[0] - 1:
                # reading thread indices
                scatter_start = Ns * tid
                voxel_start = voxel_inds[tid]
                voxels_size = voxel_inds[tid + 1] - voxel_start


                N_seg = scatter_sizes[tid]
                path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=np.float32)
                # rendering
                for ind in range(voxels_size):
                    row_ind = voxel_start + ind
                    i, j, k, cam_ind, seg = voxels_mat[row_ind]
                    L = lengths[row_ind]
                    if cam_ind == 255:
                        for cam_j in range(N_cams):
                            for seg_j in range(N_seg - seg):
                                path_contrib[cam_j, seg + seg_j] += betas[i, j, k] * L
                    else:
                        path_contrib[cam_ind, seg] += betas[i, j, k] * L

                prod = 1
                for seg in range(N_seg):
                    seg_ind = scatter_start + seg
                    prod *= (w0_cloud * (betas[sv[0, seg_ind], sv[1, seg_ind], sv[2, seg_ind]] - beta_air) + w0_air * beta_air)
                    for cam_j in range(N_cams):
                        path_contrib[cam_j, seg] = ISs_mat[cam_j, seg_ind] * math.exp(-path_contrib[cam_j, seg]) * prod


                for ind in range(voxels_size):
                    row_ind = ind + voxel_start
                    i, j, k, cam_ind, seg = voxels_mat[row_ind]
                    L = lengths[row_ind]
                    seg_ind = seg + scatter_start
                    if cam_ind == 255:
                        for pj in range(Ns - seg):
                            for cam_j in range(N_cams):
                                pixel = camera_pixels[:, cam_j, seg_ind + pj]
                                grad_contrib = -L * path_contrib[cam_j, seg + pj] * I_dif[cam_j, pixel[0], pixel[1]]
                                cuda.atomic.add(total_grad, (i, j, k), grad_contrib)

                    else:
                        pixel = camera_pixels[:, cam_ind, seg_ind]
                        grad_contrib = -L * path_contrib[cam_ind, seg] * I_dif[cam_ind, pixel[0], pixel[1]]
                        cuda.atomic.add(total_grad, (i, j, k), grad_contrib)

                for seg in range(N_seg):
                    seg_ind = seg + scatter_start
                    beta_scatter = w0_cloud * (betas[sv[0, seg_ind], sv[1, seg_ind], sv[2, seg_ind]] - beta_air) + w0_air * beta_air
                    for pj in range(Ns - seg):
                        for cam_ind in range(N_cams):
                            pixel = camera_pixels[:, cam_ind, seg_ind + pj]
                            grad_contrib = (w0_cloud / beta_scatter) * path_contrib[cam_ind, seg + pj] * \
                                           I_dif[cam_ind, pixel[0], pixel[1]]
                            cuda.atomic.add(total_grad, (sv[0, seg_ind], sv[1, seg_ind], sv[2, seg_ind]), grad_contrib)

        eps = 1e-4
        @cuda.jit()
        def calculate_paths_matrix(Np, Ns, betas, bbox, bbox_size, voxel_size, N_cams, ts, is_in_medium,
                                   starting_points,
                                   scatter_points, scatter_sizes, voxel_inds, camera_pixels, voxels_mat, lengths):
            tid = cuda.grid(1)
            if tid < Np and scatter_sizes[tid] != 0:
            # if tid == 715:
                N_seg = scatter_sizes[tid]
                scatter_ind = tid * Ns
                voxel_ind = voxel_inds[tid]
                num_of_voxels = voxel_inds[tid+1] - voxel_ind
                grid_shape = betas.shape

                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                dest_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                temp_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=np.float64)
                camera_point = cuda.local.array(3, dtype=np.float64)
                next_point = cuda.local.array(3, dtype=np.float32)
                direction = cuda.local.array(3, dtype=np.float64)
                cam_direction = cuda.local.array(3, dtype=np.float64)
                dest = cuda.local.array(3, dtype=np.float64)
                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)
                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    assign_3d(next_point, scatter_points[:, seg_ind])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    ###########################################################
                    ############## voxel_fixed traversal_algorithm_save #############
                    current_length = 0
                    distance = distance_and_direction(current_point, next_point, direction)
                    reach_dest = False
                    while not reach_dest:
                        # print2_3d(current_voxel, dest_voxel)
                        # if current_voxel[0] == 255 or current_voxel[1] == 255 or current_voxel[2] == 255:
                        #     print("255 error")
                        #     break
                        # if current_length > distance:
                        #     print(current_length - distance)
                        if compare_3d(current_voxel, dest_voxel):
                            length = calc_distance(current_point, next_point)
                            assign_3d(current_point, next_point)
                            reach_dest = True
                        else:
                            length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)
                            # current_length += length
                        # update row
                            if length < 0:
                                print(length)
                        voxels_mat[voxel_ind, 0] = current_voxel[0]  # voxels
                        voxels_mat[voxel_ind, 1] = current_voxel[1]  # voxels
                        voxels_mat[voxel_ind, 2] = current_voxel[2]  # voxels
                        voxels_mat[voxel_ind, 3] = 255  # cam (255 for all cams)
                        voxels_mat[voxel_ind, 4] = seg  # segment
                        lengths[voxel_ind] = length
                        voxel_ind += 1
                        #
                        # if current_length >= distance - eps:
                        #     if current_length <= distance:
                        #         step_back = 0
                        #     else:
                        #         step_back = current_length - distance
                        #     step_back = current_length - distance + eps
                        #     lengths[voxel_ind - 1] -= step_back
                        #     current_point[0] = current_point[0] - step_back * direction[0]
                        #     current_point[1] = current_point[1] - step_back * direction[1]
                        #     current_point[2] = current_point[2] - step_back * direction[2]
                        #     break
                        if not reach_dest:
                            assign_3d(current_voxel, next_voxel)
                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################

                    for k in range(N_cams):
                        pixel = camera_pixels[:, k, seg_ind]
                        if not is_pixel_valid(pixel):
                            continue
                        else:
                            assign_3d(camera_voxel, current_voxel)
                            assign_3d(camera_point, current_point)
                            distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                            if is_in_medium[k]:
                                assign_3d(dest, ts[k])
                            else:
                                distance_to_camera = get_intersection_with_borders(camera_point, cam_direction, bbox,
                                                                                   dest)
                            get_voxel_of_point(dest, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)
                            ###########################################################################
                            ######################## local estimation save ###################
                            # current_length = 0
                            counter = 0
                            # for ps in range(path_size):
                            reach_dest = False
                            while not reach_dest:

                                if compare_3d(camera_voxel, dest_voxel):
                                    # print("reached!")
                                    length = calc_distance(camera_point, dest)
                                    reach_dest = True
                                else:

                                    length = travel_to_voxels_border(camera_point, camera_voxel, cam_direction, voxel_size,
                                                                 next_voxel)
                                if length <= 0:
                                    print(length, camera_point[0], camera_point[1], camera_point[2])

                                # update row
                                voxels_mat[voxel_ind, 0] = camera_voxel[0]  # voxels
                                voxels_mat[voxel_ind, 1] = camera_voxel[1]  # voxels
                                voxels_mat[voxel_ind, 2] = camera_voxel[2]  # voxels
                                voxels_mat[voxel_ind, 3] = k  # cam (255 for all cams)
                                voxels_mat[voxel_ind, 4] = seg  # segment
                                lengths[voxel_ind] = length
                                voxel_ind += 1
                                if not reach_dest:
                                    assign_3d(camera_voxel, next_voxel)
                                # if current_length >= distance_to_camera - eps:
                                #     if current_length <= distance_to_camera:
                                #         step_back = 0
                                #     else:
                                #         step_back = current_length - distance_to_camera
                                #     lengths[voxel_ind - 1] -= step_back
                                #     camera_point[0] = camera_point[0] - step_back * cam_direction[0]
                                #     camera_point[1] = camera_point[1] - step_back * cam_direction[1]
                                #     camera_point[2] = camera_point[2] - step_back * cam_direction[2]
                                #     break
                                # if ps == path_size -1:
                                #     print("bugggg")
                            # if counter != path_size: #and abs(counter - path_size) != 1:
                            #     print(counter, path_size)
                            ######################## local estimation save ###################
                            ###########################################################################
                            # if not compare_eps_3d(dest, camera_point, 1e-4):
                            #     print2_3d(dest, camera_point)
        @cuda.jit()
        def generate_paths(Np, Ns, betas, bbox, bbox_size, voxel_size, sun_direction, N_cams, pixels_shape, ts, Ps,
                           is_in_medium, g,
                           scatter_voxels, starting_points, scatter_points, camera_pixels, ISs_mat, scatter_sizes,
                           voxel_sizes, rng_states):

            tid = cuda.grid(1)
            if tid < Np:
                start_ind = tid * Ns
                grid_shape = betas.shape

                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                camera_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=np.float64)
                direction = cuda.local.array(3, dtype=np.float64)
                new_direction = cuda.local.array(3, dtype=np.float64)
                cam_direction = cuda.local.array(3, dtype=np.float64)
                dest = cuda.local.array(3, dtype=np.float64)
                pixel = cuda.local.array(2, dtype=np.uint8)
                assign_3d(direction, sun_direction)

                # sample entering point
                p = xoroshiro128p_uniform_float32(rng_states, tid)
                current_point[0] = bbox_size[0] * p + bbox[0, 0]

                p = xoroshiro128p_uniform_float32(rng_states, tid)
                current_point[1] = bbox_size[1] * p + bbox[1, 0]

                current_point[2] = bbox[2, 1]
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
                        if current_voxel[0] >= grid_shape[0] or current_voxel[1] >= grid_shape[1] \
                                or current_voxel[2] >= grid_shape[2]:
                            in_medium = False
                            break
                        beta = betas[current_voxel[0], current_voxel[1], current_voxel[2]]
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size,
                                                         next_voxel)
                        # if length <= 0:
                        #     print(length)
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
                    scatter_points[0, seg_ind] = current_point[0]
                    scatter_points[1, seg_ind] = current_point[1]
                    scatter_points[2, seg_ind] = current_point[2]
                    scatter_voxels[0, seg_ind] = current_voxel[0]
                    scatter_voxels[1, seg_ind] = current_voxel[1]
                    scatter_voxels[2, seg_ind] = current_voxel[2]
                    # Scatter IS in main trajectory
                    IS *= 1 / (beta * (1 - p))

                    # calculating ISs_mat and total_voxels_size

                    for k in range(N_cams):
                        project_point(current_point, Ps[k], pixels_shape, pixel)
                        camera_pixels[0, k, seg_ind] = pixel[0]
                        camera_pixels[1, k, seg_ind] = pixel[1]
                        if not is_pixel_valid(pixel):
                            continue
                        else:
                            distance_to_camera = distance_and_direction(current_point, ts[k], cam_direction)
                            if is_in_medium[k]:
                                assign_3d(dest, ts[k])
                            else:
                                get_intersection_with_borders(current_point, cam_direction, bbox, dest)

                            get_voxel_of_point(dest, grid_shape, bbox, bbox_size, camera_voxel)
                            total_voxels_size += estimate_voxels_size(current_voxel, camera_voxel)
                            # print(a)
                            cos_theta = dot_3d(direction, cam_direction)
                            ISs_mat[k, seg_ind] = (1 / (distance_to_camera ** 2)) * pdf(cos_theta, g) * IS

                    sample_direction(direction, g, new_direction, rng_states, tid)
                    assign_3d(direction, new_direction)

                N_seg = seg + int(in_medium)
                scatter_sizes[tid] = N_seg
                voxel_sizes[tid] = total_voxels_size

        self.render_cuda = render_cuda
        self.render_differentiable_cuda = render_differentiable_cuda
        self.calculate_paths_matrix = calculate_paths_matrix
        self.generate_paths = generate_paths

    def build_paths_list(self, Np, Ns):
        g = self.g
        ts = np.vstack([cam.t.reshape(1, -1) for cam in self.cameras])
        Ps = np.concatenate([cam.P.reshape(1, 3, 4) for cam in self.cameras], axis=0)
        sun_direction = self.sun_direction
        sun_direction[np.abs(sun_direction) < 1e-6] = 0
        # inputs
        dbetas = cuda.to_device(self.volume.betas)
        dbbox = cuda.to_device(self.volume.grid.bbox)
        dbbox_size = cuda.to_device(self.volume.grid.bbox_size)
        dvoxel_size = cuda.to_device(self.volume.grid.voxel_size)
        dsun_direction = cuda.to_device(sun_direction)
        dpixels_shape = cuda.to_device(self.cameras[0].pixels)
        dts = cuda.to_device(ts)
        dPs = cuda.to_device(Ps)
        dis_in_medium = cuda.to_device(self.is_camera_in_medium)


        # outputs
        dstarting_points = cuda.to_device(np.zeros((3, Np), dtype=np.float64))
        dscatter_points = cuda.to_device(np.zeros((3, Ns * Np), dtype=np.float64))
        dscatter_voxels = cuda.to_device(np.zeros((3, Ns * Np), dtype=np.uint8))
        dcamera_pixels = cuda.to_device(np.zeros((2, self.N_cams, Ns * Np), dtype=np.uint8))
        dISs_mat = cuda.to_device(np.zeros((self.N_cams, Ns * Np), dtype=np.float32))
        dscatter_sizes = cuda.to_device(np.zeros(Np, dtype=np.uint8))
        dvoxel_sizes = cuda.to_device(np.zeros(Np, dtype=np.uint32))

        # cuda parameters
        threadsperblock = 256
        blockspergrid = (Np + (threadsperblock - 1)) // threadsperblock
        # seed = np.random.randint(1, int(1e10))
        seed = 12
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=seed)


        # generating paths
        self.generate_paths[blockspergrid, threadsperblock](Np, Ns, dbetas, dbbox, dbbox_size, dvoxel_size,
                                                       dsun_direction, self.N_cams, dpixels_shape, dts, dPs,
                                                       dis_in_medium, g, dscatter_voxels, dstarting_points,
                                                       dscatter_points, dcamera_pixels, dISs_mat, dscatter_sizes,
                                                       dvoxel_sizes, rng_states)

        cuda.synchronize()
        voxel_sizes = dvoxel_sizes.copy_to_host()
        voxel_inds = np.concatenate([np.array([0]), voxel_sizes])
        voxel_inds = np.cumsum(voxel_inds)
        total_num_of_voxels = np.sum(voxel_sizes)
        GB = (total_num_of_voxels * 5 + total_num_of_voxels * 4) / 1e9
        print(f"voxels dataset weights: {GB: .2f} GB")

        dvoxel_inds = cuda.to_device(voxel_inds)
        dvoxels_mat = cuda.to_device(np.ones((total_num_of_voxels, 5), dtype=np.uint8) * 255)
        dlengths = cuda.to_device(np.zeros(total_num_of_voxels, dtype=np.float32))

        self.calculate_paths_matrix[blockspergrid, threadsperblock](Np, Ns, dbetas, dbbox, dbbox_size, dvoxel_size,
                                                               self.N_cams, ts, dis_in_medium, dstarting_points,
                                                               dscatter_points,  dscatter_sizes, dvoxel_inds,
                                                               dcamera_pixels, dvoxels_mat, dlengths)

        cuda.synchronize()
        del(dscatter_points)
        del(dstarting_points)
        del(dvoxel_sizes)
        return dvoxels_mat, dlengths, dISs_mat, dscatter_voxels, dcamera_pixels, dscatter_sizes, dvoxel_inds




    def render(self, cuda_paths, Np, I_gt=None):
        # east declerations
        N_cams = len(self.cameras)
        pixels_shape = self.cameras[0].pixels
        betas = self.volume.betas
        beta_air = self.volume.beta_air
        w0_cloud = self.volume.w0_cloud
        w0_air = self.volume.w0_air

        threadsperblock = 256
        blockspergrid = (Np + (threadsperblock - 1)) // threadsperblock
        self.dbetas.copy_to_device(self.volume.betas)
        self.dI_total.copy_to_device(np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=np.float32))
        self.dtotal_grad.copy_to_device(np.zeros_like(betas, dtype=np.float32))
        self.render_cuda[blockspergrid, threadsperblock](*cuda_paths, self.dbetas, beta_air, w0_cloud, w0_air, self.dI_total)
        cuda.synchronize()
        I_total = self.dI_total.copy_to_host()
        I_total /= Np
        if I_gt is None:
            return I_total

        I_dif = (I_total - I_gt).astype(np.float32)
        self.dI_total.copy_to_device(I_dif)
        self.render_differentiable_cuda[blockspergrid, threadsperblock](*cuda_paths, self.dbetas, beta_air, w0_cloud, w0_air,
                                                                 self.dI_total, self.dtotal_grad)
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
        text += str(self.g) +"  \n\n"
        return text




