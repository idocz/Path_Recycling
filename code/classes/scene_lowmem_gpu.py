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
from time import time
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
threadsperblock = 256


class SceneLowMemGPU(object):
    def __init__(self, volume: Volume, cameras, sun_angles, g_cloud, Ns):
        self.Ns = Ns
        self.volume = volume
        self.sun_angles = sun_angles
        self.sun_direction = theta_phi_to_direction(*sun_angles)
        self.sun_direction[np.abs(self.sun_direction) < 1e-6] = 0
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
        def generate_paths(Np, Ns, beta_cloud, beta_air, g_cloud, bbox, bbox_size, voxel_size, sun_direction, starting_points,
                           scatter_points, scatter_sizes, rng_states):

            tid = cuda.grid(1)
            if tid < Np:
                grid_shape = beta_cloud.shape
                # local memory
                current_voxel = cuda.local.array(3, dtype=np.uint8)
                next_voxel = cuda.local.array(3, dtype=np.uint8)
                current_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                new_direction = cuda.local.array(3, dtype=float_precis)
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
                    scatter_points[0, seg, tid] = current_point[0]
                    scatter_points[1, seg, tid] = current_point[1]
                    scatter_points[2, seg, tid] = current_point[2]

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
                scatter_sizes[tid] = N_seg

        @cuda.jit()
        def post_generation(scatter_points, scatter_inds, Ps, pixel_shape, scatter_points_zipped,
                            pixel_mat):
            tid = cuda.grid(1)
            if tid < scatter_inds.shape[0] - 1:

                scatter_ind = scatter_inds[tid]
                N_seg = scatter_inds[tid+1] - scatter_ind
                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    assign_3d(scatter_points_zipped[:,seg_ind], scatter_points[:,seg,tid])
                    for k in range(N_cams):
                        project_point(scatter_points[:,seg,tid], Ps[k], pixel_shape, pixel_mat[:,k,seg_ind])




        @cuda.jit()
        def render_cuda(beta_cloud, beta_zero, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, N_cams,
                        ts, is_in_medium, starting_points, scatter_points, scatter_inds, pixel_mat,
                        I_total, path_contrib):

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
                cam_direction = cuda.local.array(3, dtype=float_precis)
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
                    assign_3d(next_point, scatter_points[:, seg_ind])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    distance = distance_and_direction(current_point, next_point, direction)
                    if seg > 0:
                        cos_theta = dot_3d(cam_direction, direction)
                        cloud_prob0 = (beta0 - beta_air) / beta0
                        # angle pdf
                        IS *=  cloud_prob * HG_pdf(cos_theta, g_cloud) + (1-cloud_prob) * rayleigh_pdf(cos_theta)
                        IS /=  cloud_prob0 * HG_pdf(cos_theta, g_cloud) + (1-cloud_prob0) * rayleigh_pdf(cos_theta)
                    ###########################################################
                    ############## voxel_fixed traversal_algorithm_save #############

                    path_size = estimate_voxels_size(dest_voxel, current_voxel)

                    tau = 0
                    current_length = 0
                    for pi in range(path_size):
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel)
                        beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                        beta0 = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]]
                        tau += (beta-beta0)*length
                        current_length += length
                        assign_3d(current_voxel, next_voxel)
                    # last step
                    # length = calc_distance(current_point, next_point)
                    length = distance - current_length
                    beta = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
                    beta0 = beta_zero[current_voxel[0], current_voxel[1], current_voxel[2]]
                    tau += (beta - beta0) * length
                    assign_3d(current_point, next_point)
                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################
                    IS *= ((beta/beta0) * math.exp(-tau)) # length pdf
                    beta_c = beta - beta_air
                    cloud_prob = beta_c / beta
                    attenuation *= (cloud_prob * w0_cloud + (1-cloud_prob) * w0_air)
                    for k in range(N_cams):
                        # project_point(current_point, Ps[k], pixels_shape, pixel)
                        pixel[0] = pixel_mat[0, k, seg_ind]
                        if pixel[0] != 255:
                            pixel[1] = pixel_mat[1, k, seg_ind]
                            assign_3d(camera_voxel, current_voxel)
                            assign_3d(camera_point, current_point)
                            distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                            cos_theta = dot_3d(direction, cam_direction)
                            le_pdf = cloud_prob * HG_pdf(cos_theta, g_cloud) + (1-cloud_prob) * rayleigh_pdf(cos_theta)
                            # le_pdf = cloud_prob * dphase_vals[0,k,seg_ind] + (1-cloud_prob) *dphase_vals[1,k,seg_ind]
                            pc = (distance_to_camera ** (-2)) * IS * le_pdf * attenuation
                            if is_in_medium[k]:
                                assign_3d(next_point, ts[k])
                                distance = distance_to_camera
                            else:
                                distance = get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)

                            get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)
                            ###########################################################################
                            ######################## local estimation save ############################
                            tau = 0
                            current_length = 0
                            for pi in range(path_size):
                                length = travel_to_voxels_border(camera_point, camera_voxel, cam_direction,
                                                                 voxel_size, next_voxel)
                                beta_cam = beta_cloud[camera_voxel[0],camera_voxel[1], camera_voxel[2]] + beta_air
                                tau += beta_cam * length
                                current_length += length
                                assign_3d(camera_voxel, next_voxel)
                            # Last Step
                            # length = calc_distance(camera_point, next_point)
                            length = distance - current_length
                            beta_cam = beta_cloud[camera_voxel[0], camera_voxel[1], camera_voxel[2]] + beta_air
                            tau += beta_cam * length


                            ######################## local estimation save ############################
                            ###########################################################################
                            pc *= math.exp(-tau)
                            path_contrib[k,seg_ind] = pc
                            cuda.atomic.add(I_total, (k, pixel[0], pixel[1]), pc)

                    assign_3d(cam_direction, direction) #using cam direction as temp direction


        @cuda.jit()
        def calc_gradient_contribution(path_contrib, I_diff, pixel_mat, scatter_inds, grad_contrib):
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
                            pixel[0] = pixel_mat[0,k, sub_seg_ind]
                            if pixel[0] != 255:
                                pixel[1] = pixel_mat[1,k, sub_seg_ind]
                                grad_contrib[seg_ind] += path_contrib[k, sub_seg_ind] * I_diff[k,pixel[0],pixel[1]]


        @cuda.jit()
        def render_differentiable_cuda(beta_cloud, beta_air, g_cloud, w0_cloud, w0_air, bbox, bbox_size, voxel_size, N_cams,
                        ts, is_in_medium, starting_points, scatter_points, scatter_inds, pixel_mat,
                                       I_diff, path_contrib, grad_contrib, cloud_mask, total_grad):
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
                next_point = cuda.local.array(3, dtype=float_precis)
                direction = cuda.local.array(3, dtype=float_precis)
                cam_direction = cuda.local.array(3, dtype=float_precis)
                # dest = cuda.local.array(3, dtype=float_precis)
                pixel = cuda.local.array(2, dtype=np.uint8)
                assign_3d(current_point, starting_points[:, tid])
                get_voxel_of_point(current_point, grid_shape, bbox, bbox_size, current_voxel)

                beta_c = 1.0  # for type decleration
                le_contrib = 1.0  # for type decleration

                for seg in range(N_seg):
                    seg_ind = seg + scatter_ind
                    assign_3d(next_point, scatter_points[:, seg_ind])
                    get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                    distance = distance_and_direction(current_point, next_point, direction)
                    # GRAD CALCULATION (SCATTERING)
                    if seg > 0:
                        if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                            cos_theta_scatter = dot_3d(cam_direction, direction)
                            seg_contrib = (beta_c + (rayleigh_pdf(cos_theta_scatter)/HG_pdf(cos_theta_scatter,g_cloud)) * beta_air ) **(-1)
                            seg_contrib += le_contrib
                            grad = seg_contrib * grad_contrib[seg_ind]
                            cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]), grad)

                    ###########################################################
                    ############## voxel_fixed traversal_algorithm_save #############
                    path_size = estimate_voxels_size(dest_voxel, current_voxel)
                    current_length = 0
                    for pi in range(path_size):
                        length = travel_to_voxels_border(current_point, current_voxel, direction, voxel_size, next_voxel)
                        current_length += length
                        if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                            grad = -length * grad_contrib[seg_ind]
                            cuda.atomic.add(total_grad, (current_voxel[0],current_voxel[1],current_voxel[2]), grad)
                        assign_3d(current_voxel,next_voxel)

                    # last step
                    # length = calc_distance(current_point, next_point)
                    length = distance - current_length
                    if cloud_mask[current_voxel[0], current_voxel[1], current_voxel[2]]:
                        grad = -length * grad_contrib[seg_ind]
                        cuda.atomic.add(total_grad, (current_voxel[0], current_voxel[1], current_voxel[2]), grad)
                    assign_3d(current_point, next_point)

                    ######################## voxel_fixed_traversal_algorithm_save ###################
                    ###########################################################################
                    beta_c = beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]] + divide_beta_eps
                    le_contrib = (beta_c + (w0_air/w0_cloud) * beta_air) **(-1)
                    le_contrib -= 1 / (beta_c + beta_air)
                    for k in range(N_cams):
                        # project_point(current_point, Ps[k], pixels_shape, pixel)
                        pixel[0] = pixel_mat[0, k, seg_ind]
                        pixel[1] = pixel_mat[1, k, seg_ind]
                        if pixel[0]:
                            assign_3d(camera_voxel, current_voxel)
                            assign_3d(camera_point, current_point)
                            distance_to_camera = distance_and_direction(camera_point, ts[k], cam_direction)
                            cos_theta = dot_3d(direction, cam_direction)

                            if is_in_medium[k]:
                                assign_3d(next_point, ts[k])
                                distance = distance_to_camera
                            else:
                                distance = get_intersection_with_borders(camera_point, cam_direction, bbox, next_point)
                            get_voxel_of_point(next_point, grid_shape, bbox, bbox_size, dest_voxel)
                            path_size = estimate_voxels_size(dest_voxel, current_voxel)

                            # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                            if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]]:
                                seg_contrib = le_contrib + (beta_c + (rayleigh_pdf(cos_theta) / HG_pdf(cos_theta, g_cloud)) * beta_air) ** (-1)
                                grad = seg_contrib * path_contrib[k, seg_ind] * I_diff[k, pixel[0], pixel[1]]
                                cuda.atomic.add(total_grad, (camera_voxel[0],camera_voxel[1],camera_voxel[2]), grad)
                            ###########################################################################
                            ######################## local estimation save ############################
                            current_length = 0
                            for pi in range(path_size):
                                length = travel_to_voxels_border(camera_point, camera_voxel, cam_direction,
                                                                     voxel_size, next_voxel)
                                current_length += length
                                # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                                if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]]:
                                    grad = -length * path_contrib[k, seg_ind] * I_diff[k, pixel[0], pixel[1]]
                                    cuda.atomic.add(total_grad,(camera_voxel[0], camera_voxel[1], camera_voxel[2]), grad)
                                assign_3d(camera_voxel, next_voxel)
                            # Last Step
                            # length = calc_distance(camera_point, next_point)
                            length = distance - current_length
                            # GRAD CALCULATION (LOCAL ESTIMATION DERIVATIVE)
                            if cloud_mask[camera_voxel[0], camera_voxel[1], camera_voxel[2]]:
                                grad = -length * path_contrib[k, seg_ind] * I_diff[k, pixel[0], pixel[1]]
                                cuda.atomic.add(total_grad, (camera_voxel[0], camera_voxel[1], camera_voxel[2]), grad)
                            ######################## local estimation save ############################
                            ###########################################################################

                    assign_3d(cam_direction, direction)  # using cam direction as temp direction

        self.generate_paths = generate_paths
        self.post_generation = post_generation
        self.render_cuda = render_cuda
        self.calc_gradient_contribution = calc_gradient_contribution
        self.render_differentiable_cuda = render_differentiable_cuda

    def init_cuda_param(self, Np, init=False):
        self.threadsperblock = threadsperblock
        self.blockspergrid = (Np + (threadsperblock - 1)) // threadsperblock
        if init:
            self.seed = np.random.randint(1, int(1e10))
            self.rng_states = create_xoroshiro128p_states(threadsperblock * self.blockspergrid, seed=self.seed)

    def build_paths_list(self, Np, Ns):
        # inputs
        self.dbeta_zero.copy_to_device(self.volume.beta_cloud)
        beta_air = self.volume.beta_air
        # outputs
        print(f"preallocation weights: {3*(Ns+1)*Np*precis_size/1e9: .2f} GB")
        dstarting_points = cuda.to_device(np.zeros((3, Np), dtype=float_precis))
        dscatter_points = cuda.to_device(np.zeros((3, Ns, Np), dtype=float_precis))
        dscatter_sizes = cuda.to_device(np.zeros(Np, dtype=np.uint8))

        # cuda parameters
        self.init_cuda_param(Np)
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid

        start = time()
        self.generate_paths[blockspergrid, threadsperblock]\
            (Np, Ns, self.dbeta_zero, beta_air, self.g_cloud,  self.dbbox, self.dbbox_size, self.dvoxel_size,
            self.dsun_direction, dstarting_points, dscatter_points, dscatter_sizes, self.rng_states)

        cuda.synchronize()
        print("generate_paths took:", time() - start)


        start = time()
        scatter_sizes = dscatter_sizes.copy_to_host()
        starting_points = dstarting_points.copy_to_host()
        scatter_points = dscatter_points.copy_to_host()
        del(dscatter_sizes)
        del(dstarting_points)
        del(dscatter_points)
        active_paths = scatter_sizes != 0
        scatter_sizes = scatter_sizes[active_paths]
        starting_points = starting_points[:, active_paths]
        scatter_points = np.ascontiguousarray(scatter_points[:, :, active_paths])

        total_num_of_scatter = np.sum(scatter_sizes)
        print(f"total_num_of_scatter={total_num_of_scatter}")
        scatter_inds = np.concatenate([np.array([0]), scatter_sizes])
        scatter_inds = np.cumsum(scatter_inds)
        Np_nonan = int(np.sum(active_paths))
        print(f"not none paths={Np_nonan / Np}")

        dscatter_points = cuda.to_device(scatter_points)
        dscatter_inds = cuda.to_device(scatter_inds)
        dscatter_points_zipped = cuda.to_device(np.zeros((3, total_num_of_scatter), dtype=float_precis))
        dpixel_mat = cuda.to_device(np.zeros((2, self.N_cams, total_num_of_scatter), dtype=np.uint8))
        threadsperblock = self.threadsperblock
        blockspergrid = self.blockspergrid
        print("middle took:",time()-start)
        middle_alloc = (3*(total_num_of_scatter+Np_nonan) + 3*(Ns+1)*Np_nonan)* precis_size +  \
                       2*self.N_cams*total_num_of_scatter + Np_nonan*4
        middle_alloc /= 1e9
        print(f"middle allocation weights: {middle_alloc: .2f} GB")
        start = time()

        dstarting_points = cuda.to_device(starting_points)

        self.post_generation[blockspergrid, threadsperblock]\
            (dscatter_points, dscatter_inds, self.dPs, self.dpixels_shape,
             dscatter_points_zipped, dpixel_mat)
        cuda.synchronize()
        del(dscatter_points)
        post_alloc = 3*(total_num_of_scatter+Np_nonan) * precis_size + 2*self.N_cams*total_num_of_scatter + Np_nonan*4
        post_alloc /= 1e9
        print(f"post allocation weights: {post_alloc: .2f} GB")
        self.init_cuda_param(Np_nonan)
        print("post_generation took:", time() - start)
        self.total_num_of_scatter = total_num_of_scatter
        self.Np_nonan = Np_nonan
        del(self.dpath_contrib)
        del(self.dgrad_contrib)
        self.dpath_contrib = cuda.to_device(np.zeros((self.N_cams, self.total_num_of_scatter), dtype=float_reg))
        self.dgrad_contrib = cuda.to_device(np.zeros(self.total_num_of_scatter, dtype=float_reg))
        return dstarting_points, dscatter_points_zipped, dscatter_inds, dpixel_mat



    def render(self, cuda_paths, Np, I_gt=None):
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

        self.dpath_contrib = cuda.to_device(np.zeros((self.N_cams, self.total_num_of_scatter), dtype=float_reg))
        self.dgrad_contrib = cuda.to_device(np.zeros(self.total_num_of_scatter, dtype=float_reg))


        self.render_cuda[blockspergrid, threadsperblock] \
            (self.dbeta_cloud, self.dbeta_zero, beta_air, g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size,
             self.dvoxel_size, N_cams, self.dts, self.dis_in_medium, *cuda_paths,self.dI_total,
             self.dpath_contrib)

        cuda.synchronize()
        I_total = self.dI_total.copy_to_host()
        I_total /= Np
        print("render_cuda took:",time() - start)
        # return I_total
        if I_gt is None:
            return I_total

        ##### differentiable part ####
        self.dtotal_grad.copy_to_device(np.zeros_like(self.volume.beta_cloud, dtype=float_reg))
        # precalculating gradient contributions
        I_dif = (I_total - I_gt).astype(float_reg)
        self.dI_total.copy_to_device(I_dif)

        dpixel_mat = cuda_paths[-1]
        dscatter_inds = cuda_paths[-2]
        start = time()
        self.calc_gradient_contribution[blockspergrid, threadsperblock]\
            (self.dpath_contrib, self.dI_total, dpixel_mat,dscatter_inds, self.dgrad_contrib)

        cuda.synchronize()
        print("calc_gradient_contribution took:",time()-start)
        start = time()
        self.render_differentiable_cuda[blockspergrid, threadsperblock]\
            (self.dbeta_cloud, beta_air, g_cloud, w0_cloud, w0_air, self.dbbox, self.dbbox_size, self.dvoxel_size,
                                       N_cams, self.dts, self.dis_in_medium, *cuda_paths, self.dI_total, self.dpath_contrib,
             self.dgrad_contrib, self.dcloud_mask, self.dtotal_grad)

        cuda.synchronize()
        print("render_differentiable_cuda took:", time() - start)
        total_grad = self.dtotal_grad.copy_to_host()
        total_grad /= (Np * N_cams)
        return I_total, total_grad



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




