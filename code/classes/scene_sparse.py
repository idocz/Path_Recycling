from classes.grid import *
from classes.volume import *
from classes.sparse_path import SparsePath
from utils import  theta_phi_to_direction
from tqdm import tqdm
import math

class SceneSparse(object):
    def __init__(self, volume: Volume, cameras, sun_angles):
        self.volume = volume
        self.sun_angles = sun_angles * (np.pi / 180)
        self.sun_direction = theta_phi_to_direction(*sun_angles)
        self.cameras = cameras
        self.N_cams = len(cameras)
        self.N_pixels = cameras[0].pixels
        self.is_camera_in_medium = np.zeros(self.N_cams, dtype=np.bool)
        for k in range(self.N_cams):
            self.is_camera_in_medium[k] = self.volume.grid.is_in_bbox(self.cameras[k].t)

    def generate_path(self, Ns):
        # easy declarations
        grid = self.volume.grid

        # path list
        ISs = np.zeros(Ns, dtype=np.float64)
        scatter_tensor = np.zeros((3, Ns), dtype=np.int)
        camera_pixels = np.zeros((2, self.N_cams, Ns), dtype=np.int)
        camera_ISs = np.zeros((self.N_cams, Ns), dtype=np.float64)

        # helpful list
        voxels = []
        cam_vec =[]
        seg_vec = []
        lengths = []

        direction =np.copy(self.sun_direction)
        angles = np.empty(2, dtype=np.float64)
        # sample entering point
        start_x = grid.bbox_size[0] * np.random.rand() + grid.bbox[0, 0]
        start_y = grid.bbox_size[1] * np.random.rand() + grid.bbox[1, 0]
        start_z = grid.bbox[2,1]

        current_point = np.array([start_x, start_y, start_z])
        current_voxel = grid.get_voxel_of_point(current_point)
        visible_by_any_camera = False
        for seg in range(Ns):
            p = np.random.rand()
            tau_rand = -math.log(1 - p)
            current_point, current_voxel, in_medium, seg_voxels, seg_lengths, seg_size, beta\
                = self.volume.voxel_traversal_algorithm_save(current_point, current_voxel, direction, tau_rand)
            if in_medium == False:
                break
            voxels.extend(seg_voxels)
            cam_vec.extend([-1]*seg_size)
            seg_vec.extend([seg]*seg_size)
            lengths.extend(seg_lengths)
            scatter_tensor[:,seg] = current_voxel

            # segments_size.append(seg_size)
            ISs[seg] = (1 / (beta * (1 - p)))

            # measuring local estimations to each camera
            for k in range(self.N_cams):
                cam = self.cameras[k]
                pixel = cam.project_point(current_point)
                camera_pixels[:,k,seg] = pixel.reshape(1, -1)
                if (pixel == -1).any():
                    continue
                else:
                    visible_by_any_camera = True
                    cam_direction = cam.t - current_point
                    distance_to_camera = np.linalg.norm(cam_direction)
                    cam_direction /= distance_to_camera
                    if self.is_camera_in_medium[k]:
                        dest = cam.t
                    else:
                        dest = grid.get_intersection_with_borders(current_point, cam_direction)
                    local_est, cam_seg_voxels, cam_seg_lengths = self.volume.local_estimation_save(current_point, current_voxel, cam_direction, dest)
                    cam_seg_size = len(cam_seg_voxels)
                    voxels.extend(cam_seg_voxels)
                    cam_vec.extend([k] * cam_seg_size)
                    seg_vec.extend([seg] * cam_seg_size)
                    lengths.extend(cam_seg_lengths)
                    camera_ISs[k,seg] = (1 / ((distance_to_camera**2) * 4*np.pi))
            cos_theta = (np.random.rand() - 0.5) * 2
            angles[0] = np.arccos(cos_theta)
            angles[1] = np.random.rand() * 2 * np.pi
            direction = theta_phi_to_direction(*angles)

        N_seg = seg + in_medium

        if N_seg == 0 or not visible_by_any_camera:
            return None

        # cut tensors to N_seg
        ISs = ISs[:N_seg]
        ISs = np.cumprod(ISs)
        camera_ISs = camera_ISs[:,:N_seg]
        camera_pixels = camera_pixels[:,:,:N_seg]
        scatter_tensor = scatter_tensor[:,:N_seg]

        # calculate ISs matrix
        ISs_mat = camera_ISs * ISs.reshape(1,-1)

        # reshape
        voxels = np.vstack(voxels)
        lengths = np.array(lengths, dtype=np.float64)
        cam_vec = np.array(cam_vec, dtype=np.int)[:,None]
        seg_vec = np.array(seg_vec, dtype=np.int)[:,None]
        length_inds = np.hstack([voxels, cam_vec, seg_vec])
        path = SparsePath(length_inds, lengths, ISs_mat, scatter_tensor, camera_pixels, N_seg, self.N_cams)
        return path

    def build_paths_list(self, Np, Ns):
        paths = []
        for _ in tqdm(range(Np)):
            paths.append(self.generate_path(Ns))

        print(f"none: {len([path for path in paths if path is None])/Np}")
        return paths



    def render_path(self, path:SparsePath ):
        optical_length = np.zeros((self.N_cams, path.N_seg))
        betas = self.volume.betas

        for row_ind in range(path.lengths.shape[0]):
            i, j, k, cam_ind, seg = path.length_inds[row_ind]
            L = path.lengths[row_ind]
            if cam_ind == -1:
                optical_length[:,seg:] += betas[i,j,k] * L
            else:
                optical_length[cam_ind, seg] += betas[i, j, k] * L

        si = path.scatter_inds
        scatter_cont = (betas[si[0],si[1],si[2]])
        scatter_cont = np.cumprod(scatter_cont).reshape(1,-1)
        res = np.exp(-optical_length) * scatter_cont * path.ISs_mat
        return res

    def update_gradient(self, total_grad, path, res):
        betas = self.volume.betas
        for row_ind in range(path.lengths.shape[0]):
            i, j, k, cam_ind, seg = path.length_inds[row_ind]
            L = path.lengths[row_ind]
            if cam_ind == -1:
                pixel = path.camera_pixels[:,:, seg]
                total_grad[i,j,k,:,pixel[0],pixel[1]] -= L * np.sum(res[cam_ind, seg:])
            else:
                pixel = path.camera_pixels[:, cam_ind, seg]
                total_grad[i, j, k, cam_ind, pixel[0], pixel[1]] -= L * res[cam_ind, seg]

        si = path.scatter_inds
        scatter_beta = (betas[si[0], si[1], si[2]])
        for seg in range(path.N_seg):
            for cam_ind in range(self.N_cams):
                pixel = path.camera_pixels[:, cam_ind, seg:]
                if (pixel != -1).all():
                    total_grad[si[0,seg], si[1,seg], si[2, seg], cam_ind, pixel[0], pixel[1]] += \
                      (scatter_beta[seg] ** (-1)) * np.sum(res[cam_ind, seg:])





    def render(self, paths, differentiable=False):
        # east declerations
        Np = len(paths)
        N_cams = len(self.cameras)
        pixels_shape = self.cameras[0].pixels
        I_total = np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=np.float64)
        if differentiable:
            shape = self.volume.grid.shape
            total_grad = np.zeros((shape[0],shape[1],shape[2],N_cams,pixels_shape[0],pixels_shape[1]), dtype=np.float64)
        for path in tqdm(paths):
            if path is None:
                continue
            else:
                res = self.render_path(path)
                for cam in range(N_cams):
                    for seg in range(path.N_seg):
                        pixel = path.camera_pixels[:, cam, seg]
                        if (pixel != -1).all():
                            cont = res[cam,seg]
                            I_total[cam, pixel[0], pixel[1]] += cont
                if differentiable:
                    self.update_gradient(total_grad, path, res)



        if differentiable:
            return I_total/Np, total_grad/Np
        else:
            return I_total / Np

