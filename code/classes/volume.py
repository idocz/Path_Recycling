import numpy as np
from classes.grid import *

class Volume(object):
    def __init__(self, grid:Grid, beta_cloud, beta_air, w0_cloud, w0_air):
        self.grid = grid
        self.beta_cloud = beta_cloud
        self.beta_air = beta_air
        self.betas = beta_cloud + beta_air
        self.w0_cloud = w0_cloud
        self.w0_air = w0_air
        self.cloud_mask = beta_cloud >= 0
    def set_mask(self, mask):
        self.cloud_mask = mask

    def set_beta_cloud(self, beta_cloud):
        self.beta_cloud = beta_cloud
        self.betas = self.beta_cloud + self.beta_air

    def voxel_traversal_algorithm(self, start_point, current_voxel, direction, tau_rand):
        # easy assignment
        grid = self.grid

        current_tau = 0.0
        next_voxel = np.copy(current_voxel)
        current_point = np.copy(start_point)
        while True:
            if current_voxel[0] < 0 or current_voxel[0] >= grid.shape[0] or current_voxel[1] < 0 or current_voxel[1] >= \
                    grid.shape[1] or current_voxel[2] < 0 or current_voxel[2] >= grid.shape[2]:
                in_medium = False
                break
            beta = self.betas[current_voxel[0], current_voxel[1], current_voxel[2]]
            length, next_voxel, next_point, _ = grid.travel_to_voxels_border(current_point, direction, next_voxel)
            current_tau += length * beta
            # seg_size += 1
            if current_tau >= tau_rand:
                step_back = (current_tau - tau_rand) / beta
                current_point = next_point - step_back * direction
                in_medium = True
                break

            # update current voxel and point
            current_voxel = next_voxel
            current_point = next_point

        return current_point, current_voxel, in_medium

    def voxel_traversal_algorithm_save(self, start_point, current_voxel, direction, tau_rand):
        # easy assignment
        grid = self.grid

        current_tau = 0.0
        next_voxel = np.copy(current_voxel)
        seg_voxels = []
        seg_lengths = []
        seg_size = 0
        beta = 0
        current_point = np.copy(start_point)
        while True:
            if current_voxel[0] < 0 or current_voxel[0] >= grid.shape[0] or current_voxel[1] < 0 or current_voxel[1] >= \
                    grid.shape[1] or current_voxel[2] < 0 or current_voxel[2] >= grid.shape[2]:
                in_medium = False
                break
            beta = self.betas[current_voxel[0], current_voxel[1], current_voxel[2]]
            seg_voxels.append(np.copy(current_voxel).reshape(1, 3))
            length, next_voxel, next_point, _ = grid.travel_to_voxels_border(current_point, direction, next_voxel)
            seg_lengths.append(length)
            current_tau += length * beta
            seg_size += 1
            if current_tau >= tau_rand:
                step_back = (current_tau - tau_rand) / beta
                seg_lengths[-1] -= step_back
                current_point = next_point - step_back * direction
                in_medium = True
                break

            # update current voxel and point
            current_voxel = next_voxel
            current_point = next_point

        seg_voxels = np.vstack(seg_voxels)
        seg_lengths = np.array(seg_lengths)
        return current_point, current_voxel, in_medium, seg_voxels, seg_lengths, seg_size, beta



    def local_estimation(self, source_point, source_voxel, camera_direction,  dest):
        # easy decleration
        grid = self.grid

        current_voxel = np.copy(source_voxel)
        current_point = np.copy(source_point)
        distance = np.linalg.norm(source_point - dest)
        current_distance = 0.0
        tau = 0.0
        while True:
            if current_voxel[0] < 0 or current_voxel[0] >= grid.shape[0] or current_voxel[1] < 0 or current_voxel[1] >= \
                    grid.shape[1] or current_voxel[2] < 0 or current_voxel[2] >= grid.shape[2]:
                break
            beta = self.betas[current_voxel[0], current_voxel[1], current_voxel[2]]
            length, next_voxel, next_point, _ = grid.travel_to_voxels_border(current_point, camera_direction, current_voxel)
            tau += length * beta
            current_distance += length
            if current_distance >= distance:
                step_back = current_distance - distance
                tau -= step_back * beta
                length -= step_back
                current_point = next_point - step_back * camera_direction
                break
            current_point = next_point
            current_voxel = next_voxel
        local_est = np.exp(-tau)
        return local_est


    def local_estimation_save(self, source_point, source_voxel, camera_direction,  dest):
        # easy decleration
        grid = self.grid

        current_voxel = np.copy(source_voxel)
        current_point = np.copy(source_point)
        seg_voxels = []
        seg_lengths = []
        distance = np.linalg.norm(source_point - dest)
        current_distance = 0.0
        tau = 0.0
        while True:
            if current_voxel[0] < 0 or current_voxel[0] >= grid.shape[0] or current_voxel[1] < 0 or current_voxel[1] >= \
                    grid.shape[1] or current_voxel[2] < 0 or current_voxel[2] >= grid.shape[2]:
                break
            seg_voxels.append(np.copy(current_voxel).reshape(1, 3))
            beta = self.betas[current_voxel[0], current_voxel[1], current_voxel[2]]
            length, next_voxel, next_point, _ = grid.travel_to_voxels_border(current_point, camera_direction, current_voxel)
            seg_lengths.append(length)
            tau += length * beta
            current_distance += length
            if current_distance >= distance:
                step_back = current_distance - distance
                tau -= step_back * beta
                seg_lengths[-1] -= step_back
                current_point = next_point - step_back * camera_direction
                break
            current_point = next_point
            current_voxel = next_voxel
        local_est = np.exp(-tau)
        seg_voxels = np.vstack(seg_voxels)
        seg_lengths = np.array(seg_lengths)
        return local_est, seg_voxels, seg_lengths

