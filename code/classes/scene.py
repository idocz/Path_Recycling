from classes.grid import *
from classes.volume import *
from utils import  theta_phi_to_direction, angles_between_vectors
from tqdm import tqdm
import multiprocessing as mp
class Scene(object):
    def __init__(self, volume: Volume, cameras, sun_angles, phase_function):
        self.volume = volume
        self.sun_angles = sun_angles
        self.sun_direction = theta_phi_to_direction(*sun_angles)
        self.cameras = cameras
        self.phase_function = phase_function
        self.N_cams = len(self.cameras)
        self.pixels = self.cameras[0].pixels
        self.is_camera_in_medium = np.zeros(self.N_cams, dtype=np.bool)
        for k in range(self.N_cams):
            self.is_camera_in_medium[k] = self.volume.grid.is_in_bbox(self.cameras[k].t)

    def render(self, Np, Ns):
        I_total = np.zeros((self.N_cams, self.pixels[0], self.pixels[1]), dtype=np.float64)
        counter = 0
        for _ in tqdm(range(Np)):

            I_path = self.render_path(Ns)
            I_total += I_path
            if (I_path == 0).all():
                counter += 1
        print(f"none = {counter/Np}")
        return I_total / Np

    def render_differentiable(self, Np, Ns):
        I_total = np.zeros((self.N_cams, self.pixels[0], self.pixels[1]), dtype=np.float64)
        total_grad = np.zeros((*self.volume.grid.shape,self.N_cams, self.pixels[0], self.pixels[1]), dtype=np.float64)
        counter = 0
        for _ in tqdm(range(Np)):

            I_path, path_grad = self.render_path_differentiable(Ns)
            I_total += I_path
            total_grad += path_grad
            if (I_path == 0).all():
                counter += 1
        print(f"none = {counter/Np}")
        return I_total / Np, total_grad / Np



    def render_path(self, Ns):
        random = np.random.RandomState()
        I_total = np.zeros((self.N_cams, self.pixels[0], self.pixels[1]))
        grid = self.volume.grid
        # sample entrance point ( z axis is the TOA)
        start_x = grid.bbox_size[0] * random.rand() + grid.bbox[0, 0]
        start_y = grid.bbox_size[1] * random.rand() + grid.bbox[1, 0]
        start_z = grid.bbox[2, 1]

        # init path variable
        current_point = np.array([start_x, start_y, start_z], dtype=np.float64)
        current_voxel = grid.get_voxel_of_point(current_point)
        direction = self.sun_direction
        attenuation = 1
        # I_total_path = np.zeros_like(I_total)
        for _ in range(Ns):
            p = random.rand()
            tau_rand = -np.log(1 - p)
            current_point, current_voxel, in_medium \
                = self.volume.voxel_traversal_algorithm(current_point, current_voxel, direction, tau_rand)

            if in_medium == False:
                break
            beta_c = self.volume.beta_cloud[current_voxel[0],current_voxel[1], current_voxel[2]]
            scatter_prob = (self.volume.w0_cloud * beta_c + self.volume.w0_air * self.volume.beta_air) /\
                           (self.volume.beta_air + beta_c)

            attenuation *= scatter_prob
            for k in range(self.N_cams):
                cam = self.cameras[k]
                pixel = cam.project_point(current_point)
                # if (pixel == -1).any():
                #     print("bug")
                #     continue
                cam_direction = cam.t - current_point
                distance_to_camera = np.linalg.norm(cam_direction)
                cam_direction /= distance_to_camera
                if self.is_camera_in_medium[k]:
                    dest = cam.t
                else:
                    dest = grid.get_intersection_with_borders(current_point, cam_direction)

                # distance_to_camera = 1
                local_est = self.volume.local_estimation(current_point, current_voxel, cam_direction, dest)
                local_est *= attenuation
                cos_theta = np.dot(direction, cam_direction)
                I_total[k, pixel[0], pixel[1]] += (local_est / (distance_to_camera ** 2)) * self.phase_function.pdf(cos_theta)
            direction = self.phase_function.sample_direction(direction)

        return I_total

    def render_path_differentiable(self, Ns):
        random = np.random.RandomState()
        I_total = np.zeros((self.N_cams, self.pixels[0], self.pixels[1]))
        grid = self.volume.grid
        total_grad = np.zeros((*grid.shape,self.N_cams, self.pixels[0], self.pixels[1]))
        path_grad = np.zeros(grid.shape)
        attenuation = 1
        # sample entrance point ( z axis is the TOA)
        start_x = grid.bbox_size[0] * random.rand() + grid.bbox[0, 0]
        start_y = grid.bbox_size[1] * random.rand() + grid.bbox[1, 0]
        start_z = grid.bbox[2, 1]

        # init path variable
        current_point = np.array([start_x, start_y, start_z], dtype=np.float64)
        current_voxel = grid.get_voxel_of_point(current_point)
        direction = self.sun_direction
        # I_total_path = np.zeros_like(I_total)
        for _ in range(Ns):
            p = random.rand()
            tau_rand = -np.log(1 - p)
            current_point, current_voxel, in_medium, seg_voxels, seg_lengths, seg_size, beta\
                = self.volume.voxel_traversal_algorithm_save(current_point, current_voxel, direction, tau_rand)
            if in_medium == False:
                break

            beta_c = self.volume.beta_cloud[current_voxel[0], current_voxel[1], current_voxel[2]]
            beta_scatter = self.volume.w0_cloud * beta_c + self.volume.w0_air * self.volume.beta_air
            scatter_prob = beta_scatter/(self.volume.beta_air + beta_c)
            attenuation *= scatter_prob
            path_grad[seg_voxels[:,0], seg_voxels[:,1], seg_voxels[:,2]] -= seg_lengths
            path_grad[current_voxel[0], current_voxel[1], current_voxel[2] ] += (self.volume.w0_cloud/beta_scatter) # make sure is correct
            for k in range(self.N_cams):
                cam = self.cameras[k]
                pixel = cam.project_point(current_point)
                if (pixel == -1).any():
                    continue
                cam_direction = cam.t - current_point
                distance_to_camera = np.linalg.norm(cam_direction)
                cam_direction /= distance_to_camera
                if self.is_camera_in_medium[k]:
                    dest = cam.t
                else:
                    dest = grid.get_intersection_with_borders(current_point, cam_direction)

                # distance_to_camera = 1
                local_est, seg_voxels, seg_lengths = self.volume.local_estimation_save(current_point, current_voxel, cam_direction, dest)
                cos_theta = np.dot(direction, cam_direction)
                segment_contrib = attenuation * (local_est / (distance_to_camera ** 2)) * self.phase_function.pdf(cos_theta)
                I_total[k, pixel[0], pixel[1]] += segment_contrib
                total_grad[:,:,:,k,pixel[0],pixel[1]] += segment_contrib * path_grad
                # camera gradients ( cant add to the general grad variable)
                total_grad[seg_voxels[:,0], seg_voxels[:,1], seg_voxels[:,2], k, pixel[0], pixel[1]] -=\
                    seg_lengths * segment_contrib
            direction = self.phase_function.sample_direction(direction)


        return I_total, total_grad

