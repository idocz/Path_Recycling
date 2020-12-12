from classes.grid import *
from classes.volume import *
from classes.path import Path
from utils import  theta_phi_to_direction
from tqdm import tqdm
import math

class SceneGraph(object):
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
        voxels = []
        lengths = []
        segments_size = []
        ISs = []
        camera_voxels = []
        camera_lengths = []
        camera_segments_size = []
        camera_ISs = []
        camera_pixels = []

        direction =np.copy(self.sun_direction)
        angles = np.empty(2, dtype=np.float64)
        # sample entering point
        start_x = grid.bbox_size[0] * np.random.rand() + grid.bbox[0, 0]
        start_y = grid.bbox_size[1] * np.random.rand() + grid.bbox[1, 0]
        start_z = grid.bbox[2,1]

        current_point = np.array([start_x, start_y, start_z])
        current_voxel = grid.get_voxel_of_point(current_point)
        visible_by_any_camera = False
        for _ in range(Ns):
            p = np.random.rand()
            tau_rand = -math.log(1 - p)
            current_point, current_voxel, in_medium, seg_voxels, seg_lengths, seg_size, beta\
                = self.volume.voxel_traversal_algorithm_save(current_point, current_voxel, direction, tau_rand)
            if in_medium == False:
                break
            voxels.extend(seg_voxels)
            lengths.extend(seg_lengths)
            segments_size.append(seg_size)
            ISs.append(1 / (beta * (1 - p)))

            # measuring local estimations to each camera
            for k in range(self.N_cams):
                cam = self.cameras[k]
                pixel = cam.project_point(current_point)
                camera_pixels.append(pixel.reshape(1, -1))
                if (pixel == -1).any():
                    camera_segments_size.append(0)
                    camera_ISs.append(0)
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
                    camera_voxels.extend(cam_seg_voxels)
                    camera_lengths.extend(cam_seg_lengths)
                    camera_ISs.append(1 / ((distance_to_camera**2) * 4*np.pi))
                    camera_segments_size.append(len(cam_seg_voxels))
            cos_theta = (np.random.rand() - 0.5) * 2
            angles[0] = np.arccos(cos_theta)
            angles[1] = np.random.rand() * 2 * np.pi
            direction = theta_phi_to_direction(*angles)

        N_seg = len(segments_size)
        if N_seg == 0 or not visible_by_any_camera:
            return None

        # reshape

        path = Path(voxels, lengths, segments_size, ISs, camera_voxels, camera_lengths, camera_segments_size,
                        camera_ISs, camera_pixels, self.N_cams, N_seg)
        path.to_array()
        return path

    def build_paths_list(self, Np, Ns):
        paths = []
        for _ in tqdm(range(Np)):
            paths.append(self.generate_path(Ns))
        return paths

    def calc_path_contribution_vector(self,  voxels, lengths, segments_size, ISs):
        cont_vec = np.empty((segments_size.shape[0], 1), dtype=np.float64)
        counter = 0
        i = 0
        betas_1d = self.volume.betas[voxels[:,0], voxels[:,1], voxels[:,2]]
        for seg_size in segments_size:
            seg_lengths = lengths[counter:counter + seg_size]
            seg_betas = betas_1d[counter:counter + seg_size]
            beta = seg_betas[seg_size - 1]
            cont_vec[i, 0] = beta * np.exp(-np.sum(seg_betas * seg_lengths)) * ISs[i, 0]
            if i != 0:
                cont_vec[i, 0] *= cont_vec[i - 1, 0]
            counter += seg_size
            i += 1
        return cont_vec

    def calc_camera_contribution_matrix(self, camera_voxels, camera_lengths, camera_segments_size, N_seg):
        cont_mat = np.empty((N_seg, self.N_cams), dtype=np.float64)  # TODO:check for bugs
        betas_1d = self.volume.betas[camera_voxels[:,0], camera_voxels[:,1], camera_voxels[:,2]]
        counter = 0
        j = 0
        for i in range(N_seg):
            for k in range(self.N_cams):
                cam_seg_size = camera_segments_size[j]

                if cam_seg_size == 0:
                    cont_mat[i,k] = 0
                else:
                    seg_lengths = camera_lengths[counter:counter + cam_seg_size]
                    seg_betas = betas_1d[counter: counter + cam_seg_size]
                    cont_mat[i, k] = np.exp(-np.sum(seg_betas * seg_lengths))
                    counter += cam_seg_size
                j += 1
        return cont_mat

    def render_path(self, path ):
        I_total = np.zeros((self.N_cams, self.N_pixels[0], self.N_pixels[1]), dtype=np.float64)
        path_cont_vec = self.calc_path_contribution_vector(path.voxels, path.lengths, path.segments_size, path.ISs)
        N_seg = path.segments_size.shape[0]
        camera_cont_mat = self.calc_camera_contribution_matrix(path.camera_voxels, path.camera_lengths,
                                                          path.camera_segments_size, N_seg)
        camera_cont_mat *= path.camera_ISs
        camera_cont_mat *= path_cont_vec
        for k in range(self.N_cams):
            pixels = path.camera_pixels[:, k, :]
            for i, pixel in enumerate(pixels):
                I_total[k, pixel[0], pixel[1]] += camera_cont_mat[i, k]
        return I_total

    def render(self, paths):
        # east declerations
        Np = len(paths)
        N_cams = len(self.cameras)
        pixels_shape = self.cameras[0].pixels
        I_total = np.zeros((N_cams, pixels_shape[0], pixels_shape[1]), dtype=np.float64)
        counter = 0
        for path in tqdm(paths):
            if path is None:
                counter += 1
                continue
            I_total += self.render_path(path)
        print(f"none = {counter/Np}")
        return I_total / Np

    def render_differentiable(self, paths):
        pass

