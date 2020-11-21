from classes.grid import *
from classes.volume import *
from utils import  theta_phi_to_direction
from tqdm import tqdm
import multiprocessing as mp
class Scene(object):
    def __init__(self, volume: Volume, cameras, sun_angles):
        self.volume = volume
        self.sun_angles = sun_angles * (np.pi / 180)
        self.sun_direction = theta_phi_to_direction(*sun_angles)
        self.cameras = cameras
        self.N_cams = len(self.cameras)
        self.pixels = self.cameras[0].pixels
        self.is_camera_in_medium = np.zeros(self.N_cams, dtype=np.bool)
        for k in range(self.N_cams):
            self.is_camera_in_medium[k] = self.volume.grid.is_in_bbox(self.cameras[k].t)

    def render(self, Np, Ns):
        I_total = np.zeros((self.N_cams, self.pixels[0], self.pixels[1]), dtype=np.float64)

        counter = 0
        for _ in tqdm(range(Np)):
            res = self.render_path(Ns)
            I_total += res
            if (res == 0).all():
                counter += 1
        print(f"none = {counter/Np}")



    def render_path(self, Ns):
        random = np.randself.RandomState()
        angles = np.empty(2, dtype=np.float64)
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
        # I_total_path = np.zeros_like(I_total)
        for _ in range(Ns):
            p = random.rand()
            tau_rand = -np.log(1 - p)
            current_point, current_voxel, in_medium \
                = self.volume.voxel_traversal_algorithm(current_point, current_voxel, direction, tau_rand)

            if in_medium == False:
                break

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
                local_est = self.volume.local_estimation(current_point, current_voxel, cam_direction, dest)
                I_total[k, pixel[0], pixel[1]] += local_est / ((distance_to_camera ** 2) * 4*np.pi)
            cos_theta = (np.random.rand() - 0.5) * 2
            angles[0] = np.arccos(cos_theta)
            angles[1] = np.random.rand() * 2 * np.pi
            direction = theta_phi_to_direction(*angles)

        return I_total


def render_path_differentiable(args):
    scene, Ns = args
    random = np.random.RandomState()
    angles = np.empty(2, dtype=np.float64)
    I_total = np.zeros((scene.N_cams, scene.pixels[0], scene.pixels[1]))
    total_grad = np.zeros((scene.N_cams, scene.pixels[0], scene.pixels[1]))
    grid = scene.volume.grid
    # sample entrance point ( z axis is the TOA)
    start_x = grid.bbox_size[0] * random.rand() + grid.bbox[0, 0]
    start_y = grid.bbox_size[1] * random.rand() + grid.bbox[1, 0]
    start_z = grid.bbox[2, 1]

    # init path variable
    current_point = np.array([start_x, start_y, start_z], dtype=np.float64)
    current_voxel = grid.get_voxel_of_point(current_point)
    direction = scene.sun_direction
    # I_total_path = np.zeros_like(I_total)
    for _ in range(Ns):
        p = random.rand()
        tau_rand = -np.log(1 - p)
        current_point, current_voxel, in_medium, seg_voxels, seg_lengths, seg_size, beta \
            = scene.volume.voxel_traversal_algorithm_save(current_point, current_voxel, direction, tau_rand)

        if in_medium == False:
            break

        for k in range(scene.N_cams):
            cam = scene.cameras[k]
            pixel = cam.project_point(current_point)
            if (pixel == -1).any():
                continue
            cam_direction = cam.t - current_point
            distance_to_camera = np.linalg.norm(cam_direction)
            cam_direction /= distance_to_camera
            if scene.is_camera_in_medium[k]:
                dest = cam.t
            else:
                dest = grid.get_intersection_with_borders(current_point, cam_direction)

            # distance_to_camera = 1
            local_est = scene.volume.local_estimation(current_point, current_voxel, cam_direction, dest)
            I_total[k, pixel[0], pixel[1]] += local_est / ((distance_to_camera ** 2) * 4*np.pi)
        cos_theta = (np.random.rand() - 0.5) * 2
        angles[0] = np.arccos(cos_theta)
        angles[1] = np.random.rand() * 2 * np.pi
        direction = theta_phi_to_direction(*angles)

    return I_total
