import numpy as np
from utils import euler_to_rotmat

class Camera(object):
    def __init__(self, t, euler_angles, focal_length, sensor_size, pixels):
        self.t = t
        self.euler_angles = euler_angles
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.R = euler_to_rotmat(euler_angles)
        self.pixels = pixels
        self.sensor_size = sensor_size
        self.focal_length = focal_length
        s = pixels / sensor_size
        fx, fy = focal_length * s
        cx = pixels[0] / 2
        cy = pixels[1] / 2
        self.K = np.array([[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,   1]])
        self.K_inv = np.linalg.inv(self.K)
        self.G = np.hstack([self.R.T, -self.R.T @ t.reshape(3,1)])
        self.P = self.K @ self.G


    def project_point(self, point):
        point = point.reshape(3,1)
        points_hom = np.r_[point, np.ones((1, 1))]
        points_2d_h = self.P @ points_hom
        points_2d = points_2d_h[:2] / points_2d_h[2]
        points_2d = points_2d.reshape(-1)
        condition = (points_2d >= 0) * (points_2d <= self.pixels)

        points_2d[np.logical_not(condition)] = -1
        points_2d = np.floor(points_2d).astype("int")

        return points_2d
    def update_pixels(self, pixels):
        self.pixels = pixels
        s = pixels / self.sensor_size
        fx, fy = self.focal_length * s
        cx = pixels[0] / 2
        cy = pixels[1] / 2
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)
        self.G = np.hstack([self.R.T, -self.R.T @ self.t.reshape(3, 1)])
        self.P = self.K @ self.G


    def __str__(self):
        return f"t={self.t}, euler={self.euler_angles}, focal_length={self.focal_length}," \
                    f" sensor_size={self.sensor_size}, pixels_size={self.pixels}"






class AirMSPICamera(object):
    def __init__(self, resolution, t, P):
        self.pixels = resolution
        self.t = t
        self.P = P
        # self.camera_array_list = camera_array_list

