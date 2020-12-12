import numpy as np
from numba import jit


# 2D grid class
class Grid(object):
    def __init__(self, bbox, shape):
        assert len(shape) == 3, f"This grid class is for 2d implementation ({shape.shape}-d was given)"
        assert bbox.shape == (3,2), f"bbox should be of shape (2,2) and not {bbox.shape}"

        # bbox should be: [[min-x, max-x,
        #                  [min-y, max-y],
        #                  [min-z, max-z]

        self.bbox = bbox
        self.shape = shape
        self.bbox_size = bbox[:,1] - bbox[:,0]
        self.voxel_size = self.bbox_size / np.array(shape)

        self.x_axis = np.linspace(bbox[0,0], bbox[0,1], shape[0])
        self.y_axis = np.linspace(bbox[1,0], bbox[1,1], shape[1])
        self.z_axis = np.linspace(bbox[2,0], bbox[2,1], shape[2])

    def get_voxel_of_point(self, point):
        if point[0] == self.bbox[0, 1]:
            x = self.shape[0] - 1
        else:
            x = int(((point[0] - self.bbox[0, 0]) / self.bbox_size[0]) * self.shape[0])

        if point[1] == self.bbox[1, 1]:
            y = self.shape[1] - 1
        else:
            y = int(((point[1] - self.bbox[1, 0]) / self.bbox_size[1]) * self.shape[1])

        if point[2] == self.bbox[2, 1]:
            z = self.shape[2] - 1
        else:
            z = int(((point[2] - self.bbox[2, 0]) / self.bbox_size[2]) * self.shape[2])
        return np.array([x, y, z], dtype=np.int32)

    def is_in_bbox(self, point):
        if (point < self.bbox[:, 0]).any() or (point > self.bbox[:, 1]).any():
            return False
        else:
            return True

    def travel_to_voxels_border(self, current_point, direction, current_voxel):
        next_point = np.copy(current_point)
        next_voxel = np.copy(current_voxel)
        inc = np.sign(direction).astype(np.int32)
        voxel_fix = (inc - 1) / 2
        ts = np.ones((3,), dtype=np.float64) * 10

        if direction[0] != 0:
            ts[0] = (((current_voxel[0] + 1 + voxel_fix[0]) * self.voxel_size[0]) - current_point[0]) / direction[0]
        if direction[1] != 0:
            ts[1] = (((current_voxel[1] + 1 + voxel_fix[1]) * self.voxel_size[1]) - current_point[1]) / direction[1]
        if direction[2] != 0:
            ts[2] = (((current_voxel[2] + 1 + voxel_fix[2]) * self.voxel_size[2]) - current_point[2]) / direction[2]

        min_ind = np.argmin(ts)
        t = ts[min_ind]
        inc_condition = ts == t
        next_voxel[inc_condition] += inc[inc_condition]
        next_point += t * direction
        return abs(t), next_voxel, next_point, min_ind

    def get_intersection_with_borders(self, point, direction):
        bbox = self.bbox
        ts = np.ones((3,), dtype=np.float64) * 10
        if direction[0] > 0:
            ts[0] = (bbox[0, 1] - point[0]) / direction[0]
        elif direction[0] < 0:
            ts[0] = (bbox[0, 0] - point[0]) / direction[0]

        if direction[1] > 0:
            ts[1] = (bbox[1, 1] - point[1]) / direction[1]
        elif direction[1] < 0:
            ts[1] = (bbox[1, 0] - point[1]) / direction[1]

        if direction[2] > 0:
            ts[2] = (bbox[2, 1] - point[2]) / direction[2]
        elif direction[2] < 0:
            ts[2] = (bbox[2, 0] - point[2]) / direction[2]

        t = np.min(ts)
        res = point + t*direction
        return res

    def __str__(self):
        text = f"grid_shape={self.shape}  \n"
        text += f"bounding_box[KM]: x_lim={self.bbox[0]}, y_lim={self.bbox[1]}, z_lim={self.bbox[2]}"
        return text