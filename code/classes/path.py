import numpy as np
from numba import cuda

class Path(object):
    def __init__(self, voxels, lengths, segments_size, ISs, camera_voxels, camera_lengths, camera_segments_size,
                 camera_ISs, camera_pixels, N_cams, N_seg):
        self.voxels = voxels
        self.lengths = lengths
        self.segments_size = segments_size
        self.ISs = ISs
        self.camera_voxels = camera_voxels
        self.camera_lengths = camera_lengths
        self.camera_segments_size = camera_segments_size
        self.camera_ISs = camera_ISs
        self.camera_pixels = camera_pixels
        self.N_cams = N_cams
        self.N_seg = N_seg

    def to_array(self):
        self.voxels = np.vstack(self.voxels)
        self.lengths = np.array(self.lengths, dtype=np.float64)
        self.segments_size = np.array(self.segments_size, dtype=np.int32)
        self.ISs = np.array(self.ISs, dtype=np.float64)
        self.camera_voxels = np.vstack(self.camera_voxels)
        self.camera_lengths = np.array(self.camera_lengths, dtype=np.float64)
        self.camera_segments_size = np.array(self.camera_segments_size, dtype=np.int32)
        self.camera_ISs = np.array(self.camera_ISs, dtype=np.float64)
        self.camera_pixels = np.vstack(self.camera_pixels)
        self.ISs = self.ISs.reshape(-1, 1)
        self.camera_ISs = self.camera_ISs.reshape((self.N_seg, self.N_cams))
        self.camera_pixels = self.camera_pixels.reshape((self.N_seg, self.N_cams, 2))

    def get_arrays(self):
        return self.voxels, self.lengths, self.segments_size, self.ISs, self.camera_voxels, self.camera_lengths, \
               self.camera_segments_size, self.camera_ISs, self.camera_pixels
