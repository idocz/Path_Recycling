import numpy as np
from numba import cuda

class SparsePath(object):
    def __init__(self, length_inds, lengths, ISs_mat, scatter_inds, camera_pixels, N_seg, N_cams):
        self.length_inds = length_inds
        self.lengths = lengths
        self.ISs_mat = ISs_mat
        self.scatter_inds = scatter_inds
        self.N_seg = N_seg
        self.N_cams = N_cams
        self.camera_pixels = camera_pixels


