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


class CudaPaths(object):
    def __init__(self, paths):
        # each path in the paths if a list in the following order:
        # length_inds, lengths, ISs_mat, scatter_tensor, camera_pixels

        self.Np = len(paths)
        nonan_paths = [path for path in paths if path is not None]
        self.Np_nonan = len(nonan_paths)
        self.all_lengths_inds = np.concatenate([path[0] for path in nonan_paths], axis=0)
        self.all_lengths = np.concatenate([path[1] for path in nonan_paths], axis=0)
        self.all_ISs_mat = np.concatenate([path[2] for path in nonan_paths], axis=1) # first dimension is N_cams
        self.all_scatter_tensor = np.concatenate([path[3] for path in nonan_paths], axis=1) # first dimension is 3
        self.all_camera_pixels = np.concatenate([path[4] for path in nonan_paths], axis=2) # first dimension is 2 X N_cam
        scatter_sizes = [0] + [path[2].shape[1] for path in nonan_paths]
        self.scatter_inds = np.cumsum(scatter_sizes)
        voxel_sizes = [0] + [path[0].shape[0] for path in nonan_paths]
        self.voxel_inds = np.cumsum(voxel_sizes)
        self.in_device = False

    def compress(self):
        self.all_lengths_inds = self.all_lengths_inds.astype(np.uint8)
        self.all_lengths = self.all_lengths.astype(np.float32)
        self.all_ISs_mat = self.all_ISs_mat.astype(np.float32)
        self.all_scatter_tensor = self.all_scatter_tensor.astype(np.uint8)
        self.all_camera_pixels = self.all_camera_pixels.astype(np.uint8)
        self.scatter_inds = self.scatter_inds.astype(np.uint32)
        self.voxel_inds = self.voxel_inds.astype(np.uint32)

    def save(self, path):
        np.savez(path,  all_lengths_inds=self.all_lengths_inds, all_lengths=self.all_lengths, all_ISs_mat=self.all_ISs_mat,
                 all_scatter_tensor=self.all_scatter_tensor, all_camera_pixels=self.all_camera_pixels,
                 scatter_inds=self.scatter_inds, voxel_inds=self.voxel_inds)

    def to_device(self):
        self.all_lengths_inds = cuda.to_device(self.all_lengths_inds)
        self.all_lengths = cuda.to_device(self.all_lengths)
        self.all_ISs_mat = cuda.to_device(self.all_ISs_mat)
        self.all_scatter_tensor = cuda.to_device(self.all_scatter_tensor)
        self.all_camera_pixels = cuda.to_device(self.all_camera_pixels)
        self.scatter_inds = cuda.to_device(self.scatter_inds)
        self.voxel_inds = cuda.to_device(self.voxel_inds)
        self.in_device = True
    def get_args(self):
        return self.all_lengths_inds, self.all_lengths, self.all_ISs_mat, self.all_scatter_tensor,\
               self.all_camera_pixels, self.scatter_inds, self.voxel_inds