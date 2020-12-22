
import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
import pickle
from utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
from numba import cuda
from classes.path import *
import math
from classes.visual import Visual_wrapper
from time import time
checkpoint_id = "1612-1307-25_iter2755"
load_iter = 3950
beta_gt = loadmat(join("data", "rico.mat"))["beta"]
cp = pickle.load(open(join("checkpoints",checkpoint_id,"data",f"checkpoint_loader"), "rb" ))
print("Loading the following Scence:")
print(cp)
scene = cp.scene


Ns = 15
N_cams = len(scene.cameras)
pixels = scene.cameras[0].pixels
@cuda.jit()
def render_cuda(all_lengths_inds, all_lengths, all_ISs_mat, all_scatter_tensor,\
               all_camera_pixels, scatter_inds, voxel_inds, betas, beta_air, w0_cloud, w0_air, I_total):
    tid = cuda.grid(1)
    if tid < voxel_inds.shape[0] - 1:
        # reading thread indices

        scatter_start = scatter_inds[tid]
        scatter_end = scatter_inds[tid+1]
        voxel_start = voxel_inds[tid]
        voxel_end = voxel_inds[tid+1]

        # reading thread data
        length_inds = all_lengths_inds[voxel_start:voxel_end]
        lengths = all_lengths[voxel_start:voxel_end]
        ISs_mat = all_ISs_mat[:, scatter_start:scatter_end]
        scatter_tensor = all_scatter_tensor[:, scatter_start:scatter_end]
        camera_pixels = all_camera_pixels[:, :, scatter_start:scatter_end]

        # rendering
        N_seg = scatter_tensor.shape[1]
        path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=np.float64)
        # optical_lengths = optical_lengths[:,:N_seg]

        for row_ind in range(lengths.shape[0]):
            i, j, k, cam_ind, seg = length_inds[row_ind]
            L = lengths[row_ind]
            if cam_ind == -1:
                for cam_j in range(N_cams):
                    for seg_j in range(N_seg - seg):
                        path_contrib[cam_j, seg + seg_j] += betas[i, j, k] * L
            else:
                path_contrib[cam_ind, seg] += betas[i, j, k] * L

        si = scatter_tensor
        prod = 1
        for seg in range(N_seg):
            prod *= (w0_cloud * (betas[si[0, seg], si[1, seg], si[2, seg]] - beta_air) + w0_air * beta_air)
            for cam_j in range(N_cams):
                path_contrib[cam_j, seg] = ISs_mat[cam_j, seg] * math.exp(-path_contrib[cam_j, seg]) * prod
                pixel = camera_pixels[:, cam_j, seg]
                cuda.atomic.add(I_total, (cam_j,pixel[0], pixel[1]), path_contrib[cam_j, seg])




beta_air = scene.volume.beta_air
w0_cloud = scene.volume.w0_cloud
w0_air = scene.volume.w0_air

Np = int(1e6)
to_load = True
if to_load:
    paths = np.load(f"./paths_{np.log10(Np)}.npz", allow_pickle=True)["paths"]
    print("paths has been loaded")
else:
    paths = scene.build_paths_list(Np, Ns)
    np.savez(f"./paths_{np.log10(Np)}",paths=np.array(paths, dtype=np.object))

I_total_cpu = scene.render(paths, differentiable=False)
start = time()
I_total_cpu = scene.render(paths, differentiable=False)
end = time()
print(f"CPU running time:{end-start}")

visual = Visual_wrapper(scene)
max_val = np.max(I_total_cpu, axis=(1,2))
visual.plot_images(I_total_cpu, max_val, "CPU")
plt.show()


start = time()
paths_cuda = CudaPaths(paths)
end = time()
print(f"creating cuda paths: {end-start}")

start = time()
paths_cuda.to_device()
args = paths_cuda.get_args()
end = time()
print(f"moving data to device: {end-start}")

I_total = np.zeros((N_cams, pixels[0], pixels[1]), dtype=np.float64)
dbetas = cuda.to_device(scene.volume.betas)
dI_total = cuda.to_device(I_total)
threadsperblock = 512
blockspergrid = (paths_cuda.Np_nonan + (threadsperblock - 1)) // threadsperblock


# running for compiling
render_cuda[blockspergrid, threadsperblock](*args, dbetas, beta_air, w0_cloud, w0_air, dI_total)
dI_total = cuda.to_device(I_total*0)

start = time()
render_cuda[blockspergrid, threadsperblock](*args, dbetas, beta_air, w0_cloud, w0_air, dI_total)
I_total_GPU = dI_total.copy_to_host()
I_total_GPU /= Np
end = time()
print(f"GPU running time: {end - start}")


visual.plot_images(I_total_GPU, max_val, "GPU")
plt.show()


diff = np.abs(I_total_cpu - I_total_GPU).sum()
diff /= np.abs(I_total_cpu).sum()
print(f"rel_diff = {diff}")
