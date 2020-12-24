
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
scene.volume.set_beta_cloud(beta_gt)

Ns = 15
N_cams = 5
pixels = [55,55]
@cuda.jit()
def render_cuda(all_lengths_inds, all_lengths, all_ISs_mat, all_scatter_tensor,\
               all_camera_pixels, scatter_inds, voxel_inds, betas, beta_air, w0_cloud, w0_air, I_total, total_grad):
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
        path_contrib = cuda.local.array(shape=(N_cams, Ns), dtype=np.float32)
        # optical_lengths = optical_lengths[:,:N_seg]

        for row_ind in range(lengths.shape[0]):
            i, j, k, cam_ind, seg = length_inds[row_ind]
            L = lengths[row_ind]
            if cam_ind == 255:
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
                print(ISs_mat[cam_j, seg] )
                path_contrib[cam_j, seg] = ISs_mat[cam_j, seg] * math.exp(-path_contrib[cam_j, seg]) * prod
                pixel = camera_pixels[:, cam_j, seg]
                cuda.atomic.add(I_total, (cam_j,pixel[0], pixel[1]), path_contrib[cam_j, seg])


        for row_ind in range(lengths.shape[0]):
            i, j, k, cam_ind, seg = length_inds[row_ind]
            L = lengths[row_ind]
            if cam_ind == 255:
                pixel = camera_pixels[:, :, seg:]
                for pj in range(pixel.shape[2]):
                    for cam_j in range(N_cams):
                        cuda.atomic.add(total_grad,(i, j, k, cam_j, pixel[0, cam_j, pj], pixel[1, cam_j, pj]),
                                        -L * path_contrib[cam_j, seg + pj])

            else:
                pixel = camera_pixels[:, cam_ind, seg]
                cuda.atomic.add(total_grad,(i, j, k, cam_ind, pixel[0], pixel[1]), -L * path_contrib[cam_ind, seg])


        for seg in range(N_seg):
            beta_scatter = w0_cloud * (betas[si[0, seg], si[1, seg], si[2, seg]] - beta_air) + w0_air * beta_air
            pixel = camera_pixels[:, :, seg:]
            for pj in range(pixel.shape[2]):
                for cam_ind in range(N_cams):
                    cuda.atomic.add(total_grad,
                                    (si[0, seg], si[1, seg], si[2, seg], cam_ind, pixel[0, cam_ind, pj], pixel[1, cam_ind, pj]),
                                    (w0_cloud / beta_scatter) * path_contrib[cam_ind, seg + pj])





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




start = time()
paths_cuda = CudaPaths(paths)
paths_cuda.compress()
# paths_cuda.save("./path_cuda_compressed")
end = time()
print(f"creating cuda paths: {end-start}")

start = time()
paths_cuda.to_device()
args = paths_cuda.get_args()
end = time()
print(f"moving data to device: {end-start}")


dbetas = cuda.to_device(scene.volume.betas)
threadsperblock = 256
blockspergrid = (paths_cuda.Np_nonan + (threadsperblock - 1)) // threadsperblock

dI_total = cuda.device_array((N_cams, pixels[0], pixels[1]), dtype=np.float32)
dtotal_grad = cuda.device_array((*beta_gt.shape, N_cams, pixels[0], pixels[1]), dtype=np.float32)

for i in range(1000):
    print(f"iter: {i}")
    dI_total.copy_to_device(np.zeros((N_cams, pixels[0], pixels[1]), dtype=np.float32))
    dtotal_grad.copy_to_device(np.zeros((*beta_gt.shape, N_cams, pixels[0], pixels[1]), dtype=np.float32))

    start = time()
    render_cuda[blockspergrid, threadsperblock](*args, dbetas, beta_air, w0_cloud, w0_air, dI_total, dtotal_grad)
    cuda.synchronize()
    I_total_GPU = dI_total.copy_to_host()
    total_grad_GPU = dtotal_grad.copy_to_host()
    I_total_GPU /= Np
    total_grad_GPU /= Np
    end = time()
    print(f"GPU running time: {end - start}")

visual = Visual_wrapper(scene)
max_val = np.max(I_total_GPU, axis=(1,2))


I_total_cpu, total_grad_cpu = scene.render(paths, differentiable=True)
start = time()
I_total_cpu, total_grad_cpu = scene.render(paths, differentiable=True)
end = time()
print(f"CPU running time:{end-start}")


visual.plot_images(I_total_cpu, max_val, "CPU")
plt.show()


visual.plot_images(I_total_GPU, max_val, "GPU")
plt.show()


diff = np.abs(I_total_cpu - I_total_GPU).sum()
diff /= np.abs(I_total_cpu).sum()
print(f"rel_diff = {diff}")

diff = np.abs(total_grad_GPU - total_grad_cpu).sum()
diff /= np.abs(total_grad_cpu).sum()
print(f"rel_diff_grad = {diff}")
