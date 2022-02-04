import torch
import numba
import  numpy as np
from numba import cuda

numpy_ary = np.zeros(5, dtype=int)
numba_ary = cuda.to_device(numpy_ary)
torch_ary = torch.as_tensor(numba_ary, device="cuda")
torch_ary += 5
print(torch_ary)
print(np.asarray(numba_ary))