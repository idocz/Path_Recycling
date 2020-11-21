import numpy as np
import sparse as sp
from time import time


for i in range(3):
    print("hey")

print(i)
# create dense matrix
# start = time()
# A = np.zeros((32,32,32,8,1000))
# A[16,16,16,:,3:6] = 2
# end = time()
# print(f"sparse sum took: {end-start}")
# print(A)
# convert to sparse matrix (CSR method)
# start = time()
# S = sp.COO(A)
# end = time()
# print(f"init took: {end-start}")
# S = sp.tensordot(S,S, axes=((0,1,2,3,4),(0,1,2,3,4)))

# start = time()
# su = np.sum(S)#, axis=(0,1,2))
# end = time()
# print(f"sparse sum took: {end-start}")
# print(su)

start = time()
su = np.sum(A)#, axis=(0,1,2))
end = time()
print(f"sum took: {end-start}")
# print(S)
# reconstruct dense matrix
# B = S.todense()
# print(B)