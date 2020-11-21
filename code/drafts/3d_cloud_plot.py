import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

shape = 7

# constructing beta
x = np.linspace(0,1, shape)
y = np.linspace(0,1, shape)
z = np.linspace(0,1, shape)
xx, yy, zz = np.meshgrid(x,y,z)
R = 0.5
a = 0.5
b = 0.5
c = 0.5
beta_cloud = np.zeros((shape, shape, shape), dtype=np.float64)
cond = ((xx - a)**2 + (yy - b)**2 + (zz - c)**2) <= R**2
beta_cloud[cond] = 5
# print(beta_cloud)
# for i in range(shape):
#     for j in range(shape):
#         for l in range(shape):
#             beta = beta_cloud[i,j,l]
#             print(beta)
#             if beta != 0:
#                 ax.scatter(xx[i,j,l], yy[i,j,l], zz[i,j,l], c=beta, cmap="gray", s=100)




ticks = np.linspace(0, 1, shape)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
plt.show()

