import numpy as np
import matplotlib.pyplot as plt
from utils import *
from classes.phase_function import HGPhaseFunction, UniformPhaseFunction

def rot_mat(direction, theta, phi):
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    R = np.array([[sin_t*cos_p, sin_t*sin_p, cos_t],
              [cos_t*cos_p, cos_t*sin_p, -sin_t],
              [-sin_p,      cos_p,         0]])

    return R @ direction
def angles_between_vectors(v1, v2):
    angle = np.arccos(np.dot(v1, v2))
    return angle

def HG(theta, g):
    return 0.5*(1 - g**2)/(1 + g**2 - 2*g * np.cos(theta)) ** 1.5


def HG_cos(cos_theta, g):
    return 0.5*(1 - g**2)/(1 + g**2 - 2*g * cos_theta) ** 1.5


g = 0.8
fp = HGPhaseFunction(g)
# fp = UniformPhaseFunction()
v1 = np.array([-1,1,0])
v1 = v1 / np.linalg.norm(v1)
v2 = fp.sample_direction(v1)
print(np.linalg.norm(v2))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.quiver(0,0,0,*v1)
ax.quiver(v1[0], v1[1], v1[2], *v2)
angle = angles_between_vectors(v1,v2)
print(angle*(180/np.pi))
plt.show()

hist_list = []
for i in range(10000):
    v2 = fp.sample_direction(v1)
    angle = angles_between_vectors(v1,v2)
    # hist_list.append(np.cos(angle))
    hist_list.append(angle)
plt.figure()
plt.hist(hist_list, bins=100, density=True)

N = 100
thetas = np.linspace(0,np.pi,N)
f = HG_cos(np.cos(thetas), g)
# plt.figure()
plt.plot(np.cos(thetas),f)
plt.title(g)
plt.show()
