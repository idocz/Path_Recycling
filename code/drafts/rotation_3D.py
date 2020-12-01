import numpy as np
import matplotlib.pyplot as plt

def rot_mat(direction, theta, phi):
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    R = np.array([[sin_t*cos_p, sin_t*sin_p, cos_t],
              [cos_t*cos_p, cos_t*sin_p, -sin_t],
              [-sin_p,      cos_p,         0]])

    return R @ direction
