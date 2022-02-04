import numpy as np
from utils import theta_phi_to_direction

class UniformPhaseFunction(object):
    def __init__(self):
        self.name = "uniform"

    def pdf(self, theta):
        return 1/(4*np.pi)

    def sample_direction(self, old_direction):
        p1, p2 = np.random.rand(2)
        cos_theta = (p1 - 0.5) * 2
        theta = np.arccos(cos_theta)
        phi = p2 * 2 * np.pi
        direction = theta_phi_to_direction(theta, phi)
        return direction
    def __str__(self):
        return "Uniform"

class HGPhaseFunction(object):
    def __init__(self, g):
        self.g = g
        self.name = "hg"

    def pdf(self, cos_theta):
        theta_pdf = 0.5*(1 - self.g**2)/(1 + self.g**2 - 2*self.g * cos_theta) ** 1.5
        phi_pdf = 1 / (2*np.pi)
        return theta_pdf * phi_pdf

    def sample_direction(self, old_direction):
        new_direction = np.empty(3)
        p1, p2 = np.random.rand(2)
        cos_theta = (1 / (2 * self.g)) * (1 + self.g**2 - ((1 - self.g**2)/(1 - self.g + 2*self.g*p1))**2)
        phi = p2 * 2 * np.pi
        sin_theta = np.sqrt(1 - cos_theta**2)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        if np.abs(old_direction[2]) > 0.99999:#|z| ~ 1
            z_sign = np.sign(old_direction[2])
            new_direction[0] = sin_theta * cos_phi
            new_direction[1] = z_sign * sin_theta * sin_phi
            new_direction[2] = z_sign * cos_theta
        else:
            denom = np.sqrt(1 - old_direction[2]**2)
            z_cos_phi = old_direction[2] * cos_phi
            new_direction[0] = (sin_theta * (old_direction[0] * z_cos_phi - old_direction[1] * sin_phi) / denom) + old_direction[0] * cos_theta
            new_direction[1] = (sin_theta * (old_direction[1] * z_cos_phi + old_direction[0] * sin_phi) / denom) + old_direction[1] * cos_theta
            new_direction[2] = old_direction[2] * cos_theta - denom * sin_theta * cos_phi
        return new_direction

    def __str__(self):
        return f"HG(g={self.g})"