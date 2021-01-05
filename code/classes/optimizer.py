import numpy as np
from classes.volume import Volume

class SGD(object):
    def __init__(self, volume:Volume, step_size):
        self.volume = volume
        self.step_size = step_size

    def step(self, grad):
        self.volume.beta_cloud = self.volume.beta_cloud - self.step_size * grad
        self.volume.beta_cloud[self.volume.beta_cloud<0] = 0

    def __repr__(self):
        return f"SGD_{self.step_size:10.0e}"



class MomentumSGD(object):
    def __init__(self, volume:Volume, step_size, alpha):
        self.volume = volume
        self.step_size = step_size
        self.alpha = alpha
        self.delta = np.zeros_like(volume.beta_cloud)

    def step(self, grad):
        self.delta = self.alpha * self.delta - (1 - self.alpha) * self.step_size * grad
        self.volume.beta_cloud += self.delta
        self.volume.beta_cloud[self.volume.beta_cloud<0] = 0

    def __repr__(self):
        return f"MSGD: alpha={self.alpha:10.0e}"


class AdaGrad(object):
    def __init__(self, volume:Volume, step_size, start_iter, eps=1e-2):
        self.volume = volume
        self.step_size = step_size
        self.G = np.zeros_like(volume.beta_cloud)
        self.eps = eps
        self.start_iter = start_iter
        self.iter = 0

    def step(self, grads):
        self.G += grads ** 2
        if self.iter > self.start_iter:
            delta = -(self.step_size * (self.G+self.eps)**(-1/2)) * grads
        else:
            delta = - self.step_size*grads
        self.volume.beta_cloud += delta
        self.iter += 1

    def __repr__(self):
        return f"AGRAD_{self.step_size:10.0e}"


class RMSProp(object):
    def __init__(self, volume:Volume, step_size, alpha, start_iter, eps=1e-5):
        self.volume = volume
        self.step_size = step_size
        self.alpha = alpha
        self.G = np.zeros_like(volume.beta_cloud)
        self.eps = eps
        self.start_iter = start_iter
        self.iter = 0

    def step(self, grads):
        self.G = self.alpha * self.G + (1 - self.alpha) * (grads ** 2)
        if self.iter >= self.start_iter:
            delta = -(self.step_size * (self.G+self.eps)**(-1/2)) * grads
        else:
            delta = - self.step_size*grads
        self.volume.beta_cloud += delta
        self.volume.beta_cloud[self.volume.beta_cloud < 0] = 0
        self.iter += 1

    def __repr__(self):
        return f"RMSP_lr={self.step_size:10.0e}_a={self.alpha:10.0e}"


class ADAM(object):
    def __init__(self, volume: Volume, step_size, beta1, beta2, start_iter, eps=1e-6):
        self.volume = volume
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.start_iter = start_iter
        self.iter = 0
        self.m = np.zeros_like(volume.beta_cloud)
        self.v = np.zeros_like(volume.beta_cloud)

    def step(self, grads):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)
        mask = self.volume.cloud_mask
        if self.iter >= self.start_iter:
            m_hat = self.m / (1 - self.beta1**(self.iter+1))
            v_hat = self.v / (1 - self.beta2 ** (self.iter+1))
            delta = np.zeros_like(m_hat)
            delta[mask] = -(self.step_size * m_hat[mask]) / (np.sqrt((v_hat[mask]) + self.eps))
        else:
            delta = - self.step_size * grads
        self.volume.beta_cloud += delta
        self.volume.beta_cloud[self.volume.beta_cloud < 0] = 0
        self.iter += 1
    def restart(self):
        self.iter = 0
        self.m = np.zeros_like(self.m)
        self.v = np.zeros_like(self.v)

    def __repr__(self):
        return f"ADAM: b1={self.beta1}, b2={self.beta2}, start_iter={self.start_iter}"