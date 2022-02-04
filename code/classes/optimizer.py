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
    def __init__(self, volume:Volume, step_size, alpha, beta_mean, beta_max):
        self.volume = volume
        self.step_size = step_size
        self.alpha = alpha
        self.delta = np.zeros_like(volume.beta_cloud[self.volume.cloud_mask])
        self.beta_mean = beta_mean
        self.beta_max = beta_max

    def step(self, grad):
        mask = self.volume.cloud_mask
        self.delta = self.alpha * self.delta - (1 - self.alpha) * self.step_size * grad[mask]
        self.volume.beta_cloud[mask] += self.delta
        self.volume.beta_cloud[self.volume.beta_cloud<0] = 0
        self.volume.beta_cloud[self.volume.beta_cloud>self.beta_max] = self.beta_mean

    def reset(self):
        self.delta *= 0

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
    def __init__(self, volume: Volume, step_size, beta1, beta2, start_iter, beta_mean, max_beta, max_update,
                 eps=1e-6):
        self.volume = volume
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.start_iter = start_iter
        self.iter = 0
        self.max_beta = max_beta
        self.max_update = max_update
        self.beta_mean = beta_mean
        self.m = np.zeros_like(volume.beta_cloud)[self.volume.cloud_mask]
        self.v = np.zeros_like(volume.beta_cloud)[self.volume.cloud_mask]

    def step(self, grads):
        mask = self.volume.cloud_mask
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads[mask]**2)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads[mask]
        m_hat = self.m / (1 - self.beta1**(self.iter+1))
        if self.iter >= self.start_iter:
            # if self.iter == self.start_iter:
            #     self.step_size *= np.linalg.norm(np.sqrt(self.v)) * 10
            v_hat = self.v / (1 - self.beta2 ** (self.iter+1))
            delta = -(self.step_size * m_hat) / (np.sqrt((v_hat) + self.eps))
        else:
            delta = -(self.step_size * m_hat)
        # print("max_delta = ",np.max(delta))
        # delta[delta>self.max_update] = self.max_update
        self.volume.beta_cloud[mask] += delta
        self.volume.beta_cloud[self.volume.beta_cloud < 0] = 0
        self.volume.beta_cloud[self.volume.beta_cloud > self.max_beta] = self.beta_mean
        self.iter += 1
    def restart(self):
        self.iter = 0
        self.m = np.zeros_like(self.m)
        self.v = np.zeros_like(self.v)

    def __repr__(self):
        return f"ADAM: b1={self.beta1}, b2={self.beta2}, start_iter={self.start_iter}"