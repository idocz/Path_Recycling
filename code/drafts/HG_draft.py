import numpy as np
import matplotlib.pyplot as plt


def HG(theta, g):
    return 0.5*(1 - g**2)/(1 + g**2 - 2*g * np.cos(theta)) ** 1.5

def HG_cos(cos_theta, g):
    return 0.5*(1 - g**2)/(1 + g**2 - 2*g * cos_theta) ** 1.5
def generate_theta(samples_num, g):
    ps = np.random.rand(samples_num)
    return (1 / (2 * g)) * (1 + g**2 - ((1 - g**2)/(1 - g + 2*g*ps))**2)

N = 100000
g = 0.0000001
thetas = np.linspace(0,np.pi,N)
cos_thetas = np.linspace(-1,1,N)
# f = HG(thetas, g)
f = HG_cos(cos_thetas, g)
thetas_rand = generate_theta(1000000, g)
plt.figure()
plt.hist(thetas_rand, bins=1000, density=True)
plt.plot(cos_thetas,f)
plt.show()