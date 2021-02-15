import numpy as np
import matplotlib.pyplot as plt


def HG(theta, g, b):
    return 0.5*(b + (1-b)*(1 - g**2)/(1 + g**2 - 2*g * np.cos(theta)) ** 1.5)

def HG2_cos(cos_theta, g, b ):
    return 0.5*( b*(1 - g**2)/(1 + g**2 + 2*g * cos_theta) ** 1.5 + ((1-b)*(1 - g**2))/(1 + g**2 - 2*g * cos_theta) ** 1.5)
def generate_theta2(samples_num, g, b):
    ps = np.random.rand(samples_num)
    coins = np.random.rand(samples_num)
    backscattering = np.ones_like(ps)
    backscattering[coins <= b] =  -1
    return (1 / (2 * g)) * (1 + g**2 - ((1 - g**2)/(1 - g + 2*g*ps))**2) * backscattering

def HG_cos(cos_theta, g, b ):
    return 0.5*(b + ((1-b)*(1 - g**2))/(1 + g**2 - 2*g * cos_theta) ** 1.5)
def generate_theta(samples_num, g, b):
    ps = np.random.rand(samples_num)
    coins = np.random.rand(samples_num)
    res = np.zeros(samples_num)
    cond = coins<=b
    cond_not = np.logical_not(cond)
    res[cond] = 2 * (ps[cond] - 0.5)
    res[cond_not] = (1 / (2 * g)) * (1 + g**2 - ((1 - g**2)/(1 - g + 2*g*ps[cond_not]))**2)
    return res

N = 100000
g = 0.82
b= 0.1
thetas = np.linspace(0,np.pi,N)
cos_thetas = np.linspace(-1,1,N)
# f = HG(thetas, g)
f = HG2_cos(cos_thetas, g, b)
thetas_rand = generate_theta2(N, g, b)
plt.figure()
plt.hist(thetas_rand, bins=1000, density=True)
plt.plot(cos_thetas,f)
plt.show()