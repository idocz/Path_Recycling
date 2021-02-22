import numpy as np
import math
import matplotlib.pyplot as plt

def raylie(cos_theta):
    theta_pdf = (3/8) * (1 + cos_theta**2)
    return theta_pdf

N = 1000000
p = np.random.rand(N)
U = -(2*(2* p - 1) + (4 * ((2 * p - 1) ** 2) + 1) ** (1/2))**(1/3)
samples = U - (1/ U)

# # samples = np.array(samples)
plt.hist(samples, density=True, bins=100)

x = np.linspace(-1,1,N)
y = raylie(x)
print(np.sum(y*(2/N)))

plt.plot(x,y)
plt.show()

# thetas = np.linspace(0,2*np.pi)
# cos_thetas = np.cos(thetas)
# y = raylie(cos_thetas)
print(np.sum(y*(2*np.pi/N)))