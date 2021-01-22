import numpy as np
import math
N = 10
for i in range(N):
    p1 = np.random.rand(1)[0]
    u = -(2*(2* p1 - 1) + (4 * ((2 * p1 - 1) ** 2) + 1) ** (1/2))**(1/3)

    cos_theta = u - (1/ u)

    temp1 = (-2*(2*p1 -1) + np.sqrt(4*(2*p1-1)**2 + 1))
    temp2 = (-2*(2*p1 -1) - np.sqrt(4*(2*p1-1)**2 + 1))
    res = 0
    if temp1 < 0:
        res += (-(-temp1)**(1/3))
    else:
        res += (temp1**(1/3))
    if temp2 < 0:
        res += (-(-temp2)**(1/3))
    else:
        res += (temp2**(1/3))
    # cos_theta = (-2*(2*p1 -1) + np.sqrt(4*(2*p1-1)**2 + 1))**(1/3) + (-2*(2*p1 -1) - np.sqrt(4*(2*p1-1)**2 + 1))**(1/3)
    print(res, cos_theta)