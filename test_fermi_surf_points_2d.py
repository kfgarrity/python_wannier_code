import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


def test():

    nk1 = 32
    nk2 = 32
    origin = [-0.5, -0.5, 0]
    k1 = [1.0, 0.0, 0.0]
    k2 = [0.0, 1.0, 0.0]
    
    VALS = np.zeros((nk1+1, nk2+1, 1))

    for c1 in range(nk1+1):
        for c2 in range(nk2+1):
            k = [origin[0] + k1[0] * (c1 )/float(nk1) , origin[1] + k2[1] * (c2 )/float(nk2)]
            VALS[c1,c2,0] = -0.5 + 3.0 * (k[0]**2 + k[1]**2)
            print(c1, c2, k[0], k[1],  VALS[c1,c2,0] )
    return VALS
    
h = ham_ops()

VALS = test()
points = h.fermi_surf_points_2d(VALS,0.0)


print(points)

p = np.array(points)

plt.scatter(p[:,0], p[:,1], 2.0)
plt.show()
