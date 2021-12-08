import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


#no SOC
h = wan_ham('data/wannier90_hr.dat')
#SOC
#h_soc = wan_ham('data/so/wannier90_hr.dat')

ops = ham_ops()

fermi = 5.6148 + 0.2

IMAGE, VALS, points = ops.fermi_surf_2d(h, fermi, [-0.5,-0.5,0.0], [1.0, 0.0, 0.0], [0.0,1.0,0.0], 100, 100, 0.05)


plt.imshow(IMAGE, origin="lower")
plt.show()

p = np.array(points)

plt.scatter(p[:,0], p[:,1], 5.0)
plt.ylim(-0.5,0.5)
plt.xlim(-0.5,0.5)
plt.show()
