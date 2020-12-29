import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
import json
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D

from wan_ham import wan_ham
from ham_ops import ham_ops


h = wan_ham("wannier90_hr.dat")
h.trim()
ops = ham_ops()
nelec = 173-104
num_occ = nelec



#d = np.loadtxt("d")
w = np.loadtxt("w")

good = []
thresh2 = 1e-4
gaps = []
for c in range(w.shape[0]):
    k = w[c,:]
    val,vect,p = h.solve_ham(k,proj=None)    
    gap = (val[num_occ] - val[num_occ-1])
    if gap < 0.0003:
        gaps.append(gap)
        print(k, " " , gap)
        good.append(c)

w = w[good,:]
        #h = np.loadtxt("h")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


plt.plot(w[:,0], w[:,1], w[:,2], "b.")

#p = ax.scatter(w[:,0], w[:,1], w[:,2], c = gaps)
#fig.colorbar(p)

ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,0.5)
ax.set_zlim(-0.5,0.5)

ax.set_xlabel("kx")
ax.set_ylabel("ky")
ax.set_zlabel("kz")


plt.savefig("weyl_points.pdf")
plt.show()

