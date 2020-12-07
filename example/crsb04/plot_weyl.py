import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
import json
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D

from wan_ham import wan_ham
from ham_ops import ham_ops


#d = np.loadtxt("d")
w = np.loadtxt("w")
#h = np.loadtxt("h")

h = wan_ham("wannier90_hr.dat")
h.trim()
ops = ham_ops()
nelec = 82-28
num_occ = nelec



gaps = []
for c in range(w.shape[0]):
    k = w[c,:]
    val,vect,p = h.solve_ham(k,proj=None)    
    gap = (val[num_occ] - val[num_occ-1])

    gaps.append(gap)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#plt.plot(w[:,0], w[:,1], w[:,2], "b.")

ax.scatter(w[:,0], w[:,1], w[:,2],c=gaps)

ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,0.5)
ax.set_zlim(-0.5,0.5)

ax.set_xlabel("kx")
ax.set_ylabel("ky")
ax.set_zlabel("kz")

plt.savefig("weyl_points.pdf")
plt.show()



good = []
thresh2 = 2e-4

for c in range(w.shape[0]):
    gap = gaps[c]
    if gap < thresh2:
        good.append(c)

w2 = w[good,:]



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(w2[:,0], w2[:,1], w2[:,2], c="b")

ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,0.5)
ax.set_zlim(-0.5,0.5)

ax.set_xlabel("kx")
ax.set_ylabel("ky")
ax.set_zlabel("kz")

plt.savefig("weyl_points_higher_thresh.pdf")
plt.show()
