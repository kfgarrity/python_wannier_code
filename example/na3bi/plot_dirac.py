import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
import json
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D

d = np.loadtxt("d")
#w = np.loadtxt("w")
#h = np.loadtxt("h")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.plot(d[:,0], d[:,1], d[:,2], "b.")

ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5,0.5)
ax.set_zlim(-0.5,0.5)

ax.set_xlabel("kx")
ax.set_ylabel("ky")
ax.set_zlabel("kz")

plt.savefig("dirac_points.pdf")
plt.show()
