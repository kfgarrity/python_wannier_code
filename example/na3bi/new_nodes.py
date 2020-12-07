import matplotlib
matplotlib.use('TkAgg')
import  matplotlib.pyplot as plt
import json
import numpy as np
import sys

from wan_ham import wan_ham
from ham_ops import ham_ops

h = wan_ham("wannier90_hr.dat")
h.trim()

#fermi = 6.0588

ops = ham_ops()

num_occ = 52 - 36
IMAGE, DIRECTGAP,VAL, gaps, weyl_points, dirac_points, ho_points = ops.find_nodes(h, num_occ, nk1=16, nk2=16, nk3=16 , use_min = True )
#
#print("gaps ", gaps)



for d in dirac_points:
    val,vect,p = h.solve_ham(d,proj=None)    
    gap = (val[num_occ] - val[num_occ-1])
    print(d, " dirac ", gap)

np.save("IMAGE", IMAGE)
np.save("DIRECTGAP", DIRECTGAP)

    
np.savetxt("d", np.array(dirac_points))
np.savetxt("w", np.array(weyl_points))

plt.imshow(np.sum(IMAGE[:,:,:,1], axis=0))
plt.savefig("projected_direct_gap.pdf")
plt.show()
