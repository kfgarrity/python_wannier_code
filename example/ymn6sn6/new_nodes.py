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

nelec = 173-104
num_occ = nelec

IMAGE, DIRECTGAP, VALS, gaps, weyl_points, dirac_points, ho_points = ops.find_nodes(h, nelec, nk1=8, nk2=8, nk3=8)

print("gaps ", gaps)

for d in dirac_points:
    val,vect,p = h.solve_ham(d,proj=None)    
    gap = (val[num_occ] - val[num_occ-1])
    print(d, " dirac ", gap)

for d in weyl_points:
    val,vect,p = h.solve_ham(d,proj=None)    
    gap = (val[num_occ] - val[num_occ-1])
    print(d, " weyl ", gap)
    
    
np.savetxt("d", np.array(dirac_points))
np.savetxt("w", np.array(weyl_points))

    
np.save("IMAGE", IMAGE)
np.save("DIRECTGAP", DIRECTGAP)
    
plt.imshow(np.sum(IMAGE[:,:,:,1], axis=0))
plt.savefig("projected_direct_gap.pdf")
plt.show()


#points, gaps, vals = ops.get_gap_points(DIRECTGAP, 0.01, val=VALS)

#plt.imshow(IMAGE)
#plt.savefig("fermi.pdf")
