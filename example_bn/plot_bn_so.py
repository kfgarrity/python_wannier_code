import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


h = wan_ham('BN/spin_orbit/wannier90_hr.dat')
#h = wan_ham('BN/wannier90_hr.dat')

ops = ham_ops()

fermi = 3.0545

#chemical formula is B2N2, each one has s and p orbitals
orbital_info = [["B", 2, ["s", "p"]], ["N", 2, ["s", "p"]]]

#N px+py orbitals
orbs = ops.get_orbitals(orbital_info, [["N", "px"], ["N", "py"]], so=True)
ops.band_struct(h,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], proj = orbs, names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='SO.bs.N.px.py.pdf', colorbar = True)

#calculate chern number (should be zero)                     #16 occupied
chern, direct_gap, indirect_gap = ops.chern_number_simple(h, 16, [1,0,0], [0,1,0], 25, 25)

print "Chern number ", chern