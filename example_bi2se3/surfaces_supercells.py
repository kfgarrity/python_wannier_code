import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


#h = wan_ham('BN/spin_orbit/wannier90_hr.dat')
h = wan_ham('BN/wannier90_hr.dat')

ops = ham_ops()

fermi = 3.0545

supercell = [2,2,2]
hsuper = ops.generate_supercell(h, supercell, sparse=False)

#normal supercell band structure
ops.band_struct(hsuper,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs.super.pdf')

#unfold supercell (this is trivial in this case, see my paper https://arxiv.org/abs/2003.03439)
image, lims = ops.unfold_bandstructure( h, hsuper, supercell, [[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], temp=0.05, fermi=fermi, yrange=[-8, 8], names=['A', '$\Gamma$', 'M', 'K'], pdfname='unfold.222.pdf', num_energies=150)


#surface
supercell = [1,1,6]
hsurf = ops.generate_supercell(h, supercell, cut=[0,0,1], sparse=False)

#normal supercell band structure
ops.band_struct(hsurf, [[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs.surf.pdf')

