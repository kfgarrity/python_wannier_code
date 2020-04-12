import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


#h = wan_ham('BN/spin_orbit/wannier90_hr.dat')
h = wan_ham('BN/wannier90_hr.dat')

ops = ham_ops()

fermi = 3.0545

#chemical formula is B2N2, each one has s and p orbitals
orbital_info = [["B", 2, ["s", "p"]], ["N", 2, ["s", "p"]]]


#normal band structure
ops.band_struct(h,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs.pdf')

#B-projected
orbs = ops.get_orbitals(orbital_info, [["B"]])
ops.band_struct(h,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], proj = orbs, names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs.B.pdf', colorbar = True)

#N-s orbitals
orbs = ops.get_orbitals(orbital_info, [["N", "s"]])
ops.band_struct(h,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], proj = orbs, names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs.N.s.pdf', colorbar = True)

#N px+py orbitals
orbs = ops.get_orbitals(orbital_info, [["N", "px"], ["N", "py"]])
ops.band_struct(h,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], proj = orbs, names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs.N.px.py.pdf', colorbar = True)

h.trim()

#normal DOS
energies, dos, pdos = ops.dos(h, [20,20,20], fermi=fermi, xrange=[-20,25], nenergy=500, sig = 0.2, pdf="dos.pdf")

#N-projected DOS
orbs = ops.get_orbitals(orbital_info, [["N"]])
energies, dos, pdos = ops.dos(h, [20,20,20], proj=orbs, fermi=fermi, xrange=[-20,25], nenergy=500, sig = 0.2, pdf="pdos.pdf")
