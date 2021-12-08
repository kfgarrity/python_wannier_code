import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


#no SOC
h = wan_ham('data/wannier90_hr.dat')
#SOC
h_soc = wan_ham('data/so/wannier90_hr.dat')

ops = ham_ops()

fermi = 5.4

orbital_info = [["Bi", 2, ["p"]], ["Se", 3, ["p"]]]

#normal dos
#p = ops.get_orbitals(orbital_info, [["Bi"]], so=False)
#energies, dos, pdos = ops.dos(h, [12,12,12], proj=p, fermi=fermi, nenergy = 500, xrange=[-2,2],  pdf="dos_nosoc.pdf")

#kpoints [16,16,16], sig is a smearing in eV, nenergy is number of energy points.

#normal dos soc
p_soc = ops.get_orbitals(orbital_info, [["Bi"]], so=True)
energies, dos, pdos = ops.dos(h_soc, [16,16,16], proj=p_soc, fermi=fermi, nenergy = 500, xrange=[-2,2], sig = 0.04, pdf="dos_soc.pdf")



#GAMMA overweighted dos soc
p_soc = ops.get_orbitals(orbital_info, [["Bi"]], so=True)
energies_gam, dos_gam, pdos_gam = ops.dos(h_soc, [16,16,16], proj=p_soc, fermi=fermi, nenergy = 500, xrange=[-2,2], sig = 0.04, pdf="dos_soc_gamma.pdf", gamma_mode=True)
