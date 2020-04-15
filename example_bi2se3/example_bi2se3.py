import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


#no SOC
h = wan_ham('data/wannier90_hr.dat')
#SOC
h_soc = wan_ham('data/so/wannier90_hr.dat')

ops = ham_ops()

fermi = 5.6148

orbital_info = [["Bi", 2, ["p"]], ["Se", 3, ["p"]]]

#bulk
if True:
    #bulk no SOC
    orbs = ops.get_orbitals(orbital_info, [["Bi"]])
    ops.band_struct(h,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], proj=orbs, yrange=[-2, 2], names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs.pdf')

    

    orbs = ops.get_orbitals(orbital_info, [["Bi"]], so=True)
    ops.band_struct(h_soc,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0], [0.333333333333,0.33333333333,0.0]], proj=orbs, yrange=[-2, 2], names = ['A', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs_soc.pdf')


print "surface calc 1 1 6"
#surface (see cut)
supercell = [1,1,6]
hsurf = ops.generate_supercell(h_soc, supercell, cut=[0,0,1], sparse=False)

#surface band structure

ops.band_struct(hsurf, [[0.33333333333,0.33333333333333, 0.0],[0,0,0],[0.5,0.5, 0.0], [0.333333333333,0.33333333333,0.0]], names = ['K', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs_soc.surf.pdf')


#zoom in
ops.band_struct(hsurf, [[0.33333333333,0.33333333333333, 0.0],[0,0,0],[0.5,0.5, 0.0], [0.333333333333,0.33333333333,0.0]], yrange=[-1.5, 1.0], names = ['K', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs_soc.surf.zoom.pdf')



#project onto all Bi
orbs = ops.get_orbitals(orbital_info, [["Bi"]], so=True, NCELLS=6)
ops.band_struct(hsurf, [[0.33333333333,0.33333333333333, 0.0],[0,0,0],[0.5,0.5, 0.0], [0.333333333333,0.33333333333,0.0]], proj=orbs, yrange=[-1.5, 1.0], names = ['K', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs_soc.surf.zoom.PROJ_Bi.pdf')


#project onto all surface WFs
orbs = ops.get_orbitals(orbital_info, [["Bi"], ["Se"]], so=True,NCELLS=6,surfaceonly=True)
ops.band_struct(hsurf, [[0.33333333333,0.33333333333333, 0.0],[0,0,0],[0.5,0.5, 0.0], [0.333333333333,0.33333333333,0.0]], proj=orbs, yrange=[-1.5, 1.0], names = ['K', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs_soc.surf.zoom.PROJ_surface.pdf')


print "surface calc 1x1x10"
#surface
supercell = [1,1,10]
hsurf10 = ops.generate_supercell(h_soc, supercell, cut=[0,0,1], sparse=False)

#normal supercell band structure
ops.band_struct(hsurf10, [[0.33333333333,0.33333333333333, 0.0],[0,0,0],[0.5,0.5, 0.0], [0.333333333333,0.33333333333,0.0]], yrange=[-1.5, 1.0], names = ['K', '$\Gamma$', 'M', 'K'],  fermi=fermi, pdfname='bs_soc.1x1x10.surf.zoom.pdf')
