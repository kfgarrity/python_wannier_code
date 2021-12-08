import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


h = wan_ham("artificial_hr.dat")

fermi = 0.0

ops = ham_ops()

ops.band_struct(h,[[0,0,0.5],[0,0,0],[0.5,0.0, 0.0]], names = ['Z', '$\Gamma$', 'X'],  fermi=fermi, pdfname='bs.pdf')

