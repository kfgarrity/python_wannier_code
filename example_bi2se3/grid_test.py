import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


ops = ham_ops()

normal = ops.generate_kgrid([5,5,5])

gamma = ops.generate_kgrid_gamma([5,5,5])

for i in range(len(normal)):
    print(i, " normal ", normal[i][:], " gamma  " , gamma[i][:])
