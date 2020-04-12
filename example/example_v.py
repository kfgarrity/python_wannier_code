import numpy as np

from wan_ham import wan_ham
from ham_ops import ham_ops

import matplotlib.pyplot as plt 


h = wan_ham('VAG-bulk-hr.dat')
ops = ham_ops()

fermi=3.817



#band structure plot example
vals = ops.band_struct(h,[[0.5,0,0],[0,0,0],[0,0.5,0], [0,0,0.5],[0,0,0],[0.5,0.5, 0.5]], yrange = [-2.0, 2.0], names = ['X', '$\Gamma$', 'Y', 'Z', '$\Gamma$', 'Q'],  proj=range(18), fermi=fermi, pdfname='bs.pdf') 

exit()

################################
#extract kpoint list from qe.win file

K = []
win = open('qe.win', 'r')
kpoints=False
for line in win:
    sp = line.split()
    if len(sp) == 0:
        continue

    if sp[0] == 'begin' and sp[1] == 'kpoints':
        kpoints = True
    elif sp[0] == 'end' and sp[1] == 'kpoints':
        kpoints = False
    elif kpoints == True:
        K.append(map(float, sp))

win.close()

#now we have kpoints, solve with model

VALS = []
for k in K:
    val,vect,proj = h.solve_ham(k)
    VALS.append(val)

##################################
#extract eigenvalues from qe.eig


DFT = []
eig=open('qe.eig', 'r')
k=0
for c,line in enumerate(eig):
    sp = line.split()

    if float(sp[1]) != k: #new kpoint
        k += 1
        DFT.append([])

    DFT[-1].append(float(sp[2]))

eig.close()



###################################
#plotting comparison

fig,ax = plt.subplots()
nbnd = len(DFT[0])
nwan = len(VALS[0])

for i in range(len(DFT)):
    x=np.ones(nbnd)*i
    xw=np.ones(nwan)*i
    if i == 0:
        plt.plot(x, DFT[i],'g.', label='DFT', markersize=6)
        plt.plot(xw, VALS[i],'y.', label='WANNIER', markersize=4)
    else:
        plt.plot(x, DFT[i],'g.', markersize=6)
        plt.plot(xw, VALS[i],'y.', markersize=4)

plt.ylim([fermi - 4.0, fermi + 4.0])

ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('compare.pdf')

#plt.show()
