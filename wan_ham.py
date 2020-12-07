import sys
import matplotlib
#matplotlib.use('Agg') #fixes display issues?
import numpy as np
import scipy as sp
import math
import time
import copy as copy
import scipy.sparse as sps
import scipy.sparse.linalg as sps_lin
import scipy.linalg as spl
import matplotlib.pyplot as plt
from scipy import fftpack as FFT

class wan_ham:
    def __init__(self,filename=None):


        print('initializing wan_ham')
        print(        )
        if filename is not None:
            self.load(filename)
        self.sparse=False
        

    def load(self,filename):

        f = open(filename, 'r')
        alllines = f.readlines()
        f.close()
        
        print(alllines[0])

        self.nwan = int(alllines[1].split()[0])
        self.nr   = int(alllines[2].split()[0])


        if self.nr%15 == 0:
            lines_r = int(math.floor(self.nr // 15) )
        else:
            lines_r = int(math.floor(self.nr // 15) + 1)

        print(lines_r, self.nr, )
        
        self.sym_r = np.zeros(self.nr, dtype=float)

        #load sym ops
        for i in range(lines_r):

            num = 3+i
            start = i*15
            end = (i+1)*15
            if end > self.nr:
                end = self.nr
            self.sym_r[start:end] = list(map(float, alllines[num].split()))


        tot = self.nwan**2 * self.nr
        self.H_int = np.zeros((tot,5),dtype=int)
        self.H_val = np.zeros(tot,dtype=complex) 

#        self.S_val = np.zeros(tot,dtype=float) 
        

        c=0
        rnum = 0
        dist = []

#        S_diag = np.zeros(nwan,dtype=float)
        
        for i in range(lines_r+3,lines_r+3+tot):

            rnum = (c)//self.nwan**2
            
            self.H_int[c,:] = list(map(int, alllines[i].split()[0:5]))
            #            print(c, 'H_int', self.H_int[c,:])

#            print("x")
#            print(alllines[i].split())
#            print(alllines[i].split()[5])
#            print(alllines[i].split()[6])
#            print(rnum)
            self.H_val[c] = (float(alllines[i].split()[5]) + 1j * float(alllines[i].split()[6])) / float(self.sym_r[rnum])

#            self.S_val[c] = (float(alllines[i].split()[7]) )/ float(self.sym_r[rnum])
#            dist.append(np.sum(self.H_int[c,0:3]**2)**0.5)

#            if self.H_int[c,0] == 0 and  self.H_int[c,1] == 0 and  self.H_int[c,2] == 0 and self.H_int[c,3] == self.H_int[c,4]:
#                S_diag[ self.H_int[c,3]] =  self.S_val[c]
            
            c+=1

            
            
#        plt.plot(dist, (self.H_val.real), 'bx')
#        plt.plot(dist, (self.H_val.imag), 'r+')
#        plt.plot(dist, self.S_val, 'g*')
#        plt.ylim(-3,3)
#        plt.show()
#        print(self.H_int[0,:])
#        print(self.H_val[0])

#        print(self.H_int[-1,:])
#        print(self.H_val[-1])
        
        print('loaded ', filename)
        print('nwan: ', self.nwan)
        print('nr:   ', self.nr)
        print(        )

        #reshape

        nx1 = np.min(self.H_int[:,0])
        nx2 = np.max(self.H_int[:,0])
        ny1 = np.min(self.H_int[:,1])
        ny2 = np.max(self.H_int[:,1])
        nz1 = np.min(self.H_int[:,2])
        nz2 = np.max(self.H_int[:,2])

        self.ind = [[nx1,nx2],[ny1,ny2],[nz1,nz2]]

        ix = nx2-nx1+1
        iy = ny2-ny1+1
        iz = nz2-nz1+1

        print('H size', ix,iy,iz,self.nwan,self.nwan)
        
        self.H = np.zeros((ix,iy,iz,self.nwan,self.nwan),dtype=complex)
#        self.S = np.zeros((ix,iy,iz,self.nwan,self.nwan),dtype=float)

        self.ind_dict = {}
        for i in range(self.H_val.shape[0]):
            ind = self.get_ind(self.H_int[i,0:3])

            self.ind_dict[(ind[0],ind[1],ind[2])] = i
            
            nw1 = self.H_int[i,3]
            nw2 = self.H_int[i,4]

            self.H[ind[0], ind[1], ind[2], nw1-1,nw2-1] = self.H_val[i]
#            self.S[ind[0], ind[1], ind[2], nw1-1,nw2-1] = self.S_val[i] / S_diag[nw1]**0.5/S_diag[nw2]**0.5

            
            
#        Sk = FFT.fftn(self.S, axes=[0,1,2])
#        Hk = FFT.fftn(self.H, axes=[0,1,2])

#        Skk = np.zeros((self.nwan,self.nwan),dtype=float)
#        Skk2 = np.zeros((self.nwan,self.nwan),dtype=float)        
#        Hkk = np.zeros((self.nwan,self.nwan),dtype=complex)#

#        Hko = np.zeros(self.H.shape,dtype=complex)
        
#        for k1 in range(0,Sk.shape[0]):
#            for k2 in range(0,Sk.shape[1]):
#                for k3 in range(0,Sk.shape[2]):
#                    Skk[:,:] = Sk[k1,k2,k3,:,:]
#                    Hkk[:,:] = Hk[k1,k2,k3,:,:]
#                    Skk2[:,:] = np.linalg.inv(spl.sqrtm(Skk))

#                    Hko[k1,k2,k3,:,:] = np.dot(np.dot(Skk2,Hkk), Skk2)
#                    Hko[k1,k2,k3,:,:] = Hkk

#        HRo = FFT.ifftn(Hko, axes=[0,1,2])
#        dist2 = []
#        for x in range(nx1, nx2+1):
#            for y in range(ny1, ny2+1):
#                for z in range(nz1, nz2+1):
#                    ind = self.get_ind([x,y,z])
#                    plt.plot((ind[0]**2+ind[1]**2+ind[2]**2)**0.5, HRo[x,y,z,0,0].real, 'c.')
#                    
#        plt.show()
#        print(Sk.shape, Hk.shape)
#        exit()
        
        print('done reshaping1')

        nr = ix*iy*iz
        self.R = np.zeros((nr,3),dtype=float)
        self.HR = np.zeros((nr,self.nwan**2),dtype=complex)
#        self.SR = np.zeros((nr,self.nwan**2),dtype=float)

        c=0
        for x in range(nx1, nx2+1):
            for y in range(ny1, ny2+1):
                for z in range(nz1, nz2+1):
                    #                    ind = self.ind_dict[(x,y,z)]
                    ind = self.get_ind([x,y,z])
                    self.R[c,:] = [x,y,z]
                    self.HR[c,:] = np.reshape(self.H[ind[0],ind[1],ind[2], :,:], self.nwan**2)
#                    self.SR[c,:] = np.reshape(self.H[ind[0],ind[1],ind[2], :,:], self.nwan**2)
                    c+=1

        if c != nr:
            print('errror ', c, nr)

    def trim(self, val=5e-3):

        Rnew = np.zeros(self.R.shape,dtype=float)
        if self.sparse:
            HRnew = sps.lil_matrix(self.HR.shape,dtype=complex)
        else:
            HRnew = np.zeros(self.HR.shape,dtype=complex)
            
        c=0

        oldsize=self.R.shape[0]
        
        for i in range(self.R.shape[0]):
            if np.max(np.abs(self.HR[i,:])) > val:
                Rnew[c,:] = self.R[i,:]                
                if self.sparse:
                    HRnew[c,:] = sps.lil_matrix(self.HR[i,:])
                else:
                    HRnew[c,:] = self.HR[i,:]

                c += 1

        self.R = Rnew[0:c,:]
        self.HR = HRnew[0:c,:]

        print('trimmed', oldsize, c)
        
    def get_ind(self,nxyz):

        return [nxyz[0] - self.ind[0][0], nxyz[1] - self.ind[1][0], nxyz[2] - self.ind[2][0]]
        

###    def solve_ham_S(self,k, proj=None):
###
###        nr = self.R.shape[0]
###        
###        hk = np.zeros((self.nwan,self.nwan),dtype=complex)
###        sk = np.zeros((self.nwan,self.nwan),dtype=complex)
###        
###        kmat = np.tile(k, (nr,1))
###        exp_ikr = np.exp(-1.0j*2*np.pi* np.sum(kmat*self.R, 1))
###
###        temp = np.zeros(self.nwan**2, dtype=complex)
###        tempS = np.zeros(self.nwan**2, dtype=complex)
###        for i in range(nr):
###            temp += exp_ikr[i]*self.HR[i,:]
####            tempS += exp_ikr[i]*self.SR[i,:]            
###
###        hk = np.reshape(temp, (self.nwan, self.nwan)) 
####        sk = np.reshape(tempS, (self.nwan, self.nwan)) 
###            
###        hk = (hk + hk.T.conj())/2.0
####        sk = (sk + sk.T.conj())/2.0
###
####        sk2 = np.linalg.inv(spl.sqrtm(sk))
###        hk2 = np.dot(np.dot(sk2, hk),sk2)
###
###        return hk2


        
    
    
    def solve_ham(self,k, proj=None):


#        print('solve', self.nwan, self.R.shape, self.HR.shape)
        
        nr = self.R.shape[0]
        
        hk = np.zeros((self.nwan,self.nwan),dtype=complex)
        
        
        kmat = np.tile(k, (nr,1))

        
        exp_ikr = np.exp(1.0j*2*np.pi* np.sum(kmat*self.R, 1))

        temp = np.zeros(self.nwan**2, dtype=complex)
        for i in range(nr):
            temp += exp_ikr[i]*self.HR[i,:]

        hk = np.reshape(temp, (self.nwan, self.nwan)) 
            
        hk = (hk + hk.T.conj())/2.0

        
#        print("hk ", k)
#        print(hk)
        
        tb = time.time()
        val, vect = np.linalg.eigh(hk)
#        print('TIME eig dense', time.time()-tb)

        if proj is not None:
            p = np.real(np.sum(vect[proj,:]*np.conj(vect[proj, :]), 0))
        else:
            p = np.ones(val.shape)

#        print('proj', np.sum(np.sum(p)))
        
        return val.real, vect, p
        

    def solve_ham_sparse(self,k,  num_eigs,fermi=0.0, proj=None):


#        print('solve', self.nwan, self.R.shape, self.HR.shape)
        
        nr = self.R.shape[0]
        
        hk = sps.csc_matrix((self.nwan,self.nwan),dtype=complex)
        
        
        kmat = np.tile(k, (nr,1))

#        print('kmat.shape',kmat.shape)
#        print('self.R.shape', self.R.shape)

        exp_ikr = np.exp(1.0j*2*np.pi* np.sum(kmat*self.R, 1))

        temp = sps.csc_matrix((1,self.nwan**2), dtype=complex)
        for i in range(nr):
#            print('sps.csr_matrix(self.HR[i,:]).shape',sps.csc_matrix(self.HR[i,:]).shape)
#            print(exp_ikr[i])
#            print('temp.shape', temp.shape)
            temp += exp_ikr[i]*sps.csc_matrix(self.HR[i,:])

        hk = sps.csc_matrix.reshape(temp, (self.nwan, self.nwan)) 
            
        hk = sps.csc_matrix((hk + hk.T.conj())/2.0)

        tb = time.time()
        val, vect = sps_lin.eigsh(hk, k=num_eigs, sigma=fermi, which='LM')
        print('TIME eigsh', time.time()-tb)
        
#        val=np.zeros(num_eigs)
#       vect = np.zeros((self.nwan, num_eigs))

        if proj is not None:
            p = np.real(np.sum(vect[proj,:]*np.conj(vect[proj, :]), 0))
        else:
            p = np.ones(val.shape)

#        print('proj', np.sum(np.sum(p)))
        
        return val.real, vect, p
        


    

