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
        self.B = np.zeros((3,3))
        if filename is not None:
            if filename.count("_tb.dat"):
                print("load _tb.dat")
                self.load_tb(filename)
            else:
                self.load(filename)
                
        self.sparse=False
        

    def write_tb(self,filename): 

        st = "written by wan_ham\n"

        st += str(self.B[0,0]) + " " + str(self.B[0,1]) + " "+ str(self.B[0,2]) + "\n"
        st += str(self.B[1,0]) + " " + str(self.B[1,1]) + " "+ str(self.B[1,2]) + "\n"
        st += str(self.B[2,0]) + " " + str(self.B[2,1]) + " "+ str(self.B[2,2]) + "\n"

        st += str(self.nwan) + "\n"
        nr = np.shape(self.HR)[0]
        st += str(nr) + "\n"

        
        for i in range(nr):
            st += "    1"
            if i % 15 == 14 and i != nr-1:
                st += "\n"
        st += "\n"
        
        for i in range(nr):
            st += "\n"
            st += str(int(self.R[i,0]))+"  "+str(int(self.R[i,1]))+"  "+str(int(self.R[i,2]))+"\n"
            hr = np.reshape(self.HR[i,:], (self.nwan, self.nwan))
            for b in range(self.nwan):
                for a in range(self.nwan):
                    st += "   "+str(a+1)+"   "+str(b+1)+"   "+str(np.real(hr[a,b]))+"  "+str(np.imag(hr[a,b]))+"\n"

        for i in range(nr):
            st += "\n"
            st += str(int(self.R[i,0]))+"  "+str(int(self.R[i,1]))+"  "+str(int(self.R[i,2]))+"\n"
            rr0 = np.reshape(self.RR[i,:,0], (self.nwan, self.nwan))
            rr1 = np.reshape(self.RR[i,:,1], (self.nwan, self.nwan))
            rr2 = np.reshape(self.RR[i,:,2], (self.nwan, self.nwan))
            for b in range(self.nwan):
                for a in range(self.nwan):
                    st += "   "+str(a+1)+"   "+str(b+1)+"   "+str(np.real(rr0[a,b]))+"  "+str(np.imag(rr0[a,b]))+"  "+"   "+str(np.real(rr1[a,b]))+"  "+str(np.imag(rr1[a,b]))+"  "+"   "+str(np.real(rr2[a,b]))+"  "+str(np.imag(rr2[a,b]))+"\n"
                    
                    
        f = open(filename, 'w')
        f.write(st)
        f.close()
       
        
    def load_tb(self,filename):
        #load a prefix_tb.dat file
        print("load_tb")
        f = open(filename, 'r')
        alllines = f.readlines()
        f.close()
        
        print(alllines[0])
        self.B = np.zeros((3,3))
        self.B[0,0] = float(alllines[1].split()[0])
        self.B[0,1] = float(alllines[1].split()[1])
        self.B[0,2] = float(alllines[1].split()[2])
                                                  
        self.B[1,0] = float(alllines[2].split()[0])
        self.B[1,1] = float(alllines[2].split()[1])
        self.B[1,2] = float(alllines[2].split()[2])
                                                  
        self.B[2,0] = float(alllines[3].split()[0])
        self.B[2,1] = float(alllines[3].split()[1])
        self.B[2,2] = float(alllines[3].split()[2])
        
        
        self.nwan = int(alllines[4].split()[0])
        self.nr   = int(alllines[5].split()[0])


        if self.nr%15 == 0:
            lines_r = int(math.floor(self.nr // 15) )
        else:
            lines_r = int(math.floor(self.nr // 15) + 1)

        print(lines_r, self.nr, )
        
        self.sym_r = np.zeros(self.nr, dtype=float)

        #load sym ops
        for i in range(lines_r):

            num = 6+i
            start = i*15
            end = (i+1)*15
            if end > self.nr:
                end = self.nr
            self.sym_r[start:end] = list(map(float, alllines[num].split()))


        tot = self.nwan**2 * self.nr
        self.H_int = np.zeros((tot,5),dtype=int)
        self.H_val = np.zeros(tot,dtype=np.csingle) 

        self.R_int = np.zeros((tot,5),dtype=int)
        self.R_val = np.zeros((tot, 3),dtype=np.csingle) 



        lines_counter = lines_r+6
        rval = 0
        rind = [0,0,0]
        r_dict = {}
        rind_count = -1
        c=0

        restart = True
        
        for lines in range(lines_counter, len(alllines)):
            sp = alllines[lines].split()
            if len(sp) == 3:
                rind = tuple(map(int, sp[0:3]))
                if rind not in r_dict:
                    r_dict[rind] = rind_count
                rind_count += 1
    
            elif len(sp) == 4:
                
                self.H_int[c,:] = list(rind) + list(map(int, sp[0:2]))
                self.H_val[c] = (float(sp[2]) + 1j * float(sp[3])) / float(self.sym_r[rind_count])
                c += 1

            elif len(sp) == 8:
                if restart:
                    restart = False
                    c=0
                    rind_count = 0
                    print("restart")
                    print(sp)
                    print(rind)
                    
                self.R_int[c,:] = list(rind) + list(map(int, sp[0:2]))
                self.R_val[c,0] = (float(sp[2]) + 1j * float(sp[3])) / float(self.sym_r[rind_count])
                self.R_val[c,1] = (float(sp[4]) + 1j * float(sp[5])) / float(self.sym_r[rind_count])
                self.R_val[c,2] = (float(sp[6]) + 1j * float(sp[7])) / float(self.sym_r[rind_count])
                c += 1
            elif len(sp) == 0:
                continue
            else:
                print("error loading ", sp)

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
        
        self.H = np.zeros((ix,iy,iz,self.nwan,self.nwan),dtype=np.csingle)
        self.R_mat = np.zeros((ix,iy,iz,self.nwan,self.nwan,3),dtype=np.csingle)

        self.ind_dict = {}
        for i in range(self.H_val.shape[0]):
            ind = self.get_ind(self.H_int[i,0:3])

            self.ind_dict[(ind[0],ind[1],ind[2])] = i
            
            nw1 = self.H_int[i,3]
            nw2 = self.H_int[i,4]

            self.H[    ind[0], ind[1], ind[2], nw1-1,nw2-1] = self.H_val[i]
            self.R_mat[ind[0], ind[1], ind[2], nw1-1,nw2-1,:] = self.R_val[i,:]
        
        print('done reshaping1')

        nr = ix*iy*iz
        self.R = np.zeros((nr,3),dtype=float)
        self.HR = np.zeros((nr,self.nwan**2),dtype=np.csingle)
        self.RR = np.zeros((nr,self.nwan**2,3),dtype=np.csingle)

        c=0
        for x in range(nx1, nx2+1):
            for y in range(ny1, ny2+1):
                for z in range(nz1, nz2+1):
                    #                    ind = self.ind_dict[(x,y,z)]
                    ind = self.get_ind([x,y,z])
                    self.R[c,:] = [x,y,z]
                    self.HR[c,:] = np.reshape(self.H[ind[0],ind[1],ind[2], :,:], self.nwan**2)

                    self.RR[c,:,0] = np.reshape(self.R_mat[ind[0],ind[1],ind[2], :,:, 0], self.nwan**2)
                    self.RR[c,:,1] = np.reshape(self.R_mat[ind[0],ind[1],ind[2], :,:, 1], self.nwan**2)
                    self.RR[c,:,2] = np.reshape(self.R_mat[ind[0],ind[1],ind[2], :,:, 2], self.nwan**2)                    

                    c+=1

        if c != nr:
            print('errror ', c, nr)

            
        
    def load_hr(self,filename):
        return self.load(filename)
        
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
        self.H_val = np.zeros(tot,dtype=np.csingle) 

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
        
        self.H = np.zeros((ix,iy,iz,self.nwan,self.nwan),dtype=np.csingle)
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
#        Hkk = np.zeros((self.nwan,self.nwan),dtype=np.csingle)#

#        Hko = np.zeros(self.H.shape,dtype=np.csingle)
        
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
        self.HR = np.zeros((nr,self.nwan**2),dtype=np.csingle)
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

        self.B = np.zeros((3,3))
        self.R_int = self.H_int
        self.R_val = np.zeros( (np.shape(self.H_val)[0], 3), dtype=np.csingle)

        t = np.shape(self.H)
        self.R_mat = np.zeros( (t[0],t[1],t[2],t[3],t[4],3), dtype=np.csingle)
        t = np.shape(self.HR)
        self.RR = np.zeros((t[0], t[1], 3),dtype=np.csingle)
        
            
    def trim(self, val=5e-3):

        Rnew = np.zeros(self.R.shape,dtype=float)
        if self.sparse:
            HRnew = sps.lil_matrix(self.HR.shape,dtype=np.csingle)
        else:
            HRnew = np.zeros(self.HR.shape,dtype=np.csingle)
            
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
###        hk = np.zeros((self.nwan,self.nwan),dtype=np.csingle)
###        sk = np.zeros((self.nwan,self.nwan),dtype=np.csingle)
###        
###        kmat = np.tile(k, (nr,1))
###        exp_ikr = np.exp(-1.0j*2*np.pi* np.sum(kmat*self.R, 1))
###
###        temp = np.zeros(self.nwan**2, dtype=np.csingle)
###        tempS = np.zeros(self.nwan**2, dtype=np.csingle)
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
        
        hk = np.zeros((self.nwan,self.nwan),dtype=np.csingle)
        
        
        kmat = np.tile(k, (nr,1))

        
        exp_ikr = np.exp(1.0j*2*np.pi* np.sum(kmat*self.R, 1))

        temp = np.zeros(self.nwan**2, dtype=np.csingle)
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
        
        hk = sps.csc_matrix((self.nwan,self.nwan),dtype=np.csingle)
        
        
        kmat = np.tile(k, (nr,1))

#        print('kmat.shape',kmat.shape)
#        print('self.R.shape', self.R.shape)

        exp_ikr = np.exp(1.0j*2*np.pi* np.sum(kmat*self.R, 1))

        temp = sps.csc_matrix((1,self.nwan**2), dtype=np.csingle)
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
        


    

