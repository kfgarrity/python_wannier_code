import sys
import matplotlib as matplotlib
#matplotlib.use('Agg') #fixes display issues?
import numpy as np
import scipy as sp
import math
import cmath
import time
import copy as copy
import matplotlib.pyplot as plt 
import scipy.sparse as sps

import scipy.optimize as opt

from wan_ham import wan_ham


class ham_ops:

    def __init__(self):

        print( 'ham_ops')
        self.num_k = 50
        

    def unfold(self, ham_prim, ham_ss, supercell, kprim):


        t0=time.time()
        
        Kss = np.array(kprim)*np.array(supercell)
        Kss = Kss % 1.0
        
        val_prim, vect_prim,_ = ham_prim.solve_ham(kprim)
        val_ss, vect_ss,_ = ham_ss.solve_ham(Kss)

        P = np.zeros((ham_prim.nwan,ham_ss.nwan), dtype=float)


#        t1=time.time()
#        print( 'unfold inner time SOLVE', t1-t0)
#        t0=t1
        
#        print( 'kprim', kprim, Kss)
#        print( 'val_prim')
#        print( val_prim)
#        print( 'val_ss')
#        print( val_ss)
#        print()
#        print( 'vect_prim')
#        print( vect_prim)
#        print()
#        print( 'vect_ss')
#        print( vect_ss)
#        print()

        ss = np.zeros((np.prod(supercell),3), dtype=float)
        c=0
        for x in range(supercell[0]):
            for y in range(supercell[1]):
                for z in range(supercell[2]):
                    ss[c,:] = [x,y,z]
                    c+=1
                    

        exp_ikr = []
        r = []
        for i in range(np.prod(supercell)):
            exp_ikr.append(np.exp(1.0j*2*np.pi*np.dot(kprim, ss[i,:])))    #i
            r.append(range(ham_prim.nwan*i, ham_prim.nwan*(i+1))) #i

#        t1=time.time()
#        print( 'unfold inner time EXP_ijk', t1-t0)
#        t0=t1

        vect_prim_ss = np.zeros((ham_prim.nwan, vect_ss.shape[0]), dtype=np.csingle)
                           
        for n in range(ham_prim.nwan):
            for i in range(np.prod(supercell)):
                vect_prim_ss[n,r[i]] = vect_prim[n,:]*exp_ikr[i]
                
#        t1=time.time()
#        print( 'unfold inner time vect_prim_ss', t1-t0)
#        t0=t1
        

        t=np.dot(vect_prim_ss,vect_ss)
        
        P[:,:] = (t*t.conj()).real

        
#        for n in range(ham_prim.nwan):
#            for N in range(ham_ss.nwan):
#                t = np.dot(vect_prim_ss[n,:], vect_ss[:,N]) #n,N,i
#                P[n,N] = (t*t.conj()).real
                
                #                temp += t
                    
#                P[n,N] = (temp*temp.conj()).real

#        for n in range(ham_prim.nwan):
#            for N in range(ham_ss.nwan):
#                temp = 0.0
#                for i in range(np.prod(supercell)):

#                    t = np.dot(vect_prim[n,:]*exp_ikr[i], vect_ss[r[i],N]) #n,N,i
#                    temp += t

                    
#                P[n,N] = (temp*temp.conj()).real
                

        P = P / np.prod(supercell)

#        t1=time.time()
#        print( 'unfold inner time FOLDING', t1-t0)
#        t0=t1

        
#        print( 'P')
#        print( P)
#        print( '-')
                
        return P, val_prim, val_ss


    def delta(self,energy, temp):

        return np.exp(-0.5 * energy**2/temp**2) / (temp * ( 2 * np.pi)**0.5)
    

    def unfold_bandstructure(self, ham_prim, ham_ss, supercell, kpath, temp=0.05, fermi=0.0, yrange=None, names=None, pdfname='unfold.pdf', num_energies=100):

        K = self.generate_kpath(kpath)
    
        nk=len(K)
        print( 'nktot', nk)
        print()

        t0=time.time()
        
        nwan = ham_prim.nwan
        vals_prim = np.zeros((nk,nwan), dtype=float)
        vals_ss = np.zeros((nk,ham_ss.nwan), dtype=float)
        Pall = np.zeros((nk, nwan, ham_ss.nwan))
        for i,k in enumerate(K):

            P, val_prim, val_ss = self.unfold(ham_prim, ham_ss, supercell, k)
            vals_prim[i,:] = val_prim - fermi
            Pall[i,:,:] = P
            vals_ss[i,:] = val_ss -fermi

            #        fig, ax = plt.subplots()


        t1=time.time()
        print( 'unfold time', t1-t0)
        t0=t1
        
        if yrange is not None:
            d = (yrange[1]-yrange[0])*0.05
        else:
            yrange = [np.min(np.min(vals_prim))-0.01, np.max(np.max(vals_prim))+0.01 ]
            d = (yrange[1]-yrange[0])*0.05

        print( 'yrange', yrange)
            

        #compute spectral function

#        plt.plot(vals_prim, 'y--', zorder=1)

        energies = np.arange(yrange[0], yrange[1], (yrange[1]-yrange[0])/float(num_energies))

        image = np.zeros((num_energies, nk),dtype=float)

        Ps = np.sum(Pall, 1)

        max_band = np.max(vals_ss, 0)
        min_band = np.min(vals_ss, 0)        

        N_start = 0
        N_end = ham_ss.nwan

        for i in range(ham_ss.nwan):
            if max_band[i] < yrange[0] - temp * 10:
                N_start = i
            if min_band[i] < yrange[1] + temp * 10:
                N_end = i+1

        N_start = max(N_start,0)
        N_end = min(N_end, ham_ss.nwan)
        
        print( 'Nstart, Nend', N_start,N_end,max_band[N_start],min_band[N_end-1])
                
        for i,k in enumerate(K):
            for j in range(num_energies):
                en= energies[j]
                ind = num_energies-j-1
#                for N in range(ham_ss.nwan):
                for N in range(N_start,N_end):
                    d = self.delta(en-vals_ss[i,N], temp)

                    image[ind,i] += Ps[i,N] * d                    
#                    image[ind,i] += np.sum(Pall[i,:,N] * d)
#                    for n in range(ham_prim.nwan):
#                        image[num_energies-j-1,i] += Pall[i,n,N] * d




        t1=time.time()
        print( 'unfold make image time', t1-t0)
        t0=t1

        limits = (0,nk-1,yrange[0], yrange[1])
        self.plot_image(image, limits, names, pdfname=pdfname)

        t1=time.time()
        print( 'unfold plot image time', t1-t0)
        t0=t1
        
        return image, limits


    def plot_image(self,image, limits, names=None, pdfname='unfold.pdf'):

        fig, ax = plt.subplots(figsize=(3.5,3.5))            

        d =  (limits[3]- limits[2])*0.05
        self.names_plt(names,[limits[2], limits[3]], d)
                        
        plt.ylim([limits[2], limits[3]])

        m = np.max(np.max(image))
        plt.imshow(image, aspect="auto", extent=limits, vmax = m*0.6)
#        plt.imshow(image, aspect="equal", extent=limits, vmax = m*0.6)

        plt.xlim([limits[0], limits[1]])

        ax.set_xticklabels([])
        ax.set_xticks([])

        plt.yticks(fontsize=12)
        plt.ylabel("Energy - E$_F$ (eV)", fontsize=14)

        plt.tight_layout()
        
        plt.savefig(pdfname)


    def chern_number(self,ham,nocc, k1,k2, nk1, nk2):

        K = np.zeros((nk1,nk2, 3),dtype=float)

        k1=np.array(k1,dtype=float)
        k2=np.array(k2,dtype=float)

#initial grid
        for c1 in range(nk1):
            for c2 in range(nk2):
                K[c1,c2,:] = k1 * float(c1)/float(nk1) + k2 * float(c2)/float(nk2)


                
        VECT0 = np.zeros((nk1, ham.nwan, ham.nwan),dtype=np.csingle)
        VECT1 = np.zeros((nk1, ham.nwan, ham.nwan),dtype=np.csingle)
        
        gap_min = 100000000000.0
        val_max = -10000000000000.0
        cond_min = 10000000000.0

        val_max_m1 = -10000000000000.0
        cond_min_p1 = 10000000000.0

        gap_min2 = 100000000000.0
        
        Chern = 0.0

        def calc_line(c1,nk2, K, VECT, nocc, gap_min, val_max, cond_min,gap_min2,val_max_m1,cond_min_p1):
            nwan = VECT.shape[1]
            for c2 in range(nk2):
                k=K[c1,c2,:]
                val, vect,p = ham.solve_ham(k,proj=None)
                VECT[c2,:,:] = vect

                if (val[nocc]-val[nocc-1]) < gap_min:
                    gap_min = val[nocc]-val[nocc-1]

                if val[nocc-1] > val_max:
                   val_max = val[nocc-1]

                if val[nocc] < cond_min:
                   cond_min = val[nocc]

                if nocc+1 < nwan and nocc-2 >= 0:
#                    if (val[nocc] - val[nocc-1]) < gap_min2:
#                        gap_min2 = val[nocc] - val[nocc-1]
                    if (val[nocc]+val[nocc+1]-val[nocc-1]-val[nocc-2])/2.0 < gap_min2:
                        gap_min2 = (val[nocc]+val[nocc+1]-val[nocc-1]-val[nocc-2])/2.0

                if nocc >= 2 and val[nocc-2] < val_max_m1:
                   val_max_m1 = val[nocc-2]

                if nocc+1 < nwan and val[nocc+1] < cond_min_p1:
                   cond_min_p1 = val[nocc+1]
                        
            return gap_min, val_max, cond_min, gap_min2,val_max_m1,cond_min_p1


        def plaquette(V, nocc):

            order = [0,1,2,3,0]
            phi = 0.0
            for i in range(4):
                M = np.dot(V[order[i]][:,0:nocc].T.conj(), V[order[i+1]][ :,0:nocc])
                phi = phi - (cmath.log(np.linalg.det(M))).imag

            return phi




        #get first line
        gap_min, val_max, cond_min,gap_min2,val_max_m1,cond_min_p1 = calc_line(0, nk2, K, VECT0, nocc, gap_min, val_max, cond_min,gap_min2,val_max_m1,cond_min_p1)

        for c1 in range(nk1):
            c1p = (c1+1)%nk1
            gap_min, val_max, cond_min,gap_min2,val_max_m1,cond_min_p1 = calc_line(c1p, nk2, K, VECT1, nocc, gap_min, val_max, cond_min,gap_min2,val_max_m1,cond_min_p1)
            for c2 in range(nk2):
                c2p = (c2+1)%nk2
                klist = [[0,c2],[1,c2],[1,c2p],[0,c2p],[0,c2]]

                phi = 0.0
                V = []
                for i in range(4):
                    if klist[i][0] == 0:
                        vl = VECT0[klist[i][1],:,:]
                    else:
                        vl = VECT1[klist[i][1],:,:]
                    V.append(vl)

                phi = plaquette(V, nocc)

#                    M = np.dot(vl[klist[i][1], :,0:nocc].T.conj(), vr[klist[i+1][1], :,0:nocc])
#                    phi = phi - (cmath.log(np.linalg.det(M))).imag

                if phi > np.pi:
                    phi = phi - 2*np.pi
                elif phi <  -np.pi:
                    phi = phi + 2*np.pi
                    
                if abs(phi)/(2*np.pi) > 0.5:
                    print( 'WARNING in chern calculation, phi', c1, c2, phi, ' try more k-points')
                    print( 'activate recursive')

                    K1 = np.zeros((3,3,3),dtype=float)
                    print( 'K1')
                    for c1a in range(3):
                        for c2a in range(3):                        
                            K1[c1a,c2a,:] = K[c1,c2,:] + k1 * float(c1a)/float(nk1)/2.0 + k2 * float(c2a)/float(nk2)/2.0
                            for i in range(3):
                                if abs(K1[c1a,c2a,i]-1.0) < 1e-5:
                                    K1[c1a,c2a,i]=0.0
                            
                            print( c1a, c2a, K1[c1a,c2a,:])

                    ctemp, gap_min_t, val_max_t, cond_min_t = self.chern_number_simple(ham,nocc, k1,k2, 2,2, Kmat=K1, usemod=False)
                    gap_min=min(gap_min,gap_min_t)
                    val_max=max(val_max,val_max_t)
                    cond_min=min(cond_min,cond_min_t)                                        

                    phi_temp = phi/(2.0*np.pi)

                    add_int = round(ctemp - phi_temp )

#                    print( 'ctemp', ctemp, phi_temp, phi_temp+add_int, add_int)
                    
                    Chern += phi_temp+add_int
                    
                else:            
                    Chern += phi / (2.0*np.pi)

                    
            VECT0[:,:,:] = VECT1[:,:,:]


            

                   
#        print( 'minimum direct gap', gap_min, 'indirect gap', cond_min-val_max)
        print( 'minimum_direct_gap', gap_min, 'indirect_gap', cond_min-val_max, ' direct_gap_avg2 ', gap_min2)
        
        direct_gap = gap_min
        indirect_gap = cond_min-val_max
        
        
#        Chern = 0.0
#        for c1 in range(nk1):
#            for c2 in range(nk2):
#
#                c1p = (c1+1)%nk1
#                c2p = (c2+1)%nk2
#
#                klist = [[c1,c2],[c1p,c2],[c1p,c2p],[c1,c2p],[c1,c2]]
#
#                phi = 0.0
#                for i in range(4):
#                    M = np.dot(VECT[klist[i][0], klist[i][1], :,0:nocc].T.conj(), VECT[klist[i+1][0], klist[i+1][1], :,0:nocc])
#                    phi = phi - (cmath.log(np.linalg.det(M))).imag
#
#                if phi > np.pi:
#                    phi = phi - 2*np.pi
#                elif phi <  -np.pi:
#                    phi = phi + 2*np.pi
#                    
#                if abs(phi) > 0.75:
#                    print( 'WARNING in chern calculation, phi', c1, c2, phi, ' try more k-points')
#                    
#                Chern += phi / (2.0*np.pi)

        return Chern, direct_gap, indirect_gap

        
        
    def chern_number_simple(self,ham,nocc, k1,k2, nk1, nk2, Kmat = None, usemod=True):

        if Kmat is None:
        
            K = np.zeros((nk1,nk2, 3),dtype=float)

            k1=np.array(k1,dtype=float)
            k2=np.array(k2,dtype=float)

            #        print( 'K')

            for c1 in range(nk1):
                for c2 in range(nk2):
                    K[c1,c2,:] = k1 * float(c1)/float(nk1) + k2 * float(c2)/float(nk2)

        else:
            K = Kmat
            
                    
                #                print( c1, c2, 'K', K[c1,c2,:])

                #        print()

        if usemod:

            lim1=nk1
            lim2=nk2

        else:

            lim1=nk1+1
            lim2=nk2+1
            
        VECT = np.zeros((lim1, lim2, ham.nwan, ham.nwan),dtype=np.csingle)
        gap_min = 100000000000.0
        val_max = -10000000000000.0
        cond_min = 10000000000.0

        val_max_m1 = -10000000000000.0
        cond_min_p1 = 10000000000.0

        gap_min2 = 100000000000.0
        
        nwan = ham.nwan
        
        for c1 in range(lim1):
            for c2 in range(lim2):
                k=K[c1,c2,:]
                val, vect,p = ham.solve_ham(k,proj=None)
                VECT[c1,c2,:,:] = vect
                #                print( 'val')
                #                print( val)
                if (val[nocc]-val[nocc-1]) < gap_min:
                    gap_min = val[nocc]-val[nocc-1]

#                if nocc+1 < nwan and nocc-2 >= 0:
#                    if (val[nocc]+val[nocc+1]-val[nocc-1]-val[nocc-2])/2.0 < gap_min2:
#                        gap_min2 = (val[nocc]+val[nocc+1]-val[nocc-1]-val[nocc-2])/2.0
                    
                if val[nocc-1] > val_max:
                   val_max = val[nocc-1]

                if val[nocc] < cond_min:
                   cond_min = val[nocc]

#                if nocc >= 2 and val[nocc-2] < val_max_m1:
#                   val_max_m1 = val[nocc-2]
#
#                if nocc+1 < nwan and val[nocc+1] < cond_min_p1:
#                   cond_min_p1 = val[nocc+1]
                   
                   
        print( 'minimum_direct_gap', gap_min, 'indirect_gap', cond_min-val_max)
        direct_gap = gap_min
        indirect_gap = cond_min-val_max
        
        
        Chern = 0.0
        for c1 in range(nk1):
            for c2 in range(nk2):

                if usemod:
                    c1p = (c1+1)%nk1
                    c2p = (c2+1)%nk2
                else:
                    c1p = (c1+1)
                    c2p = (c2+1)

                klist = [[c1,c2],[c1p,c2],[c1p,c2p],[c1,c2p],[c1,c2]]

                phi = 0.0
                for i in range(4):
                    M = np.dot(VECT[klist[i][0], klist[i][1], :,0:nocc].T.conj(), VECT[klist[i+1][0], klist[i+1][1], :,0:nocc])
                    phi = phi - (cmath.log(np.linalg.det(M))).imag

                if phi > np.pi:
                    phi = phi - 2*np.pi
                elif phi <  -np.pi:
                    phi = phi + 2*np.pi
                    
                if abs(phi) > 0.75:
                    print( 'WARNING in chern calculation, phi', c1, c2, phi, ' try more k-points')
                    
                Chern += phi / (2.0*np.pi)

        if usemod:
            return Chern, direct_gap, indirect_gap
        else:
            return Chern, direct_gap, val_max,cond_min

    def names_plt(self,names, yrange, d):

        if names is not None:
            for c,n in enumerate(names):
                pos = self.num_k * c
                
                plt.text(pos-3.0,yrange[0]-d*1.5 , n, fontsize=24)
                plt.plot([pos, pos],[yrange[0], yrange[1]], '--', color='lightgrey',linewidth=0.5, zorder=1)
        
    
    def generate_kpath(self,kpath):

        K = []

        for i in range(len(kpath)-1):
            dk = np.array(kpath[i+1]) - np.array(kpath[i])
            for j in range(self.num_k):
                K.append(np.array(kpath[i]) + dk * (float(j)/float(self.num_k)))

                #                print( np.array(kpath[i]) + dk * (float(j)/float(self.num_k)))
                
        K.append(kpath[-1])

        return K
    
    def band_struct(self,ham, kpath, proj = None, yrange=None, names=None, fermi=0.0, pdfname='bs.pdf', nbands=8, colorbar = True, show=False):

        K = self.generate_kpath(kpath)

        nk=len(K)
        print( 'nktot', nk)
        print()
        
        nwan = ham.nwan

        if ham.sparse:
            if nbands >= nwan:
                nbands = nwan - 2
            vals = np.zeros((nk,nbands), dtype=float)
            projs = np.zeros((nk,nbands), dtype=float)
                
        else:
            nbands=nwan
            
            vals = np.zeros((nk,nwan), dtype=float)
            projs = np.zeros((nk,nwan), dtype=float)

        
        for i,k in enumerate(K):
            #            print( k)

            if ham.sparse:
                val, vect,p = ham.solve_ham_sparse(k,nbands, fermi=fermi, proj=proj)
                ind = np.argsort(np.array(val))
                val = val[ind]
                vect = vect[:,ind]
                p = p[ind]
                
            else:
                val, vect,p = ham.solve_ham(k,proj=proj)

            vals[i,:] = val - fermi
            projs[i,:] = p

            
        fig, ax = plt.subplots()

        
        if yrange is not None:
            plt.ylim(yrange)
            d = (yrange[1]-yrange[0])*0.05
        else:
            yrange = [np.min(np.min(vals))-0.01, np.max(np.max(vals))+0.01 ]
            plt.ylim(yrange)
            d = (yrange[1]-yrange[0])*0.05



        self.names_plt(names, yrange, d)
        

                
        x=np.array(range(vals.shape[0]))
        if proj is None:
            if ham.sparse:
                plt.plot(vals, 'b.')
            else:
                plt.plot(vals, 'b')
            colorbar = False
        else:

            X = np.tile(x.T, (nbands,1)).T
#            print( x.shape)
#            print( vals.shape)
#            print( projs.shape)
            #            for i in range(nwan):

            if not ham.sparse:
                plt.plot(X, vals, 'k', linewidth=0.5,zorder=1)

            
#            projs = np.minimum(projs, 0.6)
#            projs = np.maximum(projs, 0.0)
            
            plt.scatter(X, vals, s=5, c=projs, zorder=2)




        plt.xlim([x[0], x[-1]])
        plt.yticks(fontsize=24)
        plt.ylabel("Energy - E$_F$ (eV)", fontsize=26)

        d = (yrange[1]-yrange[0])*0.05

        ax.set_xticklabels([])
        ax.set_xticks([])




        if colorbar:
            cbar = plt.colorbar()
            cbar.set_ticks([])

        plt.tight_layout()
                
        plt.savefig(pdfname)
        if show:
            plt.show()

        return vals

    def fd(self,val, fermi, temp):

        arg = (val - fermi)/temp #fixes overflow
        arg[arg>100] = 100
        
        n = (np.exp( arg) +1)**-1
        
        energy = np.sum(n * val)
        
        return np.sum(n), energy
    
    def fermi(self, ham, nocc, temp, kmesh):
        if type(kmesh) is int:
            kmesh = [kmesh, kmesh,kmesh]
        K = []
        for kx in np.arange(0,1.0001,1./float(kmesh[0]-1)):
            for ky in np.arange(0,1.0001,1./float(kmesh[1]-1)):
                if kmesh[2] > 1:
                    for kz in np.arange(0,1.0001,1./float(kmesh[2]-1)):
                        K.append([kx,ky,kz])
                else:
                    K.append([kx,ky,0])

                    
        if type(nocc) is int:
            nocc = [nocc]

        nk = len(K)
        print( kmesh, nk)
        print()
        VAL = []
        for i,k in enumerate(K):
            val, vect,p = ham.solve_ham(k,proj=None)
            VAL.append(val)

        VAL = np.array(VAL)

        print( VAL.shape)


#        print( 'start end', start, end)

        FERMI  = []
        print( 'nocc,  fermi')
        for no in nocc:

            if no >= 1:
                start = np.min(VAL[:,int(math.floor(no)-1)])
            else:
                start = np.min(VAL[:,0])
                
            if no < ham.nwan:
                end = np.max(VAL[:,int(math.ceil(no))])
            else:
                end = np.max(VAL[:,-1])

            #binary search
            for i in range(20):
                mid = (start+end)/2.0
                n, en = self.fd(VAL[:], mid, temp)
                n = n / float(nk)
#                print( i,'search fermi', mid, n)
                if n > no:
                    end = mid
                else:
                    start = mid
            print( no, "\t", mid)
            FERMI.append(mid)
        print()

        ENERGY = []
        for no, fermi in zip(nocc, FERMI):
            n,en =  self.fd(VAL, fermi, temp)
            ENERGY.append(en/ float(nk))
        

        
        return FERMI, ENERGY
            
    def generate_kgrid(self, grid):

        t = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                for k in range(grid[2]):
                    t.append([float(i)/(float(grid[0])) , float(j)/(float(grid[1])), float(k)/(float(grid[2]))])
        return t


    def generate_kgrid_gamma(self, grid):

        g1 = np.linspace(-0.25, 0.25, grid[0])
        g2 = np.linspace(-0.25, 0.25, grid[1])
        g3 = np.linspace(-0.25, 0.25, grid[2])
        t = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                for k in range(grid[2]):
                    t.append([g1[i] , g2[j], g3[k]])
        
        return t
    

    def fermi_surf_2d(self, ham, fermi, origin, k1, k2, nk1, nk2, sig):
        K = np.zeros((nk1+1,nk2+1, 3),dtype=float)

        k1=np.array(k1,dtype=float)
        k2=np.array(k2,dtype=float)

        for c1 in range(nk1+1):
            for c2 in range(nk2+1):
                K[c1,c2,:] = origin + k1 * float(c1)/float(nk1) + k2 * float(c2)/float(nk2)


        VALS = np.zeros((nk1+1,nk2+1, ham.nwan))
        IMAGE = np.zeros((nk1+1, nk2+1))

        for c1 in range(nk1+1):
            for c2 in range(nk2+1):
                k = K[c1,c2,:]
                val, vect,p = ham.solve_ham(k,proj=None)

                VALS[c1, c2, :] = val

                #                IMAGE[c1,c2] = np.sum(np.abs(1/(val - fermi + sig)))
                IMAGE[c1,c2] = np.sum( np.exp( -(val - fermi)**2 / sig**2))



        points = self.fermi_surf_points_2d(VALS, fermi)

        return IMAGE, VALS, points

    def fermi_surf_points_2d(self, VALS, fermi, K=None):
        v = VALS - fermi
        points = []
        nk1 = VALS.shape[0]
        nk2 = VALS.shape[1]
        nwan = VALS.shape[2]

        if K is None:
            k1 = np.array([1.0,0.0,0.0])
            k2 = np.array([0.0,1.0,0.0])
            origin = np.array([-0.5,-0.5,0.0])
            K = np.zeros((nk1,nk2, 3),dtype=float)
            for c1 in range(nk1):
                for c2 in range(nk2):
                    K[c1,c2,:] = origin + k1 * float(c1)/float(nk1-1) + k2 * float(c2)/float(nk2-1)
            

        for c1 in range(nk1):
            for c2 in range(nk2):
                c1p = c1+1
                c2p = c2+1
                if c1p < nk1:
                    for n in range(nwan):
                        if (v[c1,c2, n] < 0.0 and v[c1p,c2, n] > 0.0) or (v[c1,c2, n] > 0.0 and v[c1p,c2, n] < 0.0):
                            x = -v[c1,c2, n] / (v[c1p,c2, n] - v[c1,c2, n])
                            k1 = K[c1,c2,:]
                            k2 = K[c1p,c2,:]
                            kinterp = k1 * (1-x) + k2*x                            
                            points.append(kinterp.tolist())

                            #                            points.append([ (float(c1)+x)/(nk1-1), float(c2)/(nk2-1), n])

                if c2p < nk2:
                    for n in range(nwan):
                        if (v[c1,c2, n] < 0.0 and v[c1,c2p, n] > 0.0) or (v[c1,c2, n] > 0.0 and v[c1,c2p, n] < 0.0):
                            x = -v[c1,c2, n] / (v[c1,c2p, n] - v[c1,c2, n])
                            k1 = K[c1,c2,:]
                            k2 = K[c1,c2p,:]
                            kinterp = k1 * (1-x) + k2*x                            
                            points.append(kinterp.tolist())
#                            points.append([float(c1) / (nk1-1), (float(c2)+x)/(nk2-1), n])
        return points





    def get_gap_points(self, directgap, thresh=0.0005, val=None):
        (n1,n2,n3) = np.shape(directgap)
        points = []
        gap = []
        v = []        
        for c1 in range(n1):
            for c2 in range(n2):
                for c3 in range(n3):
                    if directgap[c1,c2,c3] < thresh:
                        points.append([c1,c2,c3])
                        gap.append(directgap[c1,c2,c3])
                        if not( val is None):
                            v.append(val[c1,c2,c3])
                            
        print("n points ", len(points))
        return np.array(points), np.array(gap), np.array(v)
    
        
    def find_nodes(self, ham, num_occ, origin=[-0.5,-0.5,-0.5], k1=[1,0,0],k2=[0,1,0],k3=[0,0,1], nk1=20, nk2=20, nk3=20,sig=[0.01, 0.02, 0.05, 0.1, 0.2,0.3], thresh=-10, node_tol = 0.0015, use_min=True):

        K = np.zeros((nk1*4,nk2*4,nk3*4,3),dtype=float)

        k1=np.array(k1,dtype=float)
        k2=np.array(k2,dtype=float)
        k3=np.array(k3,dtype=float)

        for c1 in range(4*nk1):
            for c2 in range(4*nk2):
                for c3 in range(4*nk3):
                    K[c1,c2,c3,:] = origin + k1 * float(c1)/float(nk1*4) + k2 * float(c2)/float(nk2*4) + k3 * float(c3)/float(nk3*4) 


                    #        VALS = np.zeros((nk1,nk2, ham.nwan))


                    
        IMAGE = np.zeros((nk1*4, nk2*4, nk3*4, len(sig)))
        DIRECTGAP = np.zeros((nk1*4, nk2*4, nk3*4))
        DIRECTGAP[:,:,:] = 100.0

        VAL = np.ones((nk1*4, nk2*4, nk3*4)) * 100.0
        
#        sig_0 = 0.2

        #initial pass
        
        for c1 in range(0,4*nk1,4):
            print("c1 ", c1)
            for c2 in range(0,4*nk2,4):
                for c3 in range(0,4*nk3,4):
                    k = K[c1,c2,c3,:]
                    val,vect,p = ham.solve_ham(k,proj=None)

                    d = val[num_occ] - val[num_occ-1]
                    DIRECTGAP[c1,c2,c3] = d
                    
                    #IMAGE[c1,c2,c3,-1] = np.exp( -(d)**2 / sig[-1]**2)


        if thresh == -10:
            thresh = max(np.min(DIRECTGAP) * 2.0, 0.04)
            print("set thresh ", thresh)
         
                    #second pass


        
                    #        maxval = np.max(IMAGE[:,:,:,-1])
                    #        thresh = maxval * 0.1
                    #        print("thresh ", maxval, thresh)
                    

        def fun_to_min(k):
            val,vect,p = ham.solve_ham(k,proj=None)
            return val[num_occ] - val[num_occ-1]
        

        num_thresh = 0

        min_gap = 1000000000.0
        min_cond = 1000000000.0
        max_val = -1000000000.0

        dk1 = float(1)/float(nk1*4)/2.0
        dk2 = float(1)/float(nk2*4)/2.0
        dk3 = float(1)/float(nk3*4)/2.0

        weyl_points = []
        dirac_points = []        
        higher_order_points = []        
        
        
        for c1 in range(0,4*nk1,4):
            print("c1 ", c1)
            for c2 in range(0,4*nk2,4):
                for c3 in range(0,4*nk3,4):
                    for c1a in range(4):
                        for c2a in range(4):
                            for c3a in range(4):

                                if c1a > 0:
                                    c1p=(c1+4)%(4*nk1)
                                else:
                                    c1p=c1
                                if c2a > 0:
                                    c2p=(c2+4)%(4*nk2)
                                else:
                                    c2p=c2
                                if c3a > 0:
                                    c3p=(c3+4)%(4*nk3)
                                else:
                                    c3p=c3
                                    

                                if DIRECTGAP[c1,c2,c3] < thresh or DIRECTGAP[c1,c2,c3p] < thresh or DIRECTGAP[c1,c2p,c3] < thresh or DIRECTGAP[c1,c2p,c3p] < thresh or DIRECTGAP[c1p,c2,c3] < thresh or DIRECTGAP[c1p,c2,c3p] < thresh or DIRECTGAP[c1p,c2p,c3] < thresh or DIRECTGAP[c1p,c2p,c3p] < thresh :
#                                if IMAGE[c1,c2,c3,-1] > thresh or IMAGE[c1,c2,c3p,-1] > thresh or IMAGE[c1,c2p,c3,-1] > thresh or IMAGE[c1,c2p,c3p,-1] > thresh or IMAGE[c1p,c2,c3,-1] > thresh or IMAGE[c1p,c2,c3p,-1] > thresh or IMAGE[c1p,c2p,c3,-1] > thresh or IMAGE[c1p,c2p,c3p,-1] > thresh:
                                    num_thresh += 1
                                    k = K[c1+c1a,c2+c2a,c3+c3a,:]

                                    if use_min:
                                        B = opt.Bounds(k - [dk1, dk2, dk3] , k + [dk1, dk2, dk3], keep_feasible=True)
#                                        print("start ", k,  k - [dk1, dk2, dk3], k + [dk1, dk2, dk3])
                                        val,vect,p = ham.solve_ham(k,proj=None)
                                        if val[num_occ] - val[num_occ-1] < 0.15:
#                                            print(k, " start ", val[num_occ] - val[num_occ-1] )
                                            res = opt.minimize(fun_to_min, k, method="Powell", bounds=B, tol=0.5e-3, options={"maxfev": 5} )
#                                            print(res.x , " prelim ", res.fun)
                                            if res.fun < 0.005:
                                                res = opt.minimize(fun_to_min, res.x, method="Powell", bounds=B, tol=1e-5, options={"maxfev": 30} )
#                                                print(res.x , " good   ", res.fun)
                                            val,vect,p = ham.solve_ham(res.x,proj=None)
                                            
                                            k = res.x
#                                        print("end ", k, val[num_occ] - val[num_occ-1])
                                    else:
                                        val,vect,p = ham.solve_ham(k,proj=None)

                                    d = val[num_occ] - val[num_occ-1]
                                    if d < node_tol:
                                        degen = 0
                                        gap_mean = (val[num_occ] + val[num_occ-1])/2.0
                                        count = np.sum( np.abs(val - gap_mean  ) < node_tol*2)
                                        if count == 2:
                                            weyl_points.append(copy.copy(k))
                                            print("weyl point ", k, d, count)
                                        elif count == 4 or count == 4:
                                            dirac_points.append(copy.copy(k))
                                            print("dirac point ", k, d, count)                                            
                                        else:
                                            higher_order_points.append(copy.copy(k))
                                            print("nontrivial point ", k, d, count)                                                                                        

                                    DIRECTGAP[c1+c1a,c2+c2a,c3+c3a] = d
                                    VAL[c1+c1a,c2+c2a,c3+c3a] = 0.5*(val[num_occ] + val[num_occ-1])
                                    min_gap = min(d, min_gap)
                                    min_cond = min(min_cond, val[num_occ] )
                                    max_val = max(max_val, val[num_occ-1] )
                        

        for (cs,s) in enumerate(sig):
            IMAGE[:,:,:,cs] =  np.exp( -(DIRECTGAP)**2 / s**2)
                                    
        print("num thresh " , num_thresh , " out of ", nk1*nk2*nk3*4*4*4)
        print("min direct gap " , min_gap , " indirect gap  ", min_cond - max_val)

        print("weyl ", weyl_points)
        print("dirac ", dirac_points)
        print("higher order ", higher_order_points)
        
        return IMAGE, DIRECTGAP, VAL, [min_gap, min_cond - max_val], weyl_points, dirac_points, higher_order_points

    
    def dos(self,ham, grid, proj=None, fermi=0.0, xrange=None, nenergy=100, sig = 0.02,  pdf="dos.pdf", show=False, gamma_mode=False):

        plt.clf()

        if gamma_mode:
            kgrid = self.generate_kgrid_gamma(grid)

        else: #normal
            kgrid = self.generate_kgrid(grid)
            
        nk = len(kgrid)
        nwan = ham.nwan
        
        vals = np.zeros((nk,nwan), dtype=float)
        pvals = np.zeros((nk,nwan), dtype=float)

        for i,k in enumerate(kgrid):
            val, vect,p = ham.solve_ham(k, proj)
            vals[i,:] = val - fermi
            pvals[i,:] = p


        # print( vals)
        # print( "pvals")
        # print( pvals)
        print( "np.sum pvals ", np.sum(np.sum(pvals)))
        
        if xrange is not None:
            
            plt.xlim(xrange)
        else:
            
            vmin = np.min(vals[:]) 
            vmax = np.max(vals[:])

            vmin2 = vmin - (vmax-vmin) * 0.05
            vmax2 = vmax + (vmax-vmin) * 0.05
            xrange = [vmin2, vmax2]
            plt.xlim(xrange)

            
        energies = np.arange(xrange[0], xrange[1]+1e-5, (xrange[1]-xrange[0])/float(nenergy))
        dos = np.zeros(np.size(energies))
        pdos = np.zeros(np.size(energies))

        v = vals

        condmin = np.min(v[v > 0.0])
        valmax = np.max(v[v < 0.0])

        print( "DOS BAND GAP " ,  condmin - valmax , "    ", valmax, " " , condmin)

        c = -0.5/sig**2
        for i in range(np.size(energies)):
            arg = c * (v - energies[i])**2
            dos[i] = np.sum(np.exp(arg))
            if not proj is None:
                pdos[i] = np.sum(np.exp(arg) * pvals)
            
        de = energies[1] - energies[0]
        dos = dos / sig / (2.0*np.pi)**0.5  / float(nk)
        if not proj is None:
            pdos = pdos / sig / (2.0*np.pi)**0.5  / float(nk)         
        print( "np.sum(dos) ", np.sum(dos*de))
        if not proj is None:
            print( "np.sum(pdos) ", np.sum(pdos*de))

        plt.plot(energies, dos, "b")
        if not proj is None:
            plt.plot(energies, pdos, "--", color="orange", LineWidth = 3)
            
        plt.ylabel("DOS (eV)^-1")
        plt.xlabel("Energy - $E_F$ (eV)")
            
        plt.tight_layout()
        plt.savefig(pdf)
        if show:
            plt.show()

        
        return energies, dos, pdos

        
            
    def combine_spin_channels(self,hup,hdn, spin_dir=[0,0,1]):
        s_0 = np.array([[1,0],[0,1]])

        s_x = np.array([[0,1],[1,0]]) #pauli
        s_y = np.array([[0,1j],[-1j,0]])
        s_z = np.array([[1,0],[0,-1]])                

        m = spin_dir[0] * s_x + spin_dir[1]*s_y + spin_dir[2]*s_z

        print()
        print( spin_dir)
        print( 'm')
        print( m)
        print()
        
        newh = wan_ham()
        newh.nwan = hup.nwan * 2
        newh.nr = hup.nr

        newh.B = copy.copy(hup.B)

        nw = hup.nwan


        ind_list = self.index_match(hup,hdn)
        Rl = len(ind_list)
        
        newh.R = np.zeros((Rl,3),dtype=float)
        
        newh.HR = np.zeros((Rl, (hup.nwan*2)**2),dtype=np.csingle)

        h = np.zeros((newh.nwan,newh.nwan),dtype=np.csingle)

        newh.HR = np.zeros((Rl, (hup.nwan*2)**2),dtype=np.csingle)
        newh.RR = np.zeros((Rl, (hup.nwan*2)**2,3),dtype=np.csingle)

        h = np.zeros((newh.nwan,newh.nwan),dtype=np.csingle)
        r = np.zeros((newh.nwan,newh.nwan,3),dtype=np.csingle)


        h1 = np.zeros((nw,nw),dtype=np.csingle)
        h2 = np.zeros((nw,nw),dtype=np.csingle)


        temp = np.zeros((2,2),dtype=np.csingle)

        rr1 = np.zeros((nw,nw,3),dtype=np.csingle)
        rr2 = np.zeros((nw,nw,3),dtype=np.csingle)

        temp = np.zeros((2,2),dtype=np.csingle)


        
        #        for i in range(newh.R.shape[0]):
        for i,[iup, idn] in enumerate(ind_list):
            
            h[:,:] = 0.0
            r[:,:,:] = 0.0
            if iup >= 0:
                h1[:,:] = np.reshape(hup.HR[iup,:], (nw,nw))
                newh.R[i,:] = hup.R[iup,:]

                rr1[:,:,0] =  np.reshape(hup.RR[iup,:,0], (nw,nw))
                rr1[:,:,1] =  np.reshape(hup.RR[iup,:,1], (nw,nw))
                rr1[:,:,2] =  np.reshape(hup.RR[iup,:,2], (nw,nw))

            else:
                h1[:,:] = 0.0
                rr1[:,:,:] = 0.0

            if idn >= 0:
                h2[:,:] = np.reshape(hdn.HR[idn,:], (nw,nw))
                newh.R[i,:] = hdn.R[idn,:]                

                rr2[:,:,0] =  np.reshape(hdn.RR[iup,:,0], (nw,nw))
                rr2[:,:,1] =  np.reshape(hdn.RR[iup,:,1], (nw,nw))
                rr2[:,:,2] =  np.reshape(hdn.RR[iup,:,2], (nw,nw))



            else:
                h2[:,:] = 0.0
                rr2[:,:,:] = 0.0
                
#            h[0:nw,0:nw] = h1*m[0,0]
#            h[0:nw,nw:nw*2] = h1*m[0,1]
#            h[0:nw,0:nw] = h1*m[1,0]
#            h[0:nw,0:nw] = h1*m[1,1]
            
            for ii in range(nw):
                for jj in range(nw):
                    a = (h1[ii,jj] + h2[ii,jj])/2.0
                    d = (h1[ii,jj] - h2[ii,jj])/2.0

                    temp = s_0 * a + d * m
                    
                    h[ii*2,jj*2] = temp[0,0]
                    h[ii*2+0,jj*2+1] = temp[0,1]
                    h[ii*2+1,jj*2+0] = temp[1,0]
                    h[ii*2+1,jj*2+1] = temp[1,1]

                    a = (rr1[ii,jj,:] + rr2[ii,jj,:])/2.0
                    d = (rr1[ii,jj,:] - rr2[ii,jj,:])/2.0
                    
                    for q in range(3):
                        rtemp = s_0 * a[q] + d[q] * m

                        r[ii*2,jj*2,q] = rtemp[0,0]
                        r[ii*2+0,jj*2+1,q] = rtemp[0,1]
                        r[ii*2+1,jj*2+0,q] = rtemp[1,0]
                        r[ii*2+1,jj*2+1,q] = rtemp[1,1]
                        



                    
                    
                    
            newh.HR[i,:] = np.reshape(h, (nw*2*nw*2))
            newh.RR[i,:,0] = np.reshape(r[:,:,0], (nw*2*nw*2))
            newh.RR[i,:,1] = np.reshape(r[:,:,1], (nw*2*nw*2))
            newh.RR[i,:,2] = np.reshape(r[:,:,2], (nw*2*nw*2))

            
        return newh    


    def reverse_spin(self, ham):
        h = np.zeros((ham.nwan,ham.nwan),dtype=np.csingle)
        h1 = np.zeros((ham.nwan,ham.nwan),dtype=np.csingle)


        nw = ham.nwan/2

        hnew = copy.copy(ham)
        
        for i in range(ham.R.shape[0]):
            h[:,:] = np.reshape(ham.HR[i,:], (ham.nwan,ham.nwan))
            h1[:,:] = 0.0  

            for ii in range(nw):
                for jj in range(nw):
                    h1[ii*2,jj*2] = h[ii*2+1,jj*2+1]
                    h1[ii*2+1,jj*2+1] = h[ii*2,jj*2]
                    h1[ii*2,jj*2+1] = h[ii*2+1,jj*2]
                    h1[ii*2+1,jj*2] = h[ii*2,jj*2+1]
                    
            hnew.HR[i,:] = np.reshape(h1, (nw*2*nw*2))

        return hnew


    def index_match(self,h1,h2):
    
        r1 = np.array(np.round(h1.R),dtype=int)
        r2 = np.array(np.round(h2.R),dtype=int)        

        def ind(r):
            return r[:,0] * 10**8 + r[:,1]*10**4 + r[:,2]

        indr1=ind(r1)
        indr2=ind(r2)

        indr1_dict = {}
        for c,i in enumerate(indr1):
            indr1_dict[i] = c

        indr2_dict = {}
        for c,i in enumerate(indr2):
            indr2_dict[i] = c
            
#        indr1_dict =  set(indr1)
#        indr2_dict =  set(indr2)

        ind_list = []
        for i in set(indr1.tolist()+indr2.tolist()):
            if i in indr1_dict:
                i1 = indr1_dict[i]
            else:
                i1 = -1

            if i in indr2_dict:
                i2 = indr2_dict[i]
            else:
                i2 = -1
            ind_list.append([i1,i2])
#            print("ind_list ", [i1,i2])
        return ind_list

    
    def add(self,h1,h2, sign=None, fraction=None, sparse=False, percent=1.0):

        if h1.sparse or h2.sparse:
            sparse=True
            
        
        if sign is not None:
            if abs(sign+1.0) < 1e-5:
                fraction = [1.0,-1.0]
            else:
                fraction = [1.0, 1.0]

        if type(fraction) is  float:
            fraction = [1.0-x, x]
            print( 'fraction', fraction)
                
        if fraction is None:
            fraction = [1.0,1.0]

            
        print( 'fraction',fraction            )
        
        newh = wan_ham()
        if sparse:
            newh.sparse=True

        newh.B = copy.copy(h1.B)
            
        newh.nwan = h1.nwan 
        newh.nr = h1.nr

        nw = h1.nwan

        ind_list = self.index_match(h1,h2)
                
                
#        i1 = np.argsort(ind(r1))
#        i2 = np.argsort(ind(r2))        
        

#        print( 'should be zero if everything matches before reorder', np.sum(np.abs(i1-i2)))
#        print( 'should be zero if everything matches after reorder', np.sum(np.sum(np.abs(r1[i1,:]-r2[i2,:]))))
        
        Rl = len(ind_list)
        
        newh.R = np.zeros((Rl,3), dtype=float)
        if sparse:
            newh.HR = sps.lil_matrix((Rl, (newh.nwan)**2),dtype=complex)
            h_temp =  sps.lil_matrix((newh.nwan,newh.nwan),dtype=complex)

            h1_temp = sps.lil_matrix((nw,nw),dtype=complex)
            h2_temp = sps.lil_matrix((nw,nw),dtype=complex)

            rr1_temp = sps.lil_matrix((nw,nw,3),dtype=complex)
            rr2_temp = sps.lil_matrix((nw,nw,3),dtype=complex)

            newh.RR = sps.lil_matrix((Rl, (newh.nwan)**2, 3),dtype=complex)

            
        else:
            newh.HR = np.zeros((Rl, (newh.nwan)**2),dtype=complex)
            newh.RR = np.zeros((Rl, (newh.nwan)**2, 3),dtype=complex)

            h_temp = np.zeros((newh.nwan,newh.nwan),dtype=complex)
            h1_temp = np.zeros((nw,nw),dtype=complex)
            h2_temp = np.zeros((nw,nw),dtype=complex)

            rr1_temp = np.zeros((nw,nw,3),dtype=complex)
            rr2_temp = np.zeros((nw,nw,3),dtype=complex)
            


       
        for i in range(Rl):

            i1,i2=ind_list[i]
            
            #            print( 'r1r2', r1[i1[i],:],r2[i2[i],:])

            if sparse:
                if i1>=0:
                    h1_temp[:,:] = sps.lil_matrix.reshape(h1.HR[i1,:], (nw,nw))
                    
                    rr1_temp[:,:,0] = sps.lil_matrix.reshape(h1.RR[i1,:,0], (nw,nw))
                    rr1_temp[:,:,1] = sps.lil_matrix.reshape(h1.RR[i1,:,1], (nw,nw))
                    rr1_temp[:,:,2] = sps.lil_matrix.reshape(h1.RR[i1,:,2], (nw,nw))
                else:
                    h1_temp[:,:] = sps.lil_matrix( (nw,nw),dtype=complex)
                    rr1_temp[:,:,0] = sps.lil_matrix( (nw,nw),dtype=complex)
                    rr1_temp[:,:,1] = sps.lil_matrix( (nw,nw),dtype=complex)
                    rr1_temp[:,:,2] = sps.lil_matrix( (nw,nw),dtype=complex)


                if i2>=0:
                    h2_temp[:,:] = sps.lil_matrix.reshape(h2.HR[i2,:], (nw,nw))

                    rr2_temp[:,:,0] = sps.lil_matrix.reshape(h2.RR[i2,:,0], (nw,nw))
                    rr2_temp[:,:,1] = sps.lil_matrix.reshape(h2.RR[i2,:,1], (nw,nw))
                    rr2_temp[:,:,2] = sps.lil_matrix.reshape(h2.RR[i2,:,2], (nw,nw))
                else:
                    h2_temp[:,:] = sps.lil_matrix( (nw,nw),dtype=complex)
                    rr2_temp[:,:,0] = sps.lil_matrix( (nw,nw),dtype=complex)
                    rr2_temp[:,:,1] = sps.lil_matrix( (nw,nw),dtype=complex)
                    rr2_temp[:,:,2] = sps.lil_matrix( (nw,nw),dtype=complex)

            else:
                if i1>=0:
                    h1_temp[:,:] = np.reshape(h1.HR[i1,:], (nw,nw))
                    rr1_temp[:,:,0] = np.reshape(h1.RR[i1,:,0], (nw,nw))
                    rr1_temp[:,:,1] = np.reshape(h1.RR[i1,:,1], (nw,nw))
                    rr1_temp[:,:,2] = np.reshape(h1.RR[i1,:,2], (nw,nw))
                else:
                    h1_temp[:,:] = np.zeros( (nw,nw),dtype=complex)
                    rr1_temp[:,:,0] = np.zeros( (nw,nw),dtype=complex)
                    rr1_temp[:,:,1] = np.zeros( (nw,nw),dtype=complex)
                    rr1_temp[:,:,2] = np.zeros( (nw,nw),dtype=complex)

                if i2>=0:
                    h2_temp[:,:] = np.reshape(h2.HR[i2,:], (nw,nw))
                    rr2_temp[:,:,0] = np.reshape(h2.RR[i2,:,0], (nw,nw))
                    rr2_temp[:,:,1] = np.reshape(h2.RR[i2,:,1], (nw,nw))
                    rr2_temp[:,:,2] = np.reshape(h2.RR[i2,:,2], (nw,nw))
                else:
                    h2_temp[:,:] = np.zeros( (nw,nw),dtype=complex)
                    rr2_temp[:,:,0] = np.zeros( (nw,nw),dtype=complex)
                    rr2_temp[:,:,1] = np.zeros( (nw,nw),dtype=complex)
                    rr2_temp[:,:,2] = np.zeros( (nw,nw),dtype=complex)

                
            h_temp[:,:] = h1_temp*fraction[0] + fraction[1]*h2_temp*percent
            r_temp = rr1_temp*fraction[0] + fraction[1]*rr2_temp*percent

#            if i == 0 or i == 1:
#                print("xxx")
#                print(i2, i2)
#                print("h1t")
#                print(h1_temp)
#                print("h2t")
#                print(h2_temp)
#                print("ht")
#                print(h_temp)
#                print(h1.R[i1,:] , " c " ,h1.R[i1,:])
#            for ii in range(nw):
#                for jj in range(nw):
#                    h[ii*2,jj*2] = h1[ii,jj]
#                    h[ii*2+1,jj*2+1] = h2[ii,jj]

            if sparse:
                newh.HR[i,:] = sps.lil_matrix.reshape(h_temp, (nw*nw))
                newh.RR[i,:,0] = sps.lil_matrix.reshape(r_temp[:,:,0], (nw*nw))
                newh.RR[i,:,1] = sps.lil_matrix.reshape(r_temp[:,:,1], (nw*nw))
                newh.RR[i,:,2] = sps.lil_matrix.reshape(r_temp[:,:,2], (nw*nw))
            else:
                newh.HR[i,:] = np.reshape(h_temp, (nw*nw))
                newh.RR[i,:,0] = np.reshape(r_temp[:,:,0], (nw*nw))
                newh.RR[i,:,1] = np.reshape(r_temp[:,:,1], (nw*nw))
                newh.RR[i,:,2] = np.reshape(r_temp[:,:,2], (nw*nw))

                
            if i1 >= 0:
                newh.R[i,:] = h1.R[i1,:]
            else:
                newh.R[i,:] = h2.R[i2,:]                
        return newh    
    
    def add_middle(self,h1,h2, sign=+1.0):

        newh = wan_ham()
        newh.nwan = h1.nwan 
        newh.nr = h1.nr

        nw = h1.nwan

        r1 = np.array(np.round(h1.R),dtype=int)
        r2 = np.array(np.round(h2.R),dtype=int)        

        def ind(r):
            return r[:,0] * 10**6 + r[:,1]*10**3 + r[:,2]

        i1 = np.argsort(ind(r1))
        i2 = np.argsort(ind(r2))        
        

        print( 'should be zero if everything matches before reorder', np.sum(np.abs(i1-i2)))
        print( 'should be zero if everything matches after reorder', np.sum(np.sum(np.abs(r1[i1,:]-r2[i2,:]))))
        

        
        newh.R = np.zeros(h1.R.shape, dtype=float)
        newh.HR = np.zeros((newh.R.shape[0], (newh.nwan)**2),dtype=np.csingle)

        h_temp = np.zeros((newh.nwan,newh.nwan),dtype=np.csingle)

       
        h1_temp = np.zeros((nw,nw),dtype=np.csingle)
        h2_temp = np.zeros((nw,nw),dtype=np.csingle)
        for i in range(newh.R.shape[0]):

#            print( 'r1r2', r1[i1[i],:],r2[i2[i],:])
            
            h1_temp[:,:] = np.reshape(h1.HR[i1[i],:], (nw,nw))
            h2_temp[:,:] = np.reshape(h2.HR[i2[i],:], (nw,nw))

            h_temp[:,:] = h1_temp + sign*h2_temp
            
            
#            for ii in range(nw):
#                for jj in range(nw):
#                    h[ii*2,jj*2] = h1[ii,jj]
#                    h[ii*2+1,jj*2+1] = h2[ii,jj]
                    
            newh.HR[i,:] = np.reshape(h_temp, (nw*nw))
            newh.R[i,:] = h1.R[i1[i],:]

        return newh    


    def add_simple(self,h1,h2, sign=+1.0):

        newh = wan_ham()
        newh.nwan = h1.nwan 
        newh.nr = h1.nr

        nw = h1.nwan
        
        newh.R = copy.copy(h1.R)
        newh.HR = np.zeros((newh.R.shape[0], (newh.nwan)**2),dtype=np.csingle)
        newh.RR = np.zeros((newh.RR.shape[0], (newh.nwan)**2, 3),dtype=np.csingle)

        h_temp = np.zeros((newh.nwan,newh.nwan),dtype=np.csingle)

        h1_temp = np.zeros((nw,nw),dtype=np.csingle)
        h2_temp = np.zeros((nw,nw),dtype=np.csingle)
        for i in range(newh.R.shape[0]):
            h1_temp[:,:] = np.reshape(h1.HR[i,:], (nw,nw))
            h2_temp[:,:] = np.reshape(h2.HR[i,:], (nw,nw))

            h_temp[:,:] = h1_temp + sign*h2_temp
                    
            newh.HR[i,:] = np.reshape(h_temp, (nw*nw))
            newh.RR[i,:,:]  = h1.RR[i,:,:] + sign*h2.RR[i,:,:]


        return newh    
    
    def generate_supercell(self,h, supercell, cut=[0,0,0], sparse=False):

        t0=time.time()
        
        nw = h.nwan
        
        factor = np.prod(supercell)
        NWAN = factor * h.nwan


        
        def plus_r(rold, subcell):
            rnew = subcell + rold
            cellnew = rnew // supercell   #this is integer division
            subnew  = rnew%supercell

            return cellnew,subnew
    
        def subcell_index(ss):
            t = ss[0]*supercell[1]*supercell[2] + ss[1]*supercell[2] + ss[2]
            return range(t*nw,(t+1)*nw)
        

        RH_new = {}
        h_temp = np.zeros((h.nwan,h.nwan),dtype=np.csingle)
        rr_temp = np.zeros((h.nwan,h.nwan,3),dtype=np.csingle)
        subcell = np.zeros(3,dtype=int)

        t1=time.time()

        for ii in range(h.R.shape[0]):

            rold  = np.array(h.R[ii,:], dtype=int)
            
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    for k in range(supercell[2]):
                        subcell[:] = [i,j,k]

                        cellnew, subnew = plus_r(rold,subcell)

                        if (cut[0] > 0 and cellnew[0] != 0) or (cut[1] > 0 and cellnew[1] != 0) or (cut[2] > 0 and cellnew[2] != 0):
                            continue
                        
#                        print( 'rs', rold, subcell, 'new', cellnew, subnew)
                        if tuple(cellnew) not in RH_new:
                            if sparse:
                                RH_new[tuple(cellnew)] = [cellnew, sps.lil_matrix((NWAN,NWAN ),dtype=np.csingle), sps.lil_matrix((NWAN,NWAN,3 ),dtype=np.csingle)]
                            else:
                                RH_new[tuple(cellnew)] = [cellnew, np.zeros((NWAN,NWAN ),dtype=np.csingle), np.zeros((NWAN,NWAN,3 ),dtype=np.csingle)]


                        h_temp[:,:] = np.reshape(h.HR[ii,:], (h.nwan,h.nwan))

                        rr_temp[:,:,0] = np.reshape(h.RR[ii,:,0], (h.nwan,h.nwan))
                        rr_temp[:,:,1] = np.reshape(h.RR[ii,:,1], (h.nwan,h.nwan))
                        rr_temp[:,:,2] = np.reshape(h.RR[ii,:,2], (h.nwan,h.nwan))

                        r1 = subcell_index(subcell)
                        r2 = subcell_index(subnew)

#                        print( r1,r2,h_temp.shape, RH_new[tuple(cellnew)][1][r1,r2].shape)
                        for c1,c2 in enumerate(r1):
                            for d1,d2 in enumerate(r2):
                                RH_new[tuple(cellnew)][1][c2,d2] += h_temp[c1,d1]
                                RH_new[tuple(cellnew)][2][c2,d2,:] += rr_temp[c1,d1,:]                                


        for key in RH_new:
            print("key ", key)
        print(len(RH_new))
        
        t2=time.time()
                                
        rn = len(RH_new)
        hbig = wan_ham()
        hbig.B[0,:] = h.B[0,:] / supercell[0]
        hbig.B[1,:] = h.B[1,:] / supercell[1]
        hbig.B[2,:] = h.B[2,:] / supercell[2]        
        if sparse:
            hbig.sparse = True
            
        hbig.nwan = NWAN
        hbig.nr = rn
        

        hbig.R = np.zeros((rn, 3),dtype=float)
        if sparse:
            hbig.HR = sps.lil_matrix((rn, NWAN**2),dtype=np.csingle)
            hbig.RR = sps.lil_matrix((rn, NWAN**2,3),dtype=np.csingle)
        else:
            hbig.HR = np.zeros((rn, NWAN**2),dtype=np.csingle)
            hbig.RR = np.zeros((rn, NWAN**2,3),dtype=np.csingle)            
            
        for c,i in enumerate(RH_new):
            h = RH_new[i][1]
            r = RH_new[i][0]
            rr = RH_new[i][2]

            if sparse:
                hbig.HR[c,:] = sps.lil_matrix.reshape(h, NWAN*NWAN)

                hbig.RR[c,:,0] = sps.lil_matrix.reshape(rr[:,:,0], NWAN*NWAN)
                hbig.RR[c,:,1] = sps.lil_matrix.reshape(rr[:,:,1], NWAN*NWAN)
                hbig.RR[c,:,2] = sps.lil_matrix.reshape(rr[:,:,2], NWAN*NWAN)

            else:
                hbig.HR[c,:] = np.reshape(h, NWAN*NWAN)

#                print("size h ", np.shape(h), "  rr ", np.shape(rr))
                
                hbig.RR[c,:,0] = np.reshape(rr[:,:,0], NWAN*NWAN)
                hbig.RR[c,:,1] = np.reshape(rr[:,:,1], NWAN*NWAN)
                hbig.RR[c,:,2] = np.reshape(rr[:,:,2], NWAN*NWAN)                
                
            hbig.R[c,:] = r


        t3=time.time()

        print( 'TIME SUPERCELL', t1-t0, t2-t1, t3-t2)
        return hbig






    def generate_supercell_magnetic(self,hup, hdn, supercell, mag, cut=[0,0,0], sparse=False):

        c = 0
        for i in range(supercell[0]):
            for j in range(supercell[1]):
                for k in range(supercell[2]):
                    print( 'ijk', i,j,k, 'magnetic', mag[c])
                    c += 1
                    
        nw = hup.nwan
        
        factor = np.prod(supercell)
        NWAN = factor * hup.nwan * 2

        s_0 = np.array([[1,0],[0,1]])

        s_x = np.array([[0,1],[1,0]]) #pauli
        s_y = np.array([[0,1j],[-1j,0]])
        s_z = np.array([[1,0],[0,-1]])                

        
        def plus_r(rold, subcell):
            rnew = subcell + rold
            cellnew = rnew // supercell   #this is integer division
            subnew  = rnew%supercell

            return cellnew,subnew
    
        def subcell_index(ss):
            t = ss[0]*supercell[1]*supercell[2] + ss[1]*supercell[2] + ss[2]
            return range(t*nw*2,(t+1)*nw*2)

        
        def subcell_index_mag(ss):
            t = ss[0]*supercell[1]*supercell[2] + ss[1]*supercell[2] + ss[2]
            return t
        

        RH_new = {}

#        h_temp = np.zeros((NWAN, NWAN),dtype=np.csingle)
        h_temp = np.zeros((hup.nwan*2, hup.nwan*2),dtype=np.csingle)

        rr_temp = np.zeros((hup.nwan*2, hup.nwan*2,3),dtype=np.csingle)
        
        h_temp_up = np.zeros((hup.nwan,hup.nwan),dtype=np.csingle)
        h_temp_dn = np.zeros((hup.nwan,hup.nwan),dtype=np.csingle)

        rr_temp_up = np.zeros((hup.nwan,hup.nwan,3),dtype=np.csingle)
        rr_temp_dn = np.zeros((hup.nwan,hup.nwan,3),dtype=np.csingle)

        
        ind_list = self.index_match(hup,hdn)
        
            
        for i in range(supercell[0]):
            for j in range(supercell[1]):
                for k in range(supercell[2]):
                    subcell = np.array([i,j,k],dtype=int)
                    print("subcell ", subcell)
                    #                    for ii in range(hup.R.shape[0]):
                    for [iup,idn] in ind_list:

                        if iup > 0:
                            rold  = np.array(hup.R[iup,:], dtype=int)
                        else:
                            rold  = np.array(hdn.R[idn,:], dtype=int)
                            
                        cellnew, subnew = plus_r(rold,subcell)

                        if (cut[0] > 0 and cellnew[0] != 0) or (cut[1] > 0 and cellnew[1] != 0) or (cut[2] > 0 and cellnew[2] != 0):
                            continue
                        
#                        print( 'rs', rold, subcell, 'new', cellnew, subnew)
                        if tuple(cellnew) not in RH_new:
                            if not sparse:
#                                RH_new[tuple(cellnew)] = [cellnew, np.zeros((NWAN,NWAN ),dtype=np.csingle)]
                                RH_new[tuple(cellnew)] = [cellnew, np.zeros((NWAN,NWAN ),dtype=np.csingle), np.zeros((NWAN,NWAN,3 ),dtype=np.csingle)]
                            elif sparse:
                                RH_new[tuple(cellnew)] = [cellnew, sps.lil_matrix((NWAN,NWAN ),dtype=np.csingle), sps.lil_matrix((NWAN,NWAN,3 ),dtype=np.csingle)]


                            #                        h_temp[:,:] = np.reshape(h.HR[ii,:], (h.nwan,h.nwan))

#                        print("ss ", [subcell, subnew])
                            
                        spin_dirA = mag[subcell_index_mag(subcell)]
                        spin_dirB = mag[subcell_index_mag(subnew)]


                        mA = spin_dirA[0] * s_x + spin_dirA[1]*s_y + spin_dirA[2]*s_z
                        mB = spin_dirB[0] * s_x + spin_dirB[1]*s_y + spin_dirB[2]*s_z

                        if iup >= 0:
                            h_temp_up[:,:] = np.reshape(hup.HR[iup,:], (hup.nwan,hup.nwan))
                            rr_temp_up[:,:,0] = np.reshape(hup.RR[iup,:,0], (hup.nwan,hup.nwan))
                            rr_temp_up[:,:,1] = np.reshape(hup.RR[iup,:,1], (hup.nwan,hup.nwan))
                            rr_temp_up[:,:,2] = np.reshape(hup.RR[iup,:,2], (hup.nwan,hup.nwan))
                        else:
                            h_temp_up[:,:] = np.zeros((hup.nwan,hup.nwan),dtype=np.csingle)
                            rr_temp_up[:,:,:] = np.zeros((hup.nwan,hup.nwan,3),dtype=np.csingle)
                        if idn >= 0:
                            h_temp_dn[:,:] = np.reshape(hdn.HR[idn,:], (hup.nwan,hup.nwan))
                            rr_temp_dn[:,:,0] = np.reshape(hdn.RR[idn,:,0], (hup.nwan,hup.nwan))
                            rr_temp_dn[:,:,1] = np.reshape(hdn.RR[idn,:,1], (hup.nwan,hup.nwan))
                            rr_temp_dn[:,:,2] = np.reshape(hdn.RR[idn,:,2], (hup.nwan,hup.nwan))
                        else:
                            h_temp_dn[:,:] = np.zeros((hup.nwan,hup.nwan),dtype=np.csingle)
                            rr_temp_dn[:,:,:] = np.zeros((hup.nwan,hup.nwan,3),dtype=np.csingle)
                            
                        h_temp[:,:] = 0.0
                        rr_temp[:,:,:] = 0.0
                        for m in [mA,mB]:
                            for c1 in range(nw):
                                for c2 in range(nw):
                                    a = (h_temp_up[c1,c2] + h_temp_dn[c1,c2])/2.0
                                    d = (h_temp_up[c1,c2] - h_temp_dn[c1,c2])/2.0
                                
                                    temp = s_0 * a + d * m

                                    ar = (rr_temp_up[c1,c2,:] + rr_temp_dn[c1,c2,:])/2.0
                                    dr = (rr_temp_up[c1,c2,:] - rr_temp_dn[c1,c2,:])/2.0
                                    rtemp = np.zeros((2,2,3),dtype=np.csingle)
                                    for jj in range(3):
                                        rtemp[:,:,jj] = s_0 * ar[jj] + dr[jj] * m
                                        
                                        
                                    h_temp[c1*2,c2*2] += temp[0,0]
                                    h_temp[c1*2+0,c2*2+1] += temp[0,1]
                                    h_temp[c1*2+1,c2*2+0] += temp[1,0]
                                    h_temp[c1*2+1,c2*2+1] += temp[1,1]

                                    rr_temp[c1*2,c2*2    ,:]   += rtemp[0,0,:]
                                    rr_temp[c1*2+0,c2*2+1,:] += rtemp[0,1,:]
                                    rr_temp[c1*2+1,c2*2+0,:] += rtemp[1,0,:]
                                    rr_temp[c1*2+1,c2*2+1,:] += rtemp[1,1,:]
                                    

                                    
                        h_temp = h_temp / 2.0
                        rr_temp = rr_temp / 2.0
                        

                        r1 = subcell_index(subcell)
                        r2 = subcell_index(subnew)

#                        print( r1,r2,h_temp.shape, RH_new[tuple(cellnew)][1][r1,r2].shape)
                        for c1,c2 in enumerate(r1):
                            for d1,d2 in enumerate(r2):
                                RH_new[tuple(cellnew)][1][c2,d2] += h_temp[c1,d1]
                                RH_new[tuple(cellnew)][2][c2,d2,:] += rr_temp[c1,d1,:]                                

        rn = len(RH_new)
        hbig = wan_ham()
        hbig.B[0,:] = hup.B[0,:] / supercell[0]
        hbig.B[1,:] = hup.B[1,:] / supercell[1]
        hbig.B[2,:] = hup.B[2,:] / supercell[2]        
        
        if sparse:
            hbig.sparse = True
        
        hbig.nwan = NWAN
        hbig.nr = rn

        hbig.R = np.zeros((rn, 3),dtype=float)
        if sparse:
            hbig.HR = sps.lil_matrix((rn, NWAN**2),dtype=np.csingle)
            hbig.RR = sps.lil_matrix((rn, NWAN**2,3),dtype=np.csingle)            
        else:            
            hbig.HR = np.zeros((rn, NWAN**2),dtype=np.csingle)
            hbig.RR = np.zeros((rn, NWAN**2,3),dtype=np.csingle)           

        for c,i in enumerate(RH_new):
            h = RH_new[i][1]
            rr = RH_new[i][2]
            r = RH_new[i][0]
#            print("shape rr ", np.shape(rr))
            if sparse:
                hbig.HR[c,:] = sps.lil_matrix.reshape(h, NWAN*NWAN)
                hbig.RR[c,:,0] = sps.lil_matrix.reshape(rr[:,:,0], NWAN*NWAN)
                hbig.RR[c,:,1] = sps.lil_matrix.reshape(rr[:,:,1], NWAN*NWAN)
                hbig.RR[c,:,2] = sps.lil_matrix.reshape(rr[:,:,2], NWAN*NWAN)                                
            else:
                hbig.HR[c,:] = np.reshape(h, NWAN*NWAN)

                hbig.RR[c,:,0] = np.reshape(rr[:,:,0], NWAN*NWAN)
                hbig.RR[c,:,1] = np.reshape(rr[:,:,1], NWAN*NWAN)
                hbig.RR[c,:,2] = np.reshape(rr[:,:,2], NWAN*NWAN)
                
            hbig.R[c,:] = r

        return hbig

##############################################################################
##############################################################################

    def plot_eigenvector(self, ham, nocc, pos_orig, A, supercell, view,atom_dict = None,colors=None, kpoint=[0,0,0], vect=None, pdfname='vect.pdf', fermi=fermi, sparse=False):

        print( 'supercell plot_eigenvector', supercell)
        
        nwan_small = int(ham.nwan / np.prod(supercell))
        pos_orig = np.array(pos_orig)

        plt.clf()
        
        if colors is None:
            colors = ['b']*nwan_small

        if atom_dict is None:
            atom_dict = {}
            for i in range(nwan_small):
                atom_dict[i] = i

        if vect is None:

            if sparse:
                print( 'using sparse')
                if type(nocc) == int or type(nocc) == np.int64:
                    val, vect, _ = ham.solve_ham_sparse(kpoint, nocc, fermi=fermi, proj=None)
                else:
                    val, vect, _ = ham.solve_ham_sparse(kpoint, len(nocc), fermi=fermi, proj=None)
                    
                print( 'val sparse', val)
                
            else:
                val, vect, _ = ham.solve_ham(kpoint)

                print( 'val.shape', val.shape)
                print( 'vall occ')
                print( val[nocc])
                print( 'vect.shape', vect.shape)

        if sparse:
            v2 = np.sum((vect[:,:]*vect[:,:].conj()).real, 1)
        else:
            if type(nocc) == int or  type(nocc) == np.int64:
                v2 = (vect[:,nocc]*vect[:,nocc].conj()).real

            else:
                v2 = np.sum((vect[:,nocc]*vect[:,nocc].conj()).real, 1)
            
        
        print( 'v2.shape', v2.shape)

#        nat = len(atom_dict)
        nat = pos_orig.shape[0]
        
        print( 'nat', nat)
        
        xyzV = np.zeros((nat*np.prod(supercell), 4),dtype=float)

        unique_colors = list(set(colors))

        print( 'unique_colors', unique_colors)
        COLORS = []
        for i in range(len(unique_colors)):
            COLORS.append([])
        c=0

        for x in range(supercell[0]):
            for y in range(supercell[1]):
                for z in range(supercell[2]):
                    ssnum = (x*supercell[1]*supercell[2]+y*supercell[2]+z)*nat
                    for n in range(nwan_small):
                        
                        at = atom_dict[n]
#                        print( 'ssnum at', ssnum, at, [x,y,z], n, 'ind', ssnum+at)
                        xyzV[ssnum+at,0:3] = np.dot((pos_orig[at,:]+ np.array([x,y,z])), A)
                        xyzV[ssnum+at,3] += v2[c]

                        for cnum,u in enumerate(unique_colors):
                            if colors[at] == u:
                                COLORS[cnum].append(ssnum+at)

                        c+=1


        print( 'xyzV')
        for i in range(xyzV.shape[0]):
            print( xyzV[i,:])
        print()

        vsize_scaled = xyzV[:,3] / np.max(xyzV[:,3]) * 30.0
        
        for cnum,u in enumerate(unique_colors):
            x = xyzV[COLORS[cnum], view[0]]
            y = xyzV[COLORS[cnum], view[1]]
            vsize = vsize_scaled[COLORS[cnum]]
            plt.scatter(x, y, vsize, u)

        x = xyzV[:, view[0]]
        y = xyzV[:, view[1]]
        plt.scatter(x,y,0.05, 'k')

        max_d = max(np.max(y), np.max(x))

        min_d = min([0,np.min(y), np.min(x)])

        delta = (max_d-min_d)*0.03
        
        plt.xlim([min_d-delta , max_d + delta])
        plt.ylim([min_d-delta , max_d + delta])
        
        
        plt.xlabel('Position 1 (crystal)')
        plt.ylabel('Position 2 (crystal)')

        plt.tight_layout()
        plt.savefig(pdfname)
#        plt.show()
        
    #                        
#                        
#
#                        #        Rnew = np.zeros(h.R.shape)
#        #        Hnew = np.zeros(h.R.shape[0])
#
#        conversion = {}
#        rnew_set = set()
#        rnew_index = {}
#        rn = 0
#        for ii in range(h.R.shape[0]):
#
#            r = np.zeros(3,dtype=float)
#            subcell = np.zeros(3,dtype=int)
#            rnew = np.zeros(3,dtype=float)
#            r[:] = h.R[ii,:]
#            for i in range(3):
#                rnew[i] = np.floor(r[i]/supercell[i])
#                subcell[i] = r[i]%supercell[i]
#
#            conversion[ii] = [rnew,subcell]
#            print( ii, 'r', r, 'rnew', rnew,subcell)
#            rnew_set.add(tuple(rnew.tolist()))
#            if tuple(rnew.tolist()) not in rnew_index:
#                rnew_index[tuple(rnew.tolist())] = rn
#                rn += 1
#
#            
#        hbig.R = np.zeros((rn, 3),dtype=float)
#        hbig.HR = np.zeros((rn, NWAN**2),dtype=np.csingle)
#
#        HR_temp = np.zeros((rn, NWAN,NWAN ),dtype=np.csingle)
#
#        h_temp = np.zeros((h.nwan,h.nwan),dtype=np.csingle)
#        
#        for ii in range(h.R.shape[0]):
#            rnew, subcell = conversion[ii]
#            ind = rnew_index[tuple(rnew.tolist())]
#
#
#            hbig.R[ind,:] = rnew[:]
#
#            h_temp[:,:] = np.reshape(h.HR[ii,:], (h.nwan,h.nwan))
#            
#            for i in range(supercell[0]):
#                for j in range(supercell[1]):
#                    for k in range(supercell[2]):
#
#                        block1 = i*supercell[1]*supercell[2] + j*supercell[2] + k
#                        t = np.array([i,j,k],dtype=int) + subcell
#
#                        t[0] = t[0]%supercell[0]
#                        t[1] = t[1]%supercell[1]
#                        t[2] = t[2]%supercell[2]
#                        block2 = t[0]*supercell[1]*supercell[2] + t[1]*supercell[2] + t[2]
#
#
#                        print( 'asdf', ii, rnew, subcell,'b1',[i,j,k], block1, 't', t, block2)
#                        
#                        wan1 = np.array(range(0,h.nwan),dtype=int)+block1*h.nwan
#                        wan2 = np.array(range(0,h.nwan),dtype=int)+block2*h.nwan                        
#
#                        #                        print( 'w1', wan1)
#                        #                        print( 'w2', wan2)
#                        for c1,w1 in enumerate(wan1):
#                            for c2,w2 in enumerate(wan2):
#                                HR_temp[ind,w1,w2] = h_temp[c1,c2]
#                        
#                        
#                                                      
#        for i in range(HR_temp.shape[0]):
#            hbig.HR[i,:] = np.reshape(HR_temp[i,:,:], NWAN*NWAN)
#
#            print( hbig.R[i,:])
#            print( HR_temp[i,:,:])
#            print()
#            
#
#        return  hbig



    def generate_supercell_sparse(self,h, supercell, cut=[0,0,0]):

        t0=time.time()
        
        nw = h.nwan
        
        factor = np.prod(supercell)
        NWAN = factor * h.nwan


        
        def plus_r(rold, subcell):
            rnew = subcell + rold
            cellnew = rnew/ supercell   #this is integer division
            subnew  = rnew%supercell

            return cellnew,subnew
    
        def subcell_index(ss):
            t = ss[0]*supercell[1]*supercell[2] + ss[1]*supercell[2] + ss[2]
            return range(t*nw,(t+1)*nw)
        

        RH_new = {}
        h_temp = np.zeros((h.nwan,h.nwan),dtype=np.csingle)
#        h_temp = sps.csc_matrix((h.nwan,h.nwan),dtype=np.csingle)

        
        subcell = np.zeros(3,dtype=int)

        t1=time.time()

        a=0.0
        b=0.0
        ct=0.0
        d=0.0
        e=0.0
        for ii in range(h.R.shape[0]):

            rold  = np.array(h.R[ii,:], dtype=int)
            
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    for k in range(supercell[2]):

                        t1a=time.time()

                        subcell[:] = [i,j,k]

                        cellnew, subnew = plus_r(rold,subcell)

                        if (cut[0] > 0 and cellnew[0] != 0) or (cut[1] > 0 and cellnew[1] != 0) or (cut[2] > 0 and cellnew[2] != 0):
                            continue

                        t1b=time.time()
                        
#                        print( 'rs', rold, subcell, 'new', cellnew, subnew)
                        if tuple(cellnew) not in RH_new:
#                            RH_new[tuple(cellnew)] = [cellnew, sps.csc_matrix((NWAN,NWAN ),dtype=np.csingle)]
                            RH_new[tuple(cellnew)] = [cellnew, sps.lil_matrix((NWAN,NWAN ),dtype=np.csingle)]

                        t1c=time.time()

#                        h_temp[:,:] = sps.csc_matrix.reshape(sps.csc_matrix(h.HR[ii,:]), (h.nwan,h.nwan))
                        h_temp[:,:] = np.reshape(h.HR[ii,:], (h.nwan,h.nwan))

                        t1d=time.time()
                        
                        r1 = subcell_index(subcell)
                        r2 = subcell_index(subnew)

#                        print( r1,r2,h_temp.shape, RH_new[tuple(cellnew)][1][r1,r2].shape)
                        for c1,c2 in enumerate(r1):
                            for d1,d2 in enumerate(r2):
                                RH_new[tuple(cellnew)][1][c2,d2] += h_temp[c1,d1]

                        t1e=time.time()
                        a += t1b-t1a
                        b += t1c-t1b
                        ct += t1d-t1c
                        d += t1e-t1d                        
                                
        t2=time.time()
                                
        rn = len(RH_new)
        hbig = wan_ham()
        hbig.sparse = True
        
        hbig.nwan = NWAN
        hbig.nr = rn
        

        hbig.R = np.zeros((rn, 3),dtype=float)
#        hbig.HR = np.zeros((rn, NWAN**2),dtype=np.csingle)
        hbig.HR = sps.lil_matrix((rn, NWAN**2),dtype=np.csingle)
        
        for c,i in enumerate(RH_new):
            h = RH_new[i][1]
            r = RH_new[i][0]
            hbig.HR[c,:] = sps.lil_matrix.reshape(h, NWAN*NWAN)
            hbig.R[c,:] = r


        t3=time.time()
        print( 'TIME SUPERCELL INNER', a, b, ct, d)
        print( 'TIME SUPERCELL', t1-t0, t2-t1, t3-t2)
        return hbig

    def get_orbitals(self,projection_info, desired_orbitals, so=False, NCELLS=1, surfaceonly=False):
        #projection_info example for Bi2Se3 with s and p orbital projections
        #[["Bi", 2, ["s","p"]], ["Se", 3, ["s","p"]]]
        #NCELLS for supercells
        #surface only assumes 1x1 surface

        #orbitals wanted example
        #[["Bi"]]   all Bi orbitals

        #[["Bi", "p"]] all Bi p orbitals

        #[["Bi", "px"], ["Bi" ,"py"]] all Bi px, py oribitals

        #[["Bi", "s"], ["Se", "s"]]  Bi s and Se s

        #so for spin-orbit

        c = 0

        projection_dict = {}

        print( "get_orbs ", so)
        print( projection_info)
                
        for proj in projection_info:

            atom = proj[0]
            natom = proj[1]
            orbitals = proj[2]

            for n in range(natom):
                for o in orbitals:
                    if o == "s":
                        if (atom, "s") not in projection_dict:
                            projection_dict[(atom, "s")] = []
                        projection_dict[(atom, "s")].append(c)
                        c += 1
                        
                    elif o == "p":
                        if (atom, "p") not in projection_dict:
                            projection_dict[(atom, "p")] = []
                            projection_dict[(atom, "pz")] = []
                            projection_dict[(atom, "py")] = []
                            projection_dict[(atom, "px")] = []

                        projection_dict[(atom, "p")].append(c)
                        projection_dict[(atom, "pz")].append(c)
                        c += 1

                        projection_dict[(atom, "p")].append(c)
                        projection_dict[(atom, "px")].append(c)
                        c += 1

                        projection_dict[(atom, "p")].append(c)
                        projection_dict[(atom, "py")].append(c)
                        c += 1

                    elif o == "d":
                        if (atom, "d") not in projection_dict:
                            projection_dict[(atom, "d")] = []
                            projection_dict[(atom, "dz2")] = []
                            projection_dict[(atom, "dxz")] = []
                            projection_dict[(atom, "dyz")] = []
                            projection_dict[(atom, "dx2y2")] = []
                            projection_dict[(atom, "dxy")] = []

                        projection_dict[(atom, "d")].append(c)
                        projection_dict[(atom, "dz2")].append(c)
                        c += 1

                        projection_dict[(atom, "d")].append(c)
                        projection_dict[(atom, "dxz")].append(c)
                        c += 1

                        projection_dict[(atom, "d")].append(c)
                        projection_dict[(atom, "dyz")].append(c)
                        c += 1

                        projection_dict[(atom, "d")].append(c)
                        projection_dict[(atom, "dx2y2")].append(c)
                        c += 1

                        projection_dict[(atom, "d")].append(c)
                        projection_dict[(atom, "dxy")].append(c)
                        c += 1

        nwan = c

#        if so:
#            for (atom, orb) in projection_dict.keys():
#                new_ind = []
#                for i in projection_dict[(atom, orb)]:
#                    new_ind.append(i+nwan)
#                projection_dict[(atom, orb)] += new_ind
#            nwan = nwan * 2

        if so:
            for (atom, orb) in projection_dict.keys():
                new_ind = []
                for i in projection_dict[(atom, orb)]:
                    new_ind.append(i * 2)
                    new_ind.append(i*2+1)

                    #                    new_ind.append(i+nwan)
                projection_dict[(atom, orb)] = new_ind
            nwan = nwan * 2

        print( "nwan = ", nwan)
        
        inds = []
        for d in desired_orbitals:
            if len(d) == 1:
                for orb in ["s", "p", "d"]:
                    if (d[0], orb) in projection_dict:
                        new_orbs = projection_dict[(d[0], orb)]
                        inds += new_orbs
            else:
                new_orbs = projection_dict[tuple(d)]
                inds += new_orbs

        if NCELLS > 1 and surfaceonly == False:

            inds_super = []
            for n in range(NCELLS):
                for i in inds:
                    inds_super.append(i+ n * nwan)

            return inds_super

        elif NCELLS > 1 and surfaceonly == True:
            inds_super = []
            for n in [0, NCELLS-1]:
                for i in inds:
                    inds_super.append(i+ n * nwan)

            return inds_super

            
        else:
            return inds
                
            
