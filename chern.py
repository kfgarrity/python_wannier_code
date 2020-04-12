    def chern_number_simple(self,ham,nocc, k1,k2, nk1, nk2, Kmat = None, usemod=True):

        if Kmat is None:
        
            K = np.zeros((nk1,nk2, 3),dtype=float)

            k1=np.array(k1,dtype=float)
            k2=np.array(k2,dtype=float)

            #        print 'K'

            for c1 in range(nk1):
                for c2 in range(nk2):
                    K[c1,c2,:] = k1 * float(c1)/float(nk1) + k2 * float(c2)/float(nk2)

        else:
            K = Kmat
            
                    
                #                print c1, c2, 'K', K[c1,c2,:]

                #        print

        if usemod:

            lim1=nk1
            lim2=nk2

        else:

            lim1=nk1+1
            lim2=nk2+1
            
        VECT = np.zeros((lim1, lim2, ham.nwan, ham.nwan),dtype=complex)
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
                #                print 'val'
                #                print val
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
                   
                   
        print 'minimum_direct_gap', gap_min, 'indirect_gap', cond_min-val_max
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
                    print 'WARNING in chern calculation, phi', c1, c2, phi, ' try more k-points'
                    
                Chern += phi / (2.0*np.pi)

        if usemod:
            return Chern, direct_gap, indirect_gap
        else:
            return Chern, direct_gap, val_max,cond_min
