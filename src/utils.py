"""
#######################
#        quket        #
#######################

utils.py

Utilities.

"""


import sys
import time
import numpy as np
from . import config as cf
from . import mpilib as mpi
import math
import pprint
import itertools
from openfermion.transforms import jordan_wigner
from openfermion.ops import QubitOperator
from qulacs import QuantumState

import copy

def cost_mpi(cost,theta):
    '''
       Simply run the given cost function with varaibles theta,
       but ensure that all MPI processes contain the same cost.
       This should help eliminate the possible deadlock caused by numerical round errors.
    '''
    if mpi.main_rank:
        cost_bcast = cost(theta)
    else:
        cost_bcast = 0
    cost_bcast = mpi.comm.bcast(cost_bcast,root=0) 
    return cost_bcast

def jac_mpi(cost,theta,stepsize=1e-8):
    '''
       Given a cost function of varaibles theta,
       return the first derivatives (jacobian) 
       computed with MPI.
    '''
    ### Just in case, broadcast theta...
    mpi.comm.Bcast(theta,root=0)
    ndim =theta.size
    theta_d = copy.copy(theta)
    E0 = cost(theta)
    grad = np.zeros(ndim)
    grad_r = np.zeros(ndim)
    nrem = ndim%mpi.nprocs
    nblk = int((ndim - nrem)/mpi.nprocs)
    if mpi.rank < nrem:
        my_ndim = nblk + 1 
        ipos    = my_ndim * mpi.rank
    else:
        my_ndim = nblk
        ipos    = my_ndim * mpi.rank + nrem
    for iloop in range(ipos,ipos+my_ndim):
        theta_d[iloop] += stepsize
        Ep = cost(theta_d)
        #theta_d[iloop] -= stepsize * 2
        #Em = cost(theta_d)
        #theta_d[iloop] += stepsize
        #grad.append((Ep - Em)/(2*stepsize))
        theta_d[iloop] -= stepsize 
        grad[iloop] = (Ep - E0)/(stepsize)
    mpi.comm.Allreduce(grad, grad_r, mpi.MPI.SUM)
    return grad_r
def SaveTheta(ndim,theta,filepath):
    """ Function:
    Save theta(0:ndim-1) to filepath (overwritten)
    """
    with open(filepath,'w') as f:
        for i in range(ndim):
            print(theta[i],file=f)

def LoadTheta(ndim,filepath):
    """ Function:
    Read theta(0:ndim-1) from filepath
    """
    f=open(filepath)
    line=f.readlines()
    f.close
    theta=[] 
    for i in range(ndim):
        theta.append(float(line[i]))
    return theta

def error(message): 
    if mpi.main_rank:
        with open(cf.log,'a') as f:
            print(message,file=f)
    exit()

def T1vec2mat(noa,nob,nva,nvb,kappa_list):
    """ Function:
     Expand kappa_list to ta and tb
     [in]  kappa_list: occ-vir matrices of alpha and beta
     [out] (occ+vir)-(occ+vir) matrices of alpha and beta 
           (zeroes substituted in occ-occ and vir-vir)
    """
    ta = np.zeros((noa+nva,noa+nva))
    tb = np.zeros((noa+nva,noa+nva))
    ia = 0
    for a in range(nva):
        for i in range(noa):
            ta[a+noa,i] = kappa_list[ia]
            ta[i,a+noa] = -kappa_list[ia]
            ia += 1
    for a in range(nvb):
        for i in range(nob):
            tb[a+nob,i] = kappa_list[ia]
            tb[i,a+nob] = -kappa_list[ia]
            ia += 1
    return ta,tb

def T1mat2vec(noa,nob,nva,nvb,ta,tb):
    """ Function:
     Extract occ-vir block of ta and tb to make kappa_list
     [in]  (occ+vir)-(occ+vir) matrices of alpha and beta 
           (zeroes assumed in occ-occ and vir-vir)
     [out] kappa_list: occ-vir matrices of alpha and beta
    """
    kappa_list = np.zeros(noa*nva+nob*nvb)
    ia = 0
    for a in range(nva):
        for i in range(noa):
            kappa_list[ia] = ta[a+noa,i]
            ia += 1
    for a in range(nvb):
        for i in range(nob):
            kappa_list[ia] = tb[a+nob,i]
            ia += 1
    return kappa_list

def expAexpB(n,A,B):
    # Function:
    """ Function:
     Given n-by-n matrices A and B, do log(exp(A).exp(B))
    """
    from scipy.linalg import expm,logm
    C = logm(np.matmul(expm(A),expm(B)))
    return C

def T1mult(noa,nob,nva,nvb,kappa1,kappa2):
    """ Function:
     Given two kappa's, approximately combine them.
    """
    t1a,t1b = T1vec2mat(noa,nob,nva,nvb,kappa1)
    t2a,t2b = T1vec2mat(noa,nob,nva,nvb,kappa2)
    t12a = expAexpB(noa+nva,t1a,t2a)
    t12b = expAexpB(noa+nva,t1b,t2b)
    kappa12 = T1mat2vec(noa,nob,nva,nvb,t12a,t12b)
    return kappa12


def print_state(state,n_qubit=None):
    """
    print out quantum state as qubits
    """
    if n_qubit==None:
        n_qubit = state.get_qubit_count()
    opt='0'+str(n_qubit)+'b'
    with open(cf.log,'a') as f:
        print(" Basis       Coef", file=f)
        for i in range(2**n_qubit):
            v = state.get_vector()[i]
            if abs(v)**2>0.01:
                print('|',format(i,opt),'> : ', '{a.real:+.4f} {a.imag:+.4f}j'.format(a=v),file=f) 


def print_amplitudes(theta_list,noa,nob,nva,nvb,threshold=0.01):
    """ Function:
    Print out amplitudes
    """
    with open(cf.log,'a') as f:
        ### print singles amplitudes ###
        ia = 0
        for a in range(nva):
            aa = a + 1 + noa
            for i in range(noa):
                ii = i + 1 
                if abs(theta_list[ia]) > threshold:
                    print(ii, "a -> ", aa, "a  : ", '%2.10f' % theta_list[ia],file=f)
                ia = ia + 1
        for a in range(nvb):
            aa = a + 1 + nob
            for i in range(nob):
                ii = i + 1 
                if abs(theta_list[ia]) > threshold:
                    print(ii, "b -> ", aa, "b  : ",  '%2.10f' % theta_list[ia],file=f)
                ia = ia + 1
        ### print doubles amplitudes ###
        ijab = ia
        for b in range(nva):
            bb = b + 1 + noa 
            for a in range(b):
                aa = a + 1 + noa 
                for j in range(noa):
                    jj = j + 1 
                    for i in range(j):
                        ii = i + 1 

                        if abs(theta_list[ijab]) > threshold:
                            print(ii, "a", jj,"a -> ", aa, "a", bb,"a  : ", '%2.10f' % theta_list[ijab], file=f)
                        ijab = ijab + 1
                        
###     ab -> ab ###    
        for b in range(nvb):
            bb = b + 1 + nob
            for a in range(min(b+1,nva)):
                aa = a + 1 + noa
                for j in range(nob):
                    jj = j + 1 
                    for i in range(j+1):
                        ii = i + 1 
                        # b > a, j > i
                        if abs(theta_list[ijab]) > threshold:
                            print(ii, "a", jj,"b -> ", aa, "a", bb,"b  : ", '%2.10f' % (theta_list[ijab]), file=f)
                        ijab = ijab + 1
                    for i in range(j+1,noa):
                        ii = i + 1 
                        # b > a, i > j
                        if abs(theta_list[ijab]) > threshold:
                            print(ii, "a", jj,"b -> ", aa, "a", bb,"b  : ", '%2.10f' % (theta_list[ijab]), file=f)
                        ijab = ijab + 1
            for a in range(b+1,nva):
                aa = a + 1 + noa 
                for j in range(nob):
                    jj = j + 1 
                    for i in range(j+1):
                        ii = i + 1 
                        # a > b, j > i
                        if abs(theta_list[ijab]) > threshold:
                            print(ii, "a", jj,"b -> ", aa, "a", bb,"b  : ", '%2.10f' % (theta_list[ijab]), file=f)
                        ijab = ijab + 1
                    for i in range(j+1,noa):
                        ii = i + 1 
                        # a > b, i > j
                        if abs(theta_list[ijab]) > threshold:
                            print(ii, "a", jj,"b -> ", aa, "a", bb,"b  : ", '%2.10f' % (theta_list[ijab]), file=f)
                        ijab = ijab + 1
###     bb -> bb ###    
        for b in range(nvb):
            bb = b + 1 + nob
            for a in range(b):
                aa = a + 1 + nob
                for j in range(nob):
                    jj = j + 1
                    for i in range(j):
                        ii = i + 1
                        if abs(theta_list[ijab]) > threshold:
                            print(ii, "b", jj,"b -> ", aa, "b", bb,"b  : ", '%2.10f' % (theta_list[ijab]), file=f)
                        ijab = ijab + 1


def print_amplitudes_spinfree(theta_list,no,nv,threshold=0.01):
    """ Function:
    Print out amplitudes (for spin-free case)
    """
    from .ucclib import get_baji
    thres_i = int(1/threshold) + 1
    with open(cf.log,'a') as f:
        ### print singles amplitudes ###
        ia = 0
        for a in range(nv):
            aa = a + 1 + no
            for i in range(no):
                ii = i + 1 
                if abs(theta_list[ia]) > threshold:
                    print(ii, " -> ", aa, "  : ", '%2.10f' % theta_list[ia],file=f)
                ia += 1
        ### print doubles amplitudes ###
        for b in range(nv):
            bb = b + 1 + no
            for a in range(b):
                aa = a + 1 + no 
                for j in range(no):
                    jj = j + 1 
                    for i in range(j):
                        ii = i + 1 
                        baji = get_baji(b,a,j,i,no) + ia
                        abji = get_baji(a,b,j,i,no) + ia
                        theta = theta_list[baji] + theta_list[abji]
                        if abs(theta) > threshold:
                            print(ii, "a", jj,"a -> ", aa, "a", bb,"a  : ", '%2.10f' % theta, file=f)
                            print(ii, "b", jj,"b -> ", aa, "b", bb,"b  : ", '%2.10f' % theta, file=f)
                        
###     ab -> ab ###    
        for b in range(nv):
            bb = b + 1 + no
            for a in range(min(b+1,nv)):
                aa = a + 1 + no
                for j in range(no):
                    jj = j + 1 
                    for i in range(j+1):
                        ii = i + 1 
                        # b > a, j > i
                        baji = get_baji(b,a,j,i,no) + ia
                        if abs(theta_list[baji]) > threshold:
                            print(ii, "a", jj,"b -> ", aa, "a", bb,"b  : ", '%2.10f' % (theta_list[baji]), file=f)
                    for i in range(j+1,no):
                        ii = i + 1 
                        # b > a, i > j
                        baji = get_baji(b,a,j,i,no) + ia
                        if abs(theta_list[baji]) > threshold:
                            print(ii, "a", jj,"b -> ", aa, "a", bb,"b  : ", '%2.10f' % (theta_list[baji]), file=f)
            for a in range(b+1,nv):
                aa = a + 1 + no 
                for j in range(no):
                    jj = j + 1 
                    for i in range(j+1):
                        ii = i + 1 
                        # a > b, j > i
                        baji = get_baji(b,a,j,i,no) + ia
                        if abs(theta_list[baji]) > threshold:
                            print(ii, "a", jj,"b -> ", aa, "a", bb,"b  : ", '%2.10f' % (theta_list[baji]), file=f)
                    for i in range(j+1,no):
                        ii = i + 1 
                        # a > b, i > j
                        baji = get_baji(b,a,j,i,no) + ia
                        if abs(theta_list[baji]) > threshold:
                            print(ii, "a", jj,"b -> ", aa, "a", bb,"b  : ", '%2.10f' % (theta_list[baji]), file=f)
