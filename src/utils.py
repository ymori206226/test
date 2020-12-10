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
import itertools
from .fileio import prints, printmat
from openfermion.transforms import jordan_wigner
from openfermion.ops import QubitOperator, FermionOperator
from openfermion.utils import commutator, count_qubits, hermitian_conjugated,  normal_ordered, eigenspectrum, QubitDavidson 
from qulacs.observable import create_observable_from_openfermion_text 
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from qulacs import QuantumState

import copy

def cost_mpi(cost,theta):
    """ Function
    Simply run the given cost function with varaibles theta,
    but ensure that all MPI processes contain the same cost.
    This should help eliminate the possible deadlock caused by numerical round errors.

    Author(s): Takashi Tsuchimochi
    """
    if mpi.main_rank:
        cost_bcast = cost(theta)
    else:
        cost_bcast = 0
    cost_bcast = mpi.comm.bcast(cost_bcast,root=0) 
    return cost_bcast

def jac_mpi(cost,theta,stepsize=1e-8):
    """ Function
    Given a cost function of varaibles theta,
    return the first derivatives (jacobian) 
    computed with MPI.

    Author(s): Takashi Tsuchimochi
    """
    ### Just in case, broadcast theta...
    mpi.comm.Bcast(theta,root=0)
    ndim =theta.size
    theta_d = copy.copy(theta)
    E0 = cost(theta)
    grad = np.zeros(ndim)
    grad_r = np.zeros(ndim)
    ipos, my_ndim = mpi.myrange(ndim)
    for iloop in range(ipos,ipos+my_ndim):
        theta_d[iloop] += stepsize
        Ep = cost(theta_d)
        theta_d[iloop] -= stepsize 
        grad[iloop] = (Ep - E0)/(stepsize)
    mpi.comm.Allreduce(grad, grad_r, mpi.MPI.SUM)
    return grad_r

def chkbool(string):
    """ Function:
    Check string and return True or False as bool

    Author(s): Takashi Tsuchimochi
    """
    if string.lower() in ('true', '1'):
        return True
    elif string.lower() in ('false', '0'):
        return False
    else:
        prints("Unrecognized argument `{}`".format(string))
        error("")


def T1vec2mat(noa,nob,nva,nvb,kappa_list):
    """ Function:
     Expand kappa_list to ta and tb
     [in]  kappa_list: occ-vir matrices of alpha and beta
     [out] (occ+vir)-(occ+vir) matrices of alpha and beta 
           (zeroes substituted in occ-occ and vir-vir)

    Author(s): Takashi Tsuchimochi
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

    Author(s): Takashi Tsuchimochi
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
    """ Function:
     Given n-by-n matrices A and B, do log(exp(A).exp(B))

    Author(s): Takashi Tsuchimochi
    """
    from scipy.linalg import expm,logm
    C = logm(np.matmul(expm(A),expm(B)))
    return C

def T1mult(noa,nob,nva,nvb,kappa1,kappa2):
    """ Function:
     Given two kappa's, approximately combine them.

    Author(s): Takashi Tsuchimochi
    """
    t1a,t1b = T1vec2mat(noa,nob,nva,nvb,kappa1)
    t2a,t2b = T1vec2mat(noa,nob,nva,nvb,kappa2)
    t12a = expAexpB(noa+nva,t1a,t2a)
    t12b = expAexpB(noa+nva,t1b,t2b)
    kappa12 = T1mat2vec(noa,nob,nva,nvb,t12a,t12b)
    return kappa12


def create_1body_operator(XA,XB=None):
    """ Function
    Given XA (=XB) as a (n_orbitals x n_orbitals) matrix, 
    return FermionOperator in OpenFermion Format.

    Author(s): Takashi Tsuchimochi
    """
    n_orbitals = XA.shape[0]
    Operator = FermionOperator('',0)
    for i in range(2*n_orbitals):
        for j in range(2*n_orbitals):
            string = str(j)+'^ '+str(i)
            ii = int(i/2)
            jj = int(j/2)
            if(i%2==0 and j%2==0): # Alpha-Alpha
                Operator += FermionOperator(string,XA[jj][ii]) 
            elif (i%2== 1 and j%2==1): #Beta-Beta
                if XB==None:
                    Operator += FermionOperator(string,XA[jj][ii]) 
                else:
                    Operator += FermionOperator(string,XB[jj][ii]) 
    return Operator
                    

def single_operator_gradient(p,q,jordan_wigner_hamiltonian,state,n_qubit):
    """ Function
    Compute gradient d<H>/dXpq

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    #与えられたpqからフェルミ演算子a_p!q-a_q!pを生成する
    #ダミーを作って後で引く
    dummy=FermionOperator(str(n_qubit-1)+'^ '+str(n_qubit-1),1.0)
    fermi=FermionOperator(str(p)+'^ '+str(q),1.0)+FermionOperator(str(q)+'^ '+str(p),-1.0)
    #フェルミ演算子をjordan_wigner変換する
    jordan_wigner_fermi=jordan_wigner(fermi)
    jordan_wigner_dummy=jordan_wigner(dummy)
    #交換子を用いてエネルギーの傾きを求める準備を行う
    jordan_wigner_gradient=commutator(jordan_wigner_fermi,jordan_wigner_hamiltonian)+jordan_wigner_dummy
    #オブザーバブルクラスに変換
    observable_gradient=create_observable_from_openfermion_text(str(jordan_wigner_gradient))
    observable_dummy=create_observable_from_openfermion_text(str(jordan_wigner_dummy))
    #オブザーバブルを用いてエネルギーの傾きを求める
    gradient=observable_gradient.get_expectation_value(state)-observable_dummy.get_expectation_value(state)
    
    return gradient                    

def FermionOperator_to_Observable(operator):
    """ Function
    Create qulacs observable from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    str_jw = str(jordan_wigner(operator))
    string = '(0.0000000000000000+0j) [Z'+str(2*cf.n_active_orbitals-1)+']'
    if str_jw == '0' :
        str_jw = string
    else:    
        str_jw += ' + \n' + string
    return create_observable_from_openfermion_text(str_jw)

def FermionOperator_to_Operator(operator):
    """ Function
    Create qulacs general operator from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    str_jw = str(jordan_wigner(operator))
    string = '(0.0000000000000000+0j) [Z'+str(2*cf.n_active_orbitals-1)+']'
    if str_jw == '0' :
        str_jw = string
    else:    
        str_jw += ' + \n' + string
    return create_quantum_operator_from_openfermion_text(str_jw)
