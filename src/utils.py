"""
#######################
#        quket        #
#######################

utils.py

Utilities.

"""
import copy
from operator import itemgetter
from itertools import combinations
#from operator import mul
#from functools import reduce

import numpy as np
import scipy as sp
from scipy.linalg import expm, logm, lstsq
from qulacs import QuantumState
from qulacs.state import inner_product

from . import config as cf
from . import mpilib as mpi
from .fileio import prints, printmat, error


def cost_mpi(cost, theta):
    """Function
    Simply run the given cost function with varaibles theta,
    but ensure that all MPI processes contain the same cost.
    This should help eliminate the possible deadlock caused by numerical round errors.

    Author(s): Takashi Tsuchimochi
    """
    cost_bcast = cost(theta) if mpi.main_rank else 0, 0
    cost_bcast = mpi.comm.bcast(cost_bcast, root=0)
    return cost_bcast


def jac_mpi(cost, theta, stepsize=1e-8):
    """Function
    Given a cost function of varaibles theta,
    return the first derivatives (jacobian)
    computed with MPI.

    Author(s): Takashi Tsuchimochi
    """
    ### Just in case, broadcast theta...
    mpi.comm.Bcast(theta, root=0)

    ndim = theta.size
    theta_d = copy.copy(theta)

    E0 = cost(theta)
    grad = np.zeros(ndim)
    grad_r = np.zeros(ndim)
    ipos, my_ndim = mpi.myrange(ndim)
    for iloop in range(ipos, ipos+my_ndim):
        theta_d[iloop] += stepsize
        Ep = cost(theta_d)
        theta_d[iloop] -= stepsize
        grad[iloop] = (Ep-E0)/stepsize
    mpi.comm.Allreduce(grad, grad_r, mpi.MPI.SUM)
    return grad_r


def chkbool(string):
    """Function:
    Check string and return True or False as bool

    Author(s): Takashi Tsuchimochi
    """
    if string.lower() in ("true", "1"):
        return True
    elif string.lower() in ("false", "0"):
        return False
    else:
        error(f"Unrecognized argument '{string}'")


def chkmethod(method, ansatz):
    """Function:
    Check method is available in method_list

    Author(s): Takashi Tsuchimochi
    """
    if ansatz is None:
        return True

    if method == "vqe":
        if (ansatz in cf.vqe_ansatz_list) \
        or ("pccgsd" in ansatz) or ("bcs" in ansatz):
            return True
        else:
            return False
    elif method in ("qite", "qlanczos"):
        if ansatz in cf.qite_ansatz_list:
            return True
        else:
            return False
    else:
        return False


def root_inv(A, eps=1e-8):
    """Function:
    Get A^-1/2 based on SVD. Dimensions may be reduced.

    Author(s): Takashi Tsuchimochi
    """

    u, s, vh = np.linalg.svd(A, hermitian=True)
    mask = s >= eps
    red_u = sp.compress(mask, u, axis=1)
    # Diagonal matrix of s**-1/2
    sinv2 = np.diag([1/np.sqrt(i) for i in s if i > eps])
    Sinv2 = red_u@sinv2
    return Sinv2


def T1vec2mat(noa, nob, nva, nvb, kappa_list):
    """Function:
    Expand kappa_list to ta and tb
    [in]  kappa_list: occ-vir matrices of alpha and beta
    [out] (occ+vir)-(occ+vir) matrices of alpha and beta
          (zeroes substituted in occ-occ and vir-vir)

    Author(s): Takashi Tsuchimochi
    """
    ta = np.zeros((noa+nva, noa+nva))
    tb = np.zeros((noa+nva, noa+nva))
    ia = 0
    for a in range(nva):
        for i in range(noa):
            ta[a+noa, i] = kappa_list[ia]
            ta[i, a+noa] = -kappa_list[ia]
            ia += 1
    for a in range(nvb):
        for i in range(nob):
            tb[a+nob, i] = kappa_list[ia]
            tb[i, a+nob] = -kappa_list[ia]
            ia += 1
    return ta, tb


def T1mat2vec(noa, nob, nva, nvb, ta, tb):
    """Function:
    Extract occ-vir block of ta and tb to make kappa_list
    [in]  (occ+vir)-(occ+vir) matrices of alpha and beta
          (zeroes assumed in occ-occ and vir-vir)
    [out] kappa_list: occ-vir matrices of alpha and beta

    Author(s): Takashi Tsuchimochi
    """
    kappa_list = np.zeros(noa*nva + nob*nvb)
    ia = 0
    for a in range(nva):
        for i in range(noa):
            kappa_list[ia] = ta[a+noa, i]
            ia += 1
    for a in range(nvb):
        for i in range(nob):
            kappa_list[ia] = tb[a+nob, i]
            ia += 1
    return kappa_list


def expAexpB(n, A, B):
    """Function:
    Given n-by-n matrices A and B, do log(exp(A).exp(B))

    Author(s): Takashi Tsuchimochi
    """
    return logm(np.matmul(expm(A), expm(B)))


def T1mult(noa, nob, nva, nvb, kappa1, kappa2):
    """Function:
    Given two kappa's, approximately combine them.

    Author(s): Takashi Tsuchimochi
    """
    t1a, t1b = T1vec2mat(noa, nob, nva, nvb, kappa1)
    t2a, t2b = T1vec2mat(noa, nob, nva, nvb, kappa2)
    t12a = expAexpB(noa+nva, t1a, t2a)
    t12b = expAexpB(noa+nva, t1b, t2b)
    kappa12 = T1mat2vec(noa, nob, nva, nvb, t12a, t12b)
    return kappa12


def Binomial(n, r):
    """Function:
    Given integers n and r, compute nCr

    Args:
       n (int): n of nCr
       r (int): r of nCr

    Returns:
       nCr
    """
# scipy.special.combの方が基本高速みたいです
    #r = min(r, n-r)
    #numer = reduce(mul, range(n, n-r, -1), 1)
    #denom = reduce(mul, range(1, r+1), 1)
    #return numer//denom
    return sp.special.comb(n, r, exact=True)



def orthogonal_constraint(Quket, state):
    """Function
    Compute the penalty term for excited states based on 'orthogonally-constrained VQE' scheme.
    """

    nstates = len(Quket.lower_states)
    extra_cost = 0
    for i in range(nstates):
        Ei = Quket.qulacs.Hamiltonian.get_expectation_value(
                Quket.lower_states[i])
        overlap = inner_product(Quket.lower_states[i], state)
        extra_cost += -Ei * abs(overlap)**2
    return extra_cost


def fci2qubit(norbs, nalpha, nbeta, fci_coeff):
    """Function
    Perform mapping from fci coefficients to qubit representation

    Args:
        norbs (int): number of active orbitals
        nalpha (int): number of alpha electrons
        nbeta (int): number of beta electrons
        fci_coeff (ndarray): FCI Coefficients in a (NDetA, NDetB) array with
                             NDetA = Choose(norbs, nalpha)
                             NDetB = Choose(norbs, nbeta)
    """
    #printmat(fci_coeff)
    NDetA = Binomial(norbs, nalpha)
    NDetB = Binomial(norbs, nbeta)
    if NDetA is not fci_coeff.shape[0] or NDetB is not fci_coeff.shape[1]:
        error(f"NDetA = {NDetA}  NDetB = {NDetB}  "
              f"fci_coeff = {fci_coeff.shape}"
              f"Wrong dimensions fci_coeff in fci2qubit")
    listA =  list(combinations(range(norbs), nalpha))
    listB =  list(combinations(range(norbs), nbeta))

    for isort in range(nalpha):
        listA = sorted(listA, key=itemgetter(isort))
    for isort in range(nbeta):
        listB = sorted(listB, key=itemgetter(isort))

    j = 0
    n_qubits = norbs*2
    opt = f"0{n_qubits}b"
    vec = np.zeros(2**n_qubits)
    for ib in range(NDetB):
        occB = np.array([n*2 + 1 for n in listB[ib]])

        for ia in range(NDetA):
            occA = np.array([n*2 for n in listA[ia]])
            #prints("Det {} {}".format(ia,ib),  "  occA ",occA, '  occB ',occB)
            k = np.sum(2**occA) + np.sum(2**occB)
            vec[k] = fci_coeff[ia, ib] if ia <= ib else -fci_coeff[ia, ib]
            #if abs(fci_coeff[ia, ib]) > 1e-4:
            #    prints(f"    Det# {j+1}: ", end="");
            #    prints(f"| {format(k, opt)} > : {fci_coeff[ia, ib]}")
            #j += 1
    fci_state = QuantumState(n_qubits)
    fci_state.load(vec)
    return fci_state


def lstsq(a, b,
          cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True, lapack_driver=None):
    """Function
    Wrapper for scipy.linalg.lstsq, which is known to have some bug
    related to 'SVD failure'.
    This wrapper simply tries lstsq some times until it succeeds...
    """
    for i in range(5):
        try:
            x, res, rnk, s = lstsq(a, b, cond=cond,
                                   overwrite_a=overwrite_a,
                                   overwrite_b=overwrite_b,
                                   check_finite=check_finite,
                                   lapack_driver=lapack_driver)
            break
        except:
            pass
    else:
        # Come if not break
        print("lstsq does not seem to converge...")

# Amat, b_l, a, zctが未定義
        def cost_fun(vct):
            return LA.norm(Amat@vct - b_l)**2

        def J_cost_fun(vct):
            wct = a.T@a@vct
            return 2.0*(wct-zct)

        x = scipy.optimize.minimize(cost_fun,
                                    method='Newton-CG',
                                    jac=J_cost_fun,
                                    tol=1e-8).x
        res = rnk = s = None
    return x, res, rnk, s
