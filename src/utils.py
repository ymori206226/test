"""
#######################
#        quket        #
#######################

utils.py

Utilities.

"""

from . import config as cf
from . import mpilib as mpi
from .fileio import prints, error


def cost_mpi(cost, theta):
    """Function
    Simply run the given cost function with varaibles theta,
    but ensure that all MPI processes contain the same cost.
    This should help eliminate the possible deadlock caused by numerical round errors.

    Author(s): Takashi Tsuchimochi
    """
    if mpi.main_rank:
        cost_bcast = cost(theta)
    else:
        cost_bcast = 0
    cost_bcast = mpi.comm.bcast(cost_bcast, root=0)
    return cost_bcast


def jac_mpi(cost, theta, stepsize=1e-8):
    """Function
    Given a cost function of varaibles theta,
    return the first derivatives (jacobian)
    computed with MPI.

    Author(s): Takashi Tsuchimochi
    """
    import numpy as np
    import copy

    ### Just in case, broadcast theta...
    mpi.comm.Bcast(theta, root=0)
    ndim = theta.size
    theta_d = copy.copy(theta)
    E0 = cost(theta)
    grad = np.zeros(ndim)
    grad_r = np.zeros(ndim)
    ipos, my_ndim = mpi.myrange(ndim)
    for iloop in range(ipos, ipos + my_ndim):
        theta_d[iloop] += stepsize
        Ep = cost(theta_d)
        theta_d[iloop] -= stepsize
        grad[iloop] = (Ep - E0) / (stepsize)
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
        prints("Unrecognized argument `{}`".format(string))
        error("")


def chkmethod(method):
    """Function:
    Check method is available in method_list

    Author(s): Takashi Tsuchimochi
    """
    if method in cf.vqe_method_list:
        return True
    elif "upccgsd" in method:
        return True
    else:
        return False


def root_inv(A, eps=1e-8):
    """Function:
    Get A^-1/2 based on SVD. Dimensions may be reduced.

    Author(s): Takashi Tsuchimochi
    """
    import numpy as np
    import scipy as sp

    u, s, vh = np.linalg.svd(A, hermitian=True)
    mask = s >= eps
    red_u = sp.compress(mask, u, axis=1)
    # Diagonal matrix of s**-1/2
    sinv2 = np.diag([1 / np.sqrt(i) for i in s if i > eps])
    Sinv2 = red_u @ sinv2
    return Sinv2


def T1vec2mat(noa, nob, nva, nvb, kappa_list):
    """Function:
     Expand kappa_list to ta and tb
     [in]  kappa_list: occ-vir matrices of alpha and beta
     [out] (occ+vir)-(occ+vir) matrices of alpha and beta
           (zeroes substituted in occ-occ and vir-vir)

    Author(s): Takashi Tsuchimochi
    """
    import numpy as np

    ta = np.zeros((noa + nva, noa + nva))
    tb = np.zeros((noa + nva, noa + nva))
    ia = 0
    for a in range(nva):
        for i in range(noa):
            ta[a + noa, i] = kappa_list[ia]
            ta[i, a + noa] = -kappa_list[ia]
            ia += 1
    for a in range(nvb):
        for i in range(nob):
            tb[a + nob, i] = kappa_list[ia]
            tb[i, a + nob] = -kappa_list[ia]
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
    import numpy as np

    kappa_list = np.zeros(noa * nva + nob * nvb)
    ia = 0
    for a in range(nva):
        for i in range(noa):
            kappa_list[ia] = ta[a + noa, i]
            ia += 1
    for a in range(nvb):
        for i in range(nob):
            kappa_list[ia] = tb[a + nob, i]
            ia += 1
    return kappa_list


def expAexpB(n, A, B):
    """Function:
     Given n-by-n matrices A and B, do log(exp(A).exp(B))

    Author(s): Takashi Tsuchimochi
    """
    import numpy as np
    from scipy.linalg import expm, logm

    C = logm(np.matmul(expm(A), expm(B)))
    return C


def T1mult(noa, nob, nva, nvb, kappa1, kappa2):
    """Function:
     Given two kappa's, approximately combine them.

    Author(s): Takashi Tsuchimochi
    """
    t1a, t1b = T1vec2mat(noa, nob, nva, nvb, kappa1)
    t2a, t2b = T1vec2mat(noa, nob, nva, nvb, kappa2)
    t12a = expAexpB(noa + nva, t1a, t2a)
    t12b = expAexpB(noa + nva, t1b, t2b)
    kappa12 = T1mat2vec(noa, nob, nva, nvb, t12a, t12b)
    return kappa12


def orthogonal_constraint(qulacs_hamiltonian, state):
    """Function
    Compute the penalty term for excited states based on 'orthogonally-constrained VQE' scheme.
    """
    from qulacs.state import inner_product

    nstates = len(cf.lower_states)
    extra_cost = 0
    for i in range(nstates):
        Ei = qulacs_hamiltonian.get_expectation_value(cf.lower_states[i])
        overlap = inner_product(cf.lower_states[i], state)
        extra_cost += -Ei * abs(overlap) ** 2
    return extra_cost
