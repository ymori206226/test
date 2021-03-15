import time

import scipy as sp
import numpy as np
from numpy import linalg as LA
from qulacs import QuantumState
from qulacs.state import inner_product

from .qite_function import calc_delta, calc_psi, calc_inner1, make_state1
from .. import config as cf
from .. import mpilib as mpi
from ..fileio import prints, print_state
from ..utils import lstsq


def qite_exact(Quket):
    nspin = Quket.n_qubits
    db = Quket.dt
    ntime = Quket.maxiter
    qbit = Quket.det
    observable = Quket.qulacs.Hamiltonian
    threshold = Quket.ftol

    active_qubit = [x for x in range(nspin)]
    n = nspin
    size = 4**nspin
    index = np.arange(n)
    delta = QuantumState(n)
    first_state = QuantumState(n)
    first_state.set_computational_basis(qbit)

    prints(f"Exact QITE: Pauli operator group size = {size}")

    energy = []
    psi_dash = first_state.copy()
    value = observable.get_expectation_value(psi_dash)
    energy.append(value)

    t1 = time.time()
    cf.t_old = t1
    dE = 100
    for t in range(ntime):
        t2 = time.time()
        cput = t2 - cf.t_old
        cf.t_old = t2
        if cf.debug:
            print_state(psi_dash)
        prints(f"{t*db:6.2f}: E = {value:.12f}  CPU Time = {cput:5.2f}")

        if abs(dE) < threshold:
            break
        #if t == 0:
        #    xv = np.zeros(size)
        psi_dash_copy = psi_dash.copy()

        #mpi.comm.bcast(size, root=0)
        #S_part = np.zeros((size, size), dtype=complex)
        #S = np.zeros((size, size), dtype=complex)
        #sizeT = size*(size+1)//2
        #nblock = sizeT//mpi.nprocs

        #ij = mpi.rank*nblock
        #start = int(np.sqrt(2*ij + 1/4) - 1/2)
        #end = int(np.sqrt(2*(ij+nblock) + 1/4) - 1/2)
        #for i in range(start, end):
        #    for j in range(i+1):
        #        S_part[i, j] = calc_inner1(i, j, n, active_qubit,
        #                                   index, psi_dash)
        #        ij += 1
        #    S[:i, i] = S[i, :i]
        #mpi.comm.Allreduce(S_part, S, mpi.MPI.SUM)

        S_part = np.zeros((size, size), dtype=complex)
        S = np.zeros((size, size), dtype=complex)
        sizeT = size*(size-1)//2
        ipos, my_ndim = mpi.myrange(sizeT)
        ij = 0
        for i in range(size):
            for j in range(i):
                if ij in range(ipos, ipos+my_ndim):
                    S_part[i, j] = calc_inner1(i, j, n, active_qubit,
                                               index, psi_dash)
                    S_part[j, i] = S_part[i, j].conjugate()
                ij += 1
        mpi.comm.Allreduce(S_part, S, mpi.MPI.SUM)
        for i in range(size):
            S[i, i] = 1

        sigma = []
        for i in range(size):
            state_i = make_state1(i, n, active_qubit, index, psi_dash)
            sigma.append(state_i)

        delta = calc_delta(psi_dash, observable, n, db)
        b_l = []
        b_l = np.empty(size)
        for i in range(size):
            b_i = inner_product(sigma[i], delta)
            b_i = -2*b_i.imag
            b_l[i] = b_i

        Amat = 2*np.real(S)
        zct = b_l@Amat

        #def cost_fun(vct):
        #    return LA.norm(Amat@vct - b_l)**2

        #def J_cost_fun(vct):
        #    wct = Amat.T@Amat@vct
        #    return 2.0*(wct-zct)

        #x = sp.optimize.minimize(cost_fun, x0=xv, method="Newton-CG",
        #                         jac=J_cost_fun, tol=1e-8).x
        #xv = x.copy()
        x, res, rnk, s = lstsq(Amat, b_l, cond=1.0e-8)
        a = x.copy()
        ### Just in case, broadcast a...
        mpi.comm.Bcast(a, root=0)
        psi_dash = calc_psi(psi_dash_copy, n, index, a, active_qubit)

        value = observable.get_expectation_value(psi_dash)
        energy.append(value)
        dE = energy[t+1] - energy[t]
