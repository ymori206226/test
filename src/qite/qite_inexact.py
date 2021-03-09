import scipy
import numpy as np
from numpy import linalg as LA
import time

from qulacs import QuantumState
from openfermion.ops import QubitOperator
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.state import inner_product

from .. import mpilib as mpi
from .. import config as cf
from ..fileio import prints, print_state
from ..utils import lstsq
from .qite_function import calc_delta, calc_psi
from .qite_function import make_gate, make_pauli_id, make_state1, calc_inner1

def make_hamiltonian(model, nspin, nterm):
    if model == "heisenberg":
        sx = []
        sy = []
        sz = []
        for i in range(nspin):
            sx.append(QubitOperator("X" + str(i)))
            sy.append(QubitOperator("Y" + str(i)))
            sz.append(QubitOperator("Z" + str(i)))

        H = []
        active = []

        for term in range(nterm):
            jw_hamiltonian = 0 * QubitOperator("")
            if term == nspin - 1:
                i = term
                j = 0
                active.append([i, j])
                jw_hamiltonian += sx[i] * sx[j] + sy[i] * sy[j] + sz[i] * sz[j]
                str_jw_hamiltonian = str(jw_hamiltonian)
                observable = create_observable_from_openfermion_text(str_jw_hamiltonian)
                H.append(observable)
            else:
                i = term
                j = i + 1
                active.append([i, j])
                jw_hamiltonian += sx[i] * sx[j] + sy[i] * sy[j] + sz[i] * sz[j]
                str_jw_hamiltonian = str(jw_hamiltonian)
                observable = create_observable_from_openfermion_text(str_jw_hamiltonian)
                H.append(observable)

        jw_hamiltonian_full = 0 * QubitOperator("")
        for i in range(nspin):
            if i < nspin - 1:
                j = i + 1
            else:
                j = 0
            jw_hamiltonian_full += sx[i] * sx[j] + sy[i] * sy[j] + sz[i] * sz[j]
        str_jw_hamiltonian_full = str(jw_hamiltonian_full)
        observable_full = create_observable_from_openfermion_text(
            str_jw_hamiltonian_full
        )
    return H, active, observable_full


def qite_inexact(Quket, nterm, D):
    ### Parameter setting
    n = Quket.n_qubits
    db = Quket.dt
    ntime = Quket.maxiter
    qbit = Quket.det
    model = Quket.model
    threshold = Quket.ftol

    H, active, observable_full = make_hamiltonian(model, n, nterm)

    size = 4 ** D
    index = np.arange(n)
    delta = QuantumState(n)
    firststate = QuantumState(n)
    firststate.set_computational_basis(qbit)

    prints("Inexact QITE: Pauli operator group size = ", size)

    energy = []
    psi_dash = firststate.copy()
    value = observable_full.get_expectation_value(psi_dash)
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
        prints(
            "{:6.2f}:".format(t * db),
            "  E  = {:.12f}".format(value),
            "  CPU Time = {:5.2f}".format(cput),
        )
        if abs(dE) < threshold:
            break

        #if t == 0:
        #    xv = np.zeros((nterm, size))
        for term in range(nterm):
            active_qubit = active[term]
            observable = H[term]
            psi_dash_copy = psi_dash.copy()

            #mpi.comm.bcast(size, root=0)
            #S_part = np.zeros((size, size), dtype=complex)
            #S = np.zeros((size, size), dtype=complex)
            #sizeT = size * (size + 1) // 2
            #nblock = sizeT // mpi.nprocs
            #ij = mpi.rank * nblock
            #start = int(np.sqrt(2 * ij + 1 / 4) - 1 / 2)
            #end = int(np.sqrt(2 * (ij + nblock) + 1 / 4) - 1 / 2)
            #for i in range(start, end):
            #    for j in range(i + 1):
            #        S_part[i, j] = calc_inner1(i, j, n, active_qubit, index, psi_dash)
            #        ij += 1
            #    S[:i, i] = S[i, :i]

            S_part = np.zeros((size, size), dtype=complex)
            S = np.zeros((size, size), dtype=complex)
            sizeT = size * (size - 1) // 2
            ipos, my_ndim = mpi.myrange(sizeT)
            ij = 0
            for i in range(size):
                for j in range(i):
                    if ij in range(ipos, ipos + my_ndim):
                        S_part[i, j] = calc_inner1(i, j, n, active_qubit, index, psi_dash)
                        S_part[j, i] = S_part[i, j].conjugate()
                    ij += 1
            mpi.comm.Allreduce(S_part, S, mpi.MPI.SUM)
            for i in range(size):
                S[i,i] = 1

            sigma = []
            for i in range(size):
                state_i = make_state1(i, n, active_qubit, index, psi_dash)
                sigma.append(state_i)

            delta = calc_delta(psi_dash, observable, n, db)
            b_l = []
            for i in range(size):
                b_i = inner_product(sigma[i], delta)
                b_i = -2 * b_i.imag
                b_l.append(b_i)
            Amat = 2 * np.real(S)

            b_l = np.array(b_l)
            zct = np.dot(b_l, Amat)

            #def cost_fun(vct):
            #    return LA.norm(np.dot(Amat, vct) - b_l) ** 2

            #def J_cost_fun(vct):
            #    wct = np.dot(Amat, vct)
            #    wct = np.dot(Amat.T, wct)
            #    return 2.0 * (wct - zct)

            #x = scipy.optimize.minimize(
            #    cost_fun, x0=xv[term], method="Newton-CG", jac=J_cost_fun, tol=1e-8
            #).x
            #xv[term] = x.copy()
            x, res, rnk, s = lstsq(Amat, b_l, cond=1.0e-8)
            ### Just in case, broadcast a...
            mpi.comm.Bcast(a, root=0)

            a = x.copy()
            psi_dash = calc_psi(psi_dash_copy, n, index, a, active_qubit)

        value = observable_full.get_expectation_value(psi_dash)
        energy.append(value)
        dE = energy[t + 1] - energy[t]
