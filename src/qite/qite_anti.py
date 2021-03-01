import time
import scipy
import numpy as np
from numpy import linalg as LA
from openfermion.transforms import reverse_jordan_wigner
from openfermion.utils import normal_ordered
from qulacs import QuantumState, QuantumCircuit
from qulacs.state import inner_product

from .. import config as cf
from .. import mpilib as mpi
from ..fileio import error, prints, print_state, printmat
from ..utils import lstsq
from .qite_function import (
    fermi_to_str,
    fermi_to_str_heisenberg,
    conv_anti,
    anti_to_base,
)
from .qite_function import make_gate, make_index
from .qite_function import calc_delta, calc_psi_lessH, qite_s_operators


def make_antisymmetric_group(
    fermionic_hamiltonian, jw_hamiltonian, model, nspin, ansatz, threshold
):
    if model == "hubbard" or "heisenberg" in model:
        fermionic_hamiltonian = reverse_jordan_wigner(jw_hamiltonian)
        fermionic_hamiltonian = normal_ordered(fermionic_hamiltonian)
        hamiltonian_list = fermi_to_str_heisenberg(fermionic_hamiltonian)
    else:
        hamiltonian_list = fermi_to_str(fermionic_hamiltonian, threshold=threshold)
    op = conv_anti(hamiltonian_list)
    id_set, size = anti_to_base(op, nspin)
    return id_set, size


def qite_anti(Quket, id_set, size):
    ### Parameter setting
    n = Quket.n_qubits
    db = Quket.dt
    ntime = Quket.maxiter
    qbit = Quket.det
    observable = Quket.qulacs.Hamiltonian
    S2_observable = Quket.qulacs.S2
    threshold = Quket.ftol
    S2 = 0

    prints("QITE: Pauli operator group size = ", size)
    if Quket.ansatz != "cite":
        sigma_list, sigma_ij_index, sigma_ij_coef = qite_s_operators(id_set, n)
        len_list = len(sigma_list)
        prints("      Unique sigma list = ", len_list)

    index = make_index(n)
    delta = QuantumState(n)
    firststate = QuantumState(n)
    firststate.set_computational_basis(qbit)

    energy = []
    psi_dash = firststate.copy()

    t1 = time.time()
    cf.t_old = t1

    En = observable.get_expectation_value(psi_dash)
    energy.append(En)
    if S2_observable is not None:
        S2 = S2_observable.get_expectation_value(psi_dash)
    dE = 100
    for t in range(ntime):
        t2 = time.time()
        cput = t2 - cf.t_old
        cf.t_old = t2
        if cf.debug:
            print_state(psi_dash)
        prints(
            "{:6.2f}:".format(t * db),
            "  E  = {:.12f}".format(En),
            "  <S**2> = {:17.15f}".format(S2),
            "  CPU Time = {:5.2f}".format(cput),
        )
        if abs(dE) < threshold:
            break
        if t == 0:
            xv = np.zeros(size)

        T0 = time.time()
        delta = calc_delta(psi_dash, observable, n, db)
        T1 = time.time()

        if Quket.ansatz == "cite":
            delta.add_state(psi_dash)
            psi_dash = delta.copy()

        else:
            # for i in range(size):
            #    pauli_id = id_set[i]
            #    circuit_i = make_gate(n, index, pauli_id)
            #    state_i = psi_dash.copy()
            #    circuit_i.update_quantum_state(state_i)
            #    print(i)
            #    for j in range(i+1):
            #        pauli_id = id_set[j]
            #        circuit_j = make_gate(n, index, pauli_id)
            #        state_j = psi_dash.copy()
            #        circuit_j.update_quantum_state(state_j)
            #        s = inner_product(state_j, state_i)
            #        S[i][j] = s
            #        S[j][i] = s
            ###
            ###  Compute Sij as expectation values of sigma_list
            Sij_my_list = np.zeros(len_list)
            Sij_list = np.zeros(len_list)
            ipos, my_ndim = mpi.myrange(len_list)
            T2 = time.time()
            for iope in range(ipos, ipos + my_ndim):
                val = sigma_list[iope].get_expectation_value(psi_dash)
                Sij_my_list[iope] = val
            T3 = time.time()
            mpi.comm.Allreduce(Sij_my_list, Sij_list, mpi.MPI.SUM)
            T4 = time.time()

            ### Distribute Sij
            ij = 0
            sizeT = size * (size - 1) // 2
            ipos, my_ndim = mpi.myrange(sizeT)
            S = np.zeros((size, size), dtype=complex)
            my_S = np.zeros((size, size), dtype=complex)
            for i in range(size):
                for j in range(i):
                    if ij in range(ipos, ipos + my_ndim):
                        ind = sigma_ij_index[ij]
                        coef = sigma_ij_coef[ij]
                        my_S[i][j] = coef * Sij_list[ind]
                        my_S[j][i] = my_S[i][j].conjugate()
                    ij += 1
            mpi.comm.Allreduce(my_S, S, mpi.MPI.SUM)
            for i in range(size):
                S[i][i] = 1

            T5 = time.time()

            sigma = []
            for i in range(size):
                pauli_id = id_set[i]
                circuit_i = make_gate(n, index, pauli_id)
                state_i = psi_dash.copy()
                circuit_i.update_quantum_state(state_i)
                sigma.append(state_i)
            T6 = time.time()

            b_l = []
            for i in range(size):
                b_i = inner_product(sigma[i], delta)
                b_i = -2 * b_i.imag
                b_l.append(b_i)
            Amat = 2 * np.real(S)

            T7 = time.time()

            b_l = np.array(b_l)
            zct = np.dot(b_l, Amat)

            def cost_fun(vct):
                return LA.norm(np.dot(Amat, vct) - b_l) ** 2

            def J_cost_fun(vct):
                wct = np.dot(Amat, vct)
                wct = np.dot(Amat.T, wct)
                return 2.0 * (wct - zct)

            # x = scipy.optimize.minimize(
            #    cost_fun, x0=xv, method='Newton-CG', jac=J_cost_fun, tol=1e-8).x
            # xv = x.copy()
            #x = scipy.optimize.least_squares(
            #    cost_fun, x0=xv, ftol=1e-8).x
            #xv = x.copy()
            x, res, rnk, s = lstsq(Amat, b_l, cond=1.0e-8)
            a = x.copy()
            ### Just in case, broadcast a...
            mpi.comm.Bcast(a, root=0)

            T8 = time.time()
            psi_dash = calc_psi_lessH(psi_dash, n, index, a, id_set)
            T9 = time.time()
            if cf.debug:
                prints('T0 -> T1  ',T1-T0 )
                prints('T1 -> T2  ',T2-T1 )
                prints('T2 -> T3  ',T3-T2 )
                prints('T3 -> T4  ',T4-T3 )
                prints('T4 -> T5  ',T5-T4 )
                prints('T5 -> T6  ',T6-T5 )
                prints('T6 -> T7  ',T7-T6 )
                prints('T7 -> T8  ',T8-T7 )
                prints('T8 -> T9  ',T9-T8 )

        En = observable.get_expectation_value(psi_dash)
        if S2_observable is not None:
            S2 = S2_observable.get_expectation_value(psi_dash)
        energy.append(En)
        dE = energy[t + 1] - energy[t]

    print_state(psi_dash, name="QITE")
    # End
