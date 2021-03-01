import numpy as np
import time
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import PauliRotation
from . import config as cf
from .fileio import (
    SaveTheta,
    print_state,
    print_amplitudes,
    print_amplitudes_spinfree,
    prints,
)
from .ucclib import single_ope_Pauli, ucc_Gsingles
from .expope import Gdouble_ope
from .utils import orthogonal_constraint

def set_circuit_upccgsd(n_qubits, norbs, theta_list, k):
    """Function:
    Construct new circuit for UpCCGSD

    Author(s): Takahiro Yoshikura
    """
    ndim1 = int(norbs * (norbs - 1) / 2)
    ndim2 = ndim1
    circuit = QuantumCircuit(n_qubits)

    i = 0

    for i in range(k):
        upcc_Gdoubles(circuit, norbs, theta_list, ndim1, ndim2, i)
        upcc_Gsingles(circuit, norbs, theta_list, ndim1, ndim2, i)
        i = i + 1

    return circuit

def set_circuit_epccgsd(n_qubits, norbs, theta_list, k):
    """Function:
    Construct new circuit for EpCCGSD

    Author(s): Takahiro Yoshikura
    """
    ndim1 = int(norbs * (norbs - 1) / 2)
    ndim2 = ndim1
    ndim  = ndim1 + ndim2
    circuit = QuantumCircuit(n_qubits)

    for i in range(k-1):
        upcc_Gdoubles(circuit, norbs, theta_list, ndim1, ndim2, i)
        upcc_Gsingles(circuit, norbs, theta_list, ndim1, ndim2, i)
    upcc_Gdoubles(circuit, norbs, theta_list, ndim1, ndim2, k-1)
    ucc_Gsingles(circuit, norbs, theta_list, ndim*(k-1) + ndim2)

    return circuit

def upcc_Gdoubles(circuit, norbs, theta_list, ndim1, ndim2, i):
    """Function:
    Construct circuit for UpCC (pair-dobles part)

    Author(s): Takahiro Yoshikura
    """

    ijab = (ndim1 + ndim2) * i

    for a in range(norbs):
        ###  alpha  ###
        a2 = 2 * a
        ### beta  ###
        a2b = a2 + 1

        for i in range(a):
            ###  alpha  ###
            i2 = 2 * i
            ###  beta  ###
            i2b = i2 + 1
            # double_ope(max(b2,a2),min(b2,a2),max(j2,i2),min(j2,i2),circuit,theta_list[ijab])
            Gdouble_ope(a2b,a2,i2b,i2,circuit,theta_list[ijab])
            ijab = ijab + 1


def upcc_Gsingles(circuit, norbs, theta_list, ndim1, ndim2, i):
    """Function:
    Construct circuit for UpCC (singles part)

    Author(s): Takahiro Yoshikura, Takashi Tsuchimochi (spin-free)
    """
    ia = ndim2 + i * (ndim1 + ndim2)
    for a in range(norbs):
        a2 = 2 * a
        for i in range(a):
            i2 = 2 * i
            ### alpha ###
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ### beta ###
            single_ope_Pauli(a2 + 1, i2 + 1, circuit,  theta_list[ia])
            ia = ia + 1

def cost_upccgsd(
    Quket,
    print_level,
    kappa_list,
    theta_list,
    k,
):
    """Function:
    Energy functional of UpCCGSD

    Author(s): Takahiro Yoshikura
    """

    t1 = time.time()
    noa = Quket.noa
    nob = Quket.nob
    norbs = Quket.n_orbitals
    n_qubits = Quket.n_qubits
    det = Quket.det

    ndim1 = int(norbs * (norbs - 1) / 2)
    ndim2 = int(ndim1)
    state = QuantumState(n_qubits)
    state.set_computational_basis(det)

    if "epccgsd" in Quket.ansatz:
        circuit = set_circuit_epccgsd(n_qubits, norbs, theta_list, k)
    else:    
        circuit = set_circuit_upccgsd(n_qubits, norbs, theta_list, k)
    circuit.update_quantum_state(state)
    if Quket.projection.SpinProj:
        from .phflib import S2Proj
        state_P = S2Proj(Quket,state)
        state   = state_P.copy()
    Eupccgsd = Quket.qulacs.Hamiltonian.get_expectation_value(state)
    cost = Eupccgsd
    ### Project out the states contained in 'lower_states'
    cost += orthogonal_constraint(Quket, state)

    S2 = Quket.qulacs.S2.get_expectation_value(state)
    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(
            "{cyc:5}:".format(cyc=cf.icyc),
            "  E[{}-UpCCGSD] = {:.12f}".format(k, Eupccgsd),
            "  <S**2> =",
            "% 17.15f" % S2,
            "  CPU Time = ",
            "%5.2f" % cput,
            " (%2.2f / step)" % cpu1,
        )
        SaveTheta(k * (ndim1 + ndim2), theta_list, cf.tmp)
    if print_level > 1:
        prints(
            "Final:  E[{}-UpCCGSD] = {:.12f}".format(k, Eupccgsd),
            "  <S**2> =",
            "% 17.15f" % S2,
        )
        prints("\n({}-UpCCGSD state)".format(k))
        print_state(state, threshold=Quket.print_amp_thres)
    # Store UpCCGSD wave function
    Quket.state = state
    return cost, S2

