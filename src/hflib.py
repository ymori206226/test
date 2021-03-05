"""
#######################
#        quket        #
#######################

hflib.py

Functions related to initial qubit states (HF,UHF,etc.).

"""
import time

import numpy as np
from qulacs import QuantumState
from qulacs import QuantumCircuit

from . import config as cf
from .ucclib import ucc_singles, single_ope_Pauli
from .fileio import prints, SaveTheta, print_state


def set_circuit_rhf(Quket):
    circuit = QuantumCircuit(Quket.n_qubits)
    for i in range(Quket.n_electrons):
        circuit.add_X_gate(i)
    return circuit


#def set_circuit_rhf(n_qubit_system, n_electrons):
#    """Function:
#    Construct circuit for RHF |0000...1111>
#
#    Author(s): Yuto Mori
#    """
#    circuit = QuantumCircuit(n_qubit_system)
#    for i in range(n_electrons):
#        circuit.add_X_gate(i)
#    return circuit


def set_circuit_rohf(Quket):
    circuit = QuantumCircuit(Quket.n_qubits)
    for i in range(Quket.noa):
        circuit.add_X_gate(2*i)
    for i in range(Quket.nob):
        circuit.add_X_gate(2*i + 1)
    return circuit


#def set_circuit_rohf(n_qubit_system, noa, nob):
#    """Function:
#    Construct circuit for ROHF |0000...10101111>
#
#    Author(s): Yuto Mori, Takashi Tsuchimochi
#    """
#    circuit = QuantumCircuit(n_qubit_system)
#    for i in range(noa):
#        circuit.add_X_gate(2*i)
#    for i in range(nob):
#        circuit.add_X_gate(2*i + 1)
#    return circuit


def set_circuit_uhf(QuketData, kappa_list):
    circuit = QuantumCircuit(Quket.n_qubits)
    ucc_singles(circuit, Quket, kappa_list)
    return circuit


#def set_circuit_uhf(n_qubit_system, noa, nob, nva, nvb, kappa_list):
#    """Function:
#    Construct circuit for UHF by orbital rotation
#
#    Author(s):  Takashi Tsuchimochi
#    """
#    circuit = QuantumCircuit(n_qubit_system)
#    ucc_singles(circuit, noa, nob, nva, nvb, kappa_list)
#    return circuit


def set_circuit_ghf(Quket, kappa_list):
    circuit = QuantumCircuit(Quket.n_qubits)
    pq = 0
    for p in range(Quket.n_qubits):
        for q in range(p):
            single_ope_Pauli(p, q, circuit, kappa_list[pq])
            pq += 1
    return circuit


#def set_circuit_ghf(n_qubit_system, kappa_list):
#    """Function:
#    Construct circuit for GHF by general spin orbital rotation
#
#    Author(s):  Takashi Tsuchimochi
#    """
#    circuit = QuantumCircuit(n_qubit_system)
#    pq = 0
#    for p in range(n_qubit_system):
#        for q in range(p):
#            single_ope_Pauli(p, q, circuit, kappa_list[pq])
#            pq += 1
#    return circuit


def cost_uhf(Quket, print_level, kappa_list):
    """Function:
    Energy functional of UHF

    Author(s):  Takashi Tsuchimochi
    """
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    n_electrons = noa + nob
    n_qubits = Quket.n_qubits

    t1 = time.time()
    state = QuantumState(Quket.n_qubits)
    if Quket.noa == Quket.nob:
        circuit_rhf = set_circuit_rhf(Quket)
    else:
        circuit_rhf = set_circuit_rohf(Quket)
    circuit_rhf.update_quantum_state(state)
    circuit_uhf = set_circuit_uhf(Quket, kappa_list)
    circuit_uhf.update_quantum_state(state)
    Euhf = Quket.qulacs.Hamiltonian.get_expectation_value(state)
    cost = Euhf
    ### Project out the states contained in 'lower_states'
    from .utils import orthogonal_constraint
    cost += orthogonal_constraint(Quket, state)
    S2 = Quket.qulacs.S2.get_expectation_value(state)
    t2 = time.time()
    cpu1 = t2 - t1
    if print_level > 0:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:5d}: "
               f"E[UHF] = {Euhf:.12f}  "
               f"<S**2> = {S2:17.15f}  "
               f"CPU Time = {cput:2.5f} ({cpu1:2.5f} / step)")
        #SaveTheta(noa * nva + nob * nvb, kappa_list, cf.tmp)
        SaveTheta(kappa_list, cf.tmp)
    if print_level > 1:
        prints("(UHF state)")
        print_state(state, n_qubits=n_qubits)

    # Store HF wave function
    Quket.state = state
    return Euhf, S2


def mix_orbitals(Quket, mix_level, random=False, angle=np.pi / 4):
    kappa = np.zeros(Quket.noa*Quket.nva + Quket.nob*Quket.nvb)
    if random:
        for a in range(Quket.nva):
            for i in range(Quket.noa):
                kappa[i + a * Quket.noa] = np.random.rand() - 0.5
        for a in range(Quket.nvb):
            for i in range(Quket.nob):
                kappa[i + a*Quket.nob + Quket.noa*Quket.nva] \
                        = np.random.rand() - 0.5
    else:
        for p in range(mix_level):
            a = p
            iA = Quket.noa - (p + 1)
            iB = Quket.nob - (p + 1)
            kappa[iA + a*Quket.noa] = angle
            kappa[iB + a*Quket.nob + Quket.noa*Quket.nva] = -angle
    return kappa


#def mix_orbitals(noa, nob, nva, nvb, mix_level, random=False, angle=np.pi / 4):
#    """Function:
#    Prepare kappa_list that mixes homo-lumo to break spin-symmetry.
#    ===parameters===
#        mix_level:  the number of orbital pairs to be mixed
#        random:     [Bool] randomly mix orbitals
#        angle:      mixing angle
#        kappa:      kappa amplitudes in return
#
#    Author(s):  Takashi Tsuchimochi
#    """
#    kappa = np.zeros(noa * nva + nob * nvb)
#    if random:
#        for a in range(nva):
#            for i in range(noa):
#                kappa[i + a * noa] = np.random.rand() - 0.5
#        for a in range(nvb):
#            for i in range(nob):
#                kappa[i + a * nob + noa * nva] = np.random.rand() - 0.5
#    else:
#        for p in range(mix_level):
#            a = p
#            iA = noa - (p + 1)
#            iB = nob - (p + 1)
#            kappa[iA + a * noa] = angle
#            kappa[iB + a * nob + noa * nva] = -angle
#    return kappa


def bs_orbitals(kappa, ialpha, aalpha, jbeta, bbeta, Quket):
    kappa[ialpha + (aalpha-Quket.noa)*Quket.noa] += np.pi/6
    kappa[jbeta + (bbeta-Quket.nob)*Quket.nob + Quket.noa*Quket.nva] += -np.pi/6


#def bs_orbitals(kappa, ialpha, aalpha, jbeta, bbeta, noa, nob, nva, nvb):
#    """Function:
#    Prepare kappa amplitudes that generates broken-symmetry exicted state
#    #   kappa:      kappa amplitudes in return
#    #   ialpha -->  aalpha
#    #   jbeta -->  bbeta
#
#    Author(s):  Takashi Tsuchimochi
#    """
#    kappa[ialpha + (aalpha - noa) * noa] += np.pi / 6
#    kappa[jbeta + (bbeta - nob) * nob + noa * nva] += -np.pi / 6
