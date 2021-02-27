"""
#######################
#        quket        #
#######################

hflib.py

Functions related to initial qubit states (HF,UHF,etc.).

"""

import numpy as np
import time
from qulacs import QuantumState
from qulacs import QuantumCircuit
from .ucclib import ucc_singles, single_ope_Pauli
from . import config as cf
from .fileio import prints, SaveTheta, print_state


def set_circuit_rhf(n_qubit_system, n_electron):
    """Function:
    Construct circuit for RHF |0000...1111>

    Author(s): Yuto Mori
    """
    circuit = QuantumCircuit(n_qubit_system)
    for i in range(n_electron):
        circuit.add_X_gate(i)
    return circuit


def set_circuit_rohf(n_qubit_system, noa, nob):
    """Function:
    Construct circuit for ROHF |0000...10101111>

    Author(s): Yuto Mori, Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubit_system)
    for i in range(noa):
        circuit.add_X_gate(2 * i)
    for i in range(nob):
        circuit.add_X_gate(2 * i + 1)
    return circuit


def set_circuit_uhf(n_qubit_system, noa, nob, nva, nvb, kappa_list):
    """Function:
    Construct circuit for UHF by orbital rotation

    Author(s):  Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubit_system)
    ucc_singles(circuit, noa, nob, nva, nvb, kappa_list)
    return circuit


def set_circuit_ghf(n_qubit_system, kappa_list):
    """Function:
    Construct circuit for GHF by general spin orbital rotation

    Author(s):  Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubit_system)
    pq = 0
    for p in range(n_qubit_system):
        for q in range(p):
            single_ope_Pauli(p, q, circuit, kappa_list[pq])
            pq += 1
    return circuit


def cost_uhf(
    Quket,
    print_level,
    kappa_list,
):
    """Function:
    Energy functional of UHF

    Author(s):  Takashi Tsuchimochi
    """
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    n_electron = noa + nob
    n_qubit = Quket.n_qubit

    t1 = time.time()
    state = QuantumState(n_qubit)
    if noa == nob:
        circuit_rhf = set_circuit_rhf(n_qubit, n_electron)
    else:
        circuit_rhf = set_circuit_rohf(n_qubit, noa, nob)
    circuit_rhf.update_quantum_state(state)
    circuit_uhf = set_circuit_uhf(n_qubit, noa, nob, nva, nvb, kappa_list)
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
        prints(
            "{cyc:5}:".format(cyc=cf.icyc),
            "  E[UHF] = ",
            "{:.12f}".format(Euhf),
            "  <S**2> =",
            "% 17.15f" % S2,
            "  CPU Time = ",
            "%2.5f" % cput,
            " (%2.5f / step)" % cpu1,
        )
        SaveTheta(noa * nva + nob * nvb, kappa_list, cf.tmp)
    if print_level > 1:
        prints("(UHF state)")
        print_state(state, n_qubit=n_qubit)
    # Store HF wave function
    Quket.state = state
    return Euhf, S2


def mix_orbitals(noa, nob, nva, nvb, mix_level, random=False, angle=np.pi / 4):
    """Function:
    Prepare kappa_list that mixes homo-lumo to break spin-symmetry.
    ===parameters===
        mix_level:  the number of orbital pairs to be mixed
        random:     [Bool] randomly mix orbitals
        angle:      mixing angle
        kappa:      kappa amplitudes in return

    Author(s):  Takashi Tsuchimochi
    """
    kappa = np.zeros(noa * nva + nob * nvb)
    if random:
        for a in range(nva):
            for i in range(noa):
                kappa[i + a * noa] = np.random.rand() - 0.5
        for a in range(nvb):
            for i in range(nob):
                kappa[i + a * nob + noa * nva] = np.random.rand() - 0.5
    else:
        for p in range(mix_level):
            a = p
            iA = noa - (p + 1)
            iB = nob - (p + 1)
            kappa[iA + a * noa] = angle
            kappa[iB + a * nob + noa * nva] = -angle
    return kappa


def bs_orbitals(kappa, ialpha, aalpha, jbeta, bbeta, noa, nob, nva, nvb):
    """Function:
    Prepare kappa amplitudes that generates broken-symmetry exicted state
    #   kappa:      kappa amplitudes in return
    #   ialpha -->  aalpha
    #   jbeta -->  bbeta

    Author(s):  Takashi Tsuchimochi
    """
    kappa[ialpha + (aalpha - noa) * noa] += np.pi / 6
    kappa[jbeta + (bbeta - nob) * nob + noa * nva] += -np.pi / 6
