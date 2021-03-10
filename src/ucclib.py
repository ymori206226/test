"""
#######################
#        quket        #
#######################

ucclib.py

Functions preparing UCC-type gates and circuits.
Cost functions are also defined here.

"""
import time
import itertools

import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import PauliRotation

from . import config as cf
from .fileio import (SaveTheta, print_state, print_amplitudes,
                     print_amplitudes_spinfree, prints)
from .expope import Gdouble_ope
from .init import get_occvir_lists
from .utils import orthogonal_constraint
from .hflib import set_circuit_rhf


def single_ope_Pauli(a, i, circuit, theta, approx=cf.approx_exp):
    """Function:
    Construct exp[theta(a!i - i!a)] as a whole unitary and add to circuit

    Author(s): Takashi Tsuchimochi
    """
    ndim1 = a - (i+1)
    ndim = ndim1 + 2

    ### Purpose:
    ### (1)   Exp[ i theta*0.5  Prod_{k=i+1}^{a-1} Z_k Y_i X_a]
    ### (2)   Exp[-i theta*0.5  Prod_{k=i+1}^{a-1} Z_k Y_a X_i]
    target_list = pauli_index = np.empty(ndim, dtype=int)
    target_list[:ndim1] = np.arange(i+1, a)
    target_list[ndim1:] = i, a
    pauli_index[:ndim1] = 3

    # (1)                Yi,Xa
    pauli_index[ndim1:] = 2, 1
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    # (2)                Xi,Ya
    pauli_index[ndim1:] = 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)


def double_ope_Pauli(b, a, j, i, circuit, theta):
    """Function:
    Construct exp[theta(a!b!ji - i!j!ba)]
    as a whole unitary and add to circuit

    Author(s): Takashi Tsuchimochi
    """
    ndim1 = min(a, j) - (i+1)
    ndim2 = ndim1 + b - (max(a, j)+1)
    ndim = ndim2 + 4

    ### Purpose:
    ### (1)   Exp[ i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (X_i X_j Y_a X_b)]
    ### (2)   Exp[ i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (Y_i X_j Y_a Y_b)]
    ### (3)   Exp[ i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (X_i Y_j Y_a Y_b)]
    ### (4)   Exp[ i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (X_i X_j X_a Y_b)]
    ### (5)   Exp[-i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (Y_i X_j X_a X_b)]
    ### (6)   Exp[-i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (X_i Y_j X_a X_b)]
    ### (7)   Exp[-i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (Y_i Y_j Y_a X_b)]
    ### (8)   Exp[-i theta/8  Prod_{k=i+1}^{j-1} Z_k  Prod_{l=a+1}^{b-1} Z_l  (Y_i Y_j X_a Y_b)]
    target_list = pauli_index = np.empty(ndim, dtype=int)
    target_list[:ndim1] = np.arange(i+1, min(a, j))
    target_list[ndim1:ndim2] = np.arange(max(a, j)+1, b)
    target_list[ndim2:] = i, j, a, b
    pauli_index[:ndim2] = 3

    ### (1)              Xi,Xj,Ya,Xb
    pauli_index[ndim2:] = 1, 1, 2, 1
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    ### (2)              Yi,Xj,Ya,Yb
    pauli_index[ndim2:] = 2, 1, 2, 2
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    ### (3)              Xi,Yj,Ya,Yb
    pauli_index[ndim2:] = 1, 2, 2, 2
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    ### (4)              Xi,Xj,Xa,Yb
    pauli_index[ndim2:] = 1, 1, 1, 2
    gate = PauliRotation(target_list, pauli_index, theta)
    circuit.add_gate(gate)

    ### (5)              Yi,Xj,Xa,Xb
    pauli_index[ndim2:] = 2, 1, 1, 1
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)

    ### (6)              Xi,Yj,Xa,Xb
    pauli_index[ndim2:] = 1, 2, 1, 1
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)

    ### (7)              Yi,Yj,Ya,Xb
    pauli_index[ndim2:] = 2, 2, 2, 1
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)

    ### (8)              Yi,Yj,Xa,Yb
    pauli_index[ndim2:] = 2, 2, 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta)
    circuit.add_gate(gate)


def single_ope(a, i, circuit, theta):
    """Function:
    Construct exp[theta(a!i - i!a)] by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_H_gate(a)
    circuit.add_RX_gate(i, -np.pi*0.5)
    for k in range(a, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, a):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(a)
    circuit.add_RX_gate(i, np.pi*0.5)

    circuit.add_H_gate(i)
    circuit.add_RX_gate(a, -np.pi*0.5)
    for k in range(a, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, a):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(i)
    circuit.add_RX_gate(a, np.pi*0.5)


def double_ope_1(b, a, j, i, circuit, theta):
    """Function:
    Construct the first part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_H_gate(b)
    circuit.add_H_gate(a)
    circuit.add_RX_gate(j, -np.pi*0.5)
    circuit.add_H_gate(i)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(b)
    circuit.add_H_gate(a)
    circuit.add_RX_gate(j, np.pi*0.5)
    circuit.add_H_gate(i)


def double_ope_2(b, a, j, i, circuit, theta):
    """Function:
    Construct the second part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """

    circuit.add_RX_gate(b, -np.pi*0.5)
    circuit.add_H_gate(a)
    circuit.add_RX_gate(j, -np.pi*0.5)
    circuit.add_RX_gate(i, -np.pi*0.5)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_RX_gate(b, np.pi*0.5)
    circuit.add_H_gate(a)
    circuit.add_RX_gate(j, np.pi*0.5)
    circuit.add_RX_gate(i, np.pi*0.5)


def double_ope_3(b, a, j, i, circuit, theta):
    """Function:
    Construct the third part of exp[theta ( a!b!ji - i!j!ba ) ] by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """

    circuit.add_H_gate(b)
    circuit.add_RX_gate(a, -np.pi*0.5)
    circuit.add_RX_gate(j, -np.pi*0.5)
    circuit.add_RX_gate(i, -np.pi*0.5)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(b)
    circuit.add_RX_gate(a, np.pi*0.5)
    circuit.add_RX_gate(j, np.pi*0.5)
    circuit.add_RX_gate(i, np.pi*0.5)


def double_ope_4(b, a, j, i, circuit, theta):
    """Function:
    Construct the fourth part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """
    circuit.add_H_gate(b)
    circuit.add_H_gate(a)
    circuit.add_H_gate(j)
    circuit.add_RX_gate(i, -np.pi*0.5)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(b)
    circuit.add_H_gate(a)
    circuit.add_H_gate(j)
    circuit.add_RX_gate(i, np.pi*0.5)


def double_ope_5(b, a, j, i, circuit, theta):
    """Function:
    Construct the fifth part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """

    circuit.add_RX_gate(b, -np.pi*0.5)
    circuit.add_H_gate(a)
    circuit.add_H_gate(j)
    circuit.add_H_gate(i)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_RX_gate(b, np.pi*0.5)
    circuit.add_H_gate(a)
    circuit.add_H_gate(j)
    circuit.add_H_gate(i)


def double_ope_6(b, a, j, i, circuit, theta):
    """Function:
    Construct the sixth part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """

    circuit.add_H_gate(b)
    circuit.add_RX_gate(a, -np.pi*0.5)
    circuit.add_H_gate(j)
    circuit.add_H_gate(i)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_H_gate(b)
    circuit.add_RX_gate(a, np.pi*0.5)
    circuit.add_H_gate(j)
    circuit.add_H_gate(i)


def double_ope_7(b, a, j, i, circuit, theta):
    """Function:
    Construct the seventh part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """

    circuit.add_RX_gate(b, -np.pi*0.5)
    circuit.add_RX_gate(a, -np.pi*0.5)
    circuit.add_RX_gate(j, -np.pi*0.5)
    circuit.add_H_gate(i)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_RX_gate(b, np.pi*0.5)
    circuit.add_RX_gate(a, np.pi*0.5)
    circuit.add_RX_gate(j, np.pi*0.5)
    circuit.add_H_gate(i)


def double_ope_8(b, a, j, i, circuit, theta):
    """Function:
    Construct the eighth part of exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ and add to circuit

    Author(s): Yuto Mori
    """

    circuit.add_RX_gate(b, -np.pi*0.5)
    circuit.add_RX_gate(a, -np.pi*0.5)
    circuit.add_H_gate(j)
    circuit.add_RX_gate(i, -np.pi*0.5)
    for k in range(b, a, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_CNOT_gate(a, j)

    for k in range(j, i, -1):
        circuit.add_CNOT_gate(k, k-1)

    circuit.add_RZ_gate(i, -theta)
    for k in range(i, j):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_CNOT_gate(a, j)

    for k in range(a, b):
        circuit.add_CNOT_gate(k+1, k)

    circuit.add_RX_gate(b, np.pi*0.5)
    circuit.add_RX_gate(a, np.pi*0.5)
    circuit.add_H_gate(j)
    circuit.add_RX_gate(i, np.pi*0.5)


def double_ope(b, a, j, i, circuit, theta):
    """Function:
    Wrapper for exp[theta(a!b!ji - i!j!ba)]
    by CNOTs and RZ to be added to circuit

    Author(s): Yuto Mori
    """
    double_ope_1(b, a, j, i, circuit, theta)
    double_ope_2(b, a, j, i, circuit, theta)
    double_ope_3(b, a, j, i, circuit, theta)
    double_ope_4(b, a, j, i, circuit, theta)
    double_ope_5(b, a, j, i, circuit, theta)
    double_ope_6(b, a, j, i, circuit, theta)
    double_ope_7(b, a, j, i, circuit, theta)
    double_ope_8(b, a, j, i, circuit, theta)


def ucc_occrot(circuit, noa, nob, nva, nvb, theta_list, ndim2=0):
    """Function:
    Construct circuit to rotate occ-occ part (singles),
    prod_ij exp[theta(i!j - j!i)]

    Author(s): Yuto Mori
    """
    # Test for occ-occ rotation (mostly or exactly redundant!)
    ia = ndim2
    ### alpha ###
    for a in range(noa):
        a2 = 2*a
        for i in range(a):
            i2 = 2*i
            single_ope(a2, i2, circuit, theta_list[ia])
            ia += 1
    ### beta ###
    for a in range(nob):
        a2 = 2*a + 1
        for i in range(a):
            i2 = 2*i + 1
            single_ope(a2, i2, circuit, theta_list[ia])
            ia += 1


def ucc_Gsingles(circuit, norbs, theta_list, ndim2=0):
    """Function:
    Construct circuit for generalized singles prod_pq exp[theta (p!q - q!p )]

    Author(s): Yuto Mori
    """
    ia = ndim2
    ### alpha ###
    for a in range(norbs):
        a2 = 2*a
        for i in range(a):
            i2 = 2*i
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ia += 1
    ### beta ###
    for a in range(norbs):
        a2 = 2*a + 1
        for i in range(a):
            i2 = 2*i + 1
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ia += 1


def ucc_singles(circuit, noa, nob, nva, nvb, theta_list, ndim2=0):
    """Function:
    Construct circuit for UCC singles  prod_ai exp[theta (a!i - i!a )]

    Author(s): Yuto Mori
    """
    ia = ndim2
    ### alpha ###
    for a in range(nva):
        a2 = 2*(a+noa)
        for i in range(noa):
            i2 = 2*i
            #single_ope(a2,i2,circuit,theta_list[ia])
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ia += 1
    ### beta ###
    for a in range(nvb):
        a2 = 2*(a+nob) + 1
        for i in range(nob):
            i2 = 2*i + 1
            #single_ope(a2,i2,circuit,theta_list[ia])
            single_ope_Pauli(a2, i2, circuit, theta_list[ia])
            ia += 1


def ucc_doubles(circuit, noa, nob, nva, nvb, theta_list, ndim1=0):
    """Function:
    Construct circuit for UCC doubles  prod_abij exp[theta (a!b!ji - i!j!ba )]

    Author(s): Yuto Mori
    """
    ### aa -> aa ###
    ijab = ndim1
    for b in range(nva):
        b2 = 2*(b+noa)
        for a in range(b):
            a2 = 2*(a+noa)
            for j in range(noa):
                j2 = 2*j
                for i in range(j):
                    i2 = 2*i
                    #double_ope(b2, a2, j2, i2, circuit, theta_list[ijab])
                    double_ope_Pauli(b2, a2, j2, i2, circuit, theta_list[ijab])
                    ijab += 1
    ### ab -> ab ###
    for b in range(nvb):
        b2 = 2*(b+nob) + 1
        #for a in range(min(b+1,nva)):
        for a in range(nva):
            a2 = 2*(a+noa)
            for j in range(nob):
                j2 = 2*j + 1
                #for i in range(j+1):
                for i in range(noa):
                    # b > a, j > i
                    i2 = 2*i
                    #double_ope(max(b2, a2), min(b2, a2),
                    #           max(j2, i2), min(j2, i2),
                    #           circuit, theta_list[ijab])
                    double_ope_Pauli(max(b2, a2), min(b2, a2),
                                     max(j2, i2), min(j2, i2),
                                     circuit, theta_list[ijab])
                    ijab += 1
    ### bb -> bb ###
    for b in range(nvb):
        b2 = 2*(b+nob) + 1
        for a in range(b):
            a2 = 2*(a+nob) + 1
            for j in range(nob):
                j2 = 2*j + 1
                for i in range(j):
                    i2 = 2*i + 1
                    #double_ope(b2, a2, j2, i2, circuit, theta_list[ijab])
                    double_ope_Pauli(b2, a2, j2, i2, circuit, theta_list[ijab])
                    ijab += 1


def set_circuit_occrot(n_qubit_system, noa, nob, nva, nvb, theta1):
    """Function:
    Construct new circuit for occ-occ rotation,  prod_ij exp[theta (i!j - j!i )]

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubit_system)
    ucc_occrot(circuit, noa, nob, nva, nvb, theta1)
    return circuit


def set_circuit_GS(n_qubit_system, noa, nob, nva, nvb, theta1):
    """Function:
    Construct new circuit for generalized singles,  prod_pq exp[theta (p!q - q!p )]

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubit_system)
    ucc_Gsingles(circuit, norbs, theta1)
    return circuit


def set_circuit_uccsd(n_qubits, noa, nob, nva, nvb, DS, theta_list, ndim1):
    """Function:
    Construct new circuit for UCCSD

    Author(s): Yuto Mori, Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    if DS:
        ucc_singles(circuit, noa, nob, nva, nvb, theta_list, 0)
        ucc_doubles(circuit, noa, nob, nva, nvb, theta_list, ndim1)
    else:
        ucc_doubles(circuit, noa, nob, nva, nvb, theta_list, ndim1)
        ucc_singles(circuit, noa, nob, nva, nvb, theta_list, 0)
    return circuit


def set_circuit_sauccsd(n_qubits, no, nv, DS, theta_list, ndim1):
    """Function:
    Construct new circuit for spin-adapted UCCSD

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    if DS:
        ucc_singles_spinfree(circuit, no, nv, theta_list, 0)
        ucc_doubles_spinfree1(circuit, no, no, nv, nv, theta_list, ndim1)
    else:
        ucc_doubles_spinfree1(circuit, no, no, nv, nv, theta_list, ndim1)
        ucc_singles_spinfree(circuit, no, nv, theta_list, 0)
    return circuit


def set_circuit_sauccd(n_qubits, no, nv, theta_list):
    """Function:
    Construct new circuit for spin-adapted UCCD

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    ucc_doubles_spinfree1(circuit, no, no, nv, nv, theta_list, 0)
    return circuit


def set_circuit_uccd(n_qubits, noa, nob, nva, nvb, theta_list):
    """Function:
    Construct new circuit for UCCD

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    ucc_doubles(circuit, noa, nob, nva, nvb, theta_list)
    return circuit


def cost_uccd(Quket, print_level, kappa_list, theta_list, threshold=1e-2):
    """Function:
    Energy functional of UCCD
    !!!!! Not maintained and thus may fail !!!!!

    Author(s): Takashi Tsuchimochi
    """
    t1 = time.time()

    rho = Quket.rho
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    n_qubits = Quket.n_qubits
    n_electrons = Quket.n_electrons
    ndim = Quket.ndim

    state = QuantumState(n_qubits)
    set_circuit = set_circuit_rhf(n_qubits, n_electrons)
    set_circuit.update_quantum_state(state)

    circuit = set_circuit_uccd(n_qubits, noa, nob, nva, nvb, theta_list)
    for i in range(rho):
        circuit.update_quantum_state(state)
    Euccd = Quket.qulacs.Hamiltonian.get_expectation_value(state)
    S2 = Quket.qulacs.S2.get_expectation_value(state)

    t2 = time.time()
    cput = t2 - t1
    if print_level > 0:
        cf.icyc += 1
        prints(f"{cf.icyc:5d}: E[UCCD] = {Euccd:.12f}  <S**2> = {S2:17.15f}  "
               f"CPU Time = {cput:2.5f}")
        SaveTheta(ndim, theta_list, cf.tmp)
    if print_level > 1:
        prints("\n(UCCD state)")
        print_state(state)

    # Store UCCD wave function
    Quket.state = state
    return Euccd, S2


#############################
#   Spin-free UCC modules   #
#############################
def get_baji(b, a, j, i, no):
    """Function:
    Search the position for baji in the spin-adapted index

    Author(s): Takashi Tsuchimochi
    """
    bj = b*no + j
    ai = a*no + i
    if bj > ai:
        baji = bj*(bj+1)//2 + ai
    else:
        baji = ai*(ai+1)//2 + bj
    return baji


def ucc_singles_spinfree(circuit, no, nv, theta_list, ndim2=0):
    """Function:
    Wrapper for spin-adapted singles

    Author(s): Takashi Tsuchimochi
    """
    ia = ndim2
    ### alpha ###
    for a in range(nv):
        a2 = 2*(a+no)
        for i in range(no):
            i2 = 2*i
            single_ope(a2, i2, circuit, theta_list[ia])
            single_ope(a2+1, i2+1, circuit, theta_list[ia])
            ia += 1


def ucc_doubles_spinfree1(circuit, noa, nob, nva, nvb, theta_list, ndim1=0):
    """Function:
    Wrapper for spin-adapted doubles

    Author(s): Takashi Tsuchimochi
    """
    ### aa -> aa ###
    ijab = ndim1
    for b in range(nva):
        b2 = 2*(b+noa)
        for a in range(b):
            a2 = 2*(a+noa)
            for j in range(noa):
                j2 = 2*j
                for i in range(j):
                    i2 = 2*i
                    baji = get_baji(b, a, j, i, noa) + ndim1
                    abji = get_baji(a, b, j, i, noa) + ndim1
                    theta = theta_list[baji] + theta_list[abji]
                    double_ope_Pauli(b2, a2, j2, i2, circuit, theta)
                    ijab += 1

    ### ab -> ab ###
    for b in range(nvb):
        b2 = 2*(b+nob) + 1
        #for a in range(min(b+1, nva)):
        for a in range(nva):
            a2 = 2*(a+noa)
            for j in range(nob):
                j2 = 2*j + 1
                #for i in range(j+1):
                for i in range(noa):
                    # b > a, j > i
                    i2 = 2*i
                    baji = get_baji(b, a, j, i, noa) + ndim1
                    theta = theta_list[baji]
                    if i == j:
                        if a > b:
                            theta *= -1
                    if a == b:
                        if i > j:
                            theta *= -1
                    double_ope_Pauli(max(b2, a2), min(b2, a2),
                                     max(j2, i2), min(j2, i2),
                                     circuit, theta)
                    ijab += 1

    ### bb -> bb ###
    for b in range(nvb):
        b2 = 2*(b+nob) + 1
        for a in range(b):
            a2 = 2*(a+nob) + 1
            for j in range(nob):
                j2 = 2*j + 1
                for i in range(j):
                    i2 = 2*i + 1
                    baji = get_baji(b, a, j, i, noa) + ndim1
                    abji = get_baji(a, b, j, i, noa) + ndim1
                    theta = theta_list[baji] + theta_list[abji]
                    double_ope_Pauli(b2, a2, j2, i2, circuit, theta)
                    ijab += 1


def cost_uccsdX(Quket, print_level, kappa_list, theta_list, threshold=0.01):
    """Function:
    Energy functional of UCCSD (including spin-adapted UCCSD)
    Generalized to sequential excited state calculations,
    by projecting out UCCSD lower_states

    Author(s): Takashi Tsuchimochi
    """

    t1 = time.time()

    ansatz = Quket.ansatz
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    rho = Quket.rho
    DS = Quket.DS
    det = Quket.current_det
    n_qubit_system = Quket.n_qubits
    n_qubits = Quket.n_qubits
    ndim1 = Quket.ndim1
    ndim = Quket.ndim

    if ansatz == "sauccsd":
        state = create_sauccsd_state(n_qubit_system, noa, nva, rho, DS,
                                     theta_list, det, ndim1)
    else:
        state = create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1,
                                   SpinProj=Quket.projection.SpinProj)

    Euccsd = Quket.qulacs.Hamiltonian.get_expectation_value(state)
    S2 = Quket.qulacs.S2.get_expectation_value(state)
    cost = Euccsd

    ### Project out the states contained in 'lower_states'
    cost += orthogonal_constraint(Quket, state)

    if cf.constraint_lambda > 0:
        S4 = cf.qulacs_s4.get_expectation_value(state)
        penalty = cf.constraint_lambda*(S4)
        cost += penalty

    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == -1:
        prints(f"Initial E[{ansatz}] = {Euccsd:.12f}  "
               f"<S**2> = {S2:17.15f}  "
               f"rho = {rho}")
    if print_level == 1:
        # cf.constraint_lambda *= 1.1
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:5d}: E[{ansatz}] = {Euccsd:.12f}  "
               f"<S**2> = {S2:17.15f}  "
               f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        if cf.constraint_lambda != 0:
            prints(f"lambda = {cf.constraint_lambda}  "
                   f"<S**4> = {S4:17.15f}  "
                   f"Penalty = {penalty:2.15f}")
        SaveTheta(ndim, theta_list, cf.tmp)
    if print_level > 1:
        prints(f"Final: E[{ansatz}] = {Euccsd:.12f}  "
               f"<S**2> = {S2:17.15f}  "
               f"rho = {rho}")
        prints("\n(UCCSD state)")
        print_state(state)
        if ansatz == "uccsd":
            print_amplitudes(theta_list, noa, nob, nva, nvb, threshold)
        elif ansatz == "sauccsd":
            print_amplitudes_spinfree(theta_list, noa, nva, threshold)

    # Store UCCSD wave function
    Quket.state = state
    return cost, S2


def set_circuit_uccsdX(n_qubits, DS, theta_list, occ_list, vir_list, ndim1):
    """Function
    Prepare a Quantum Circuit for a UCC state
    from an arbitrary determinant specified by occ_list and vir_list.

    Author(s):  Yuto Mori
    """
    circuit = QuantumCircuit(n_qubits)
    if DS:
        ucc_singlesX(circuit, theta_list, occ_list, vir_list, 0)
        ucc_doublesX(circuit, theta_list, occ_list, vir_list, ndim1)
    else:
        ucc_doublesX(circuit, theta_list, occ_list, vir_list, ndim1)
        ucc_singlesX(circuit, theta_list, occ_list, vir_list, 0)
    return circuit


def ucc_singlesX(circuit, theta_list, occ_list, vir_list, ndim2=0):
    """Function
    Prepare a Quantum Circuit for the single exictation part of
    a Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ia = ndim2
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]

    ### alpha ###
    for a in vir_list_a:
        for i in occ_list_a:
            single_ope_Pauli(a, i, circuit, theta_list[ia])
            ia += 1

    ### beta ###
    for a in vir_list_b:
        for i in occ_list_b:
            single_ope_Pauli(a, i, circuit, theta_list[ia])
            ia += 1


def ucc_doublesX(circuit, theta_list, occ_list, vir_list, ndim1=0):
    """Function
    Prepare a Quantum Circuit for the double exictation part of
    a Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ijab = ndim1
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]

    ### aa -> aa ###
    for a, b in itertools.combinations(vir_list_a, 2):
        for i, j in itertools.combinations(occ_list_a, 2):
            if b > j:
                Gdouble_ope(b, a, j, i, circuit, theta_list[ijab])
            else:
                Gdouble_ope(j, i, b, a, circuit, theta_list[ijab])
            ijab += 1

    ### ab -> ab ###
    for b in vir_list_b:
        for a in vir_list_a:
            for j in occ_list_b:
                for i in occ_list_a:
                    if max(b, a) > max(j, i):
                        Gdouble_ope(max(b, a), min(b, a), max(j, i), min(j, i),
                                    circuit, theta_list[ijab])
                    else:
                        Gdouble_ope(max(j, i), min(j, i), max(b, a), min(b, a),
                                    circuit, theta_list[ijab])
                    ijab += 1

    ### bb -> bb ###
    for a, b in itertools.combinations(vir_list_b, 2):
        for i, j in itertools.combinations(occ_list_b, 2):
            if b > j:
                Gdouble_ope(b, a, j, i, circuit, theta_list[ijab])
            else:
                Gdouble_ope(j, i, b, a, circuit, theta_list[ijab])
            ijab += 1


def create_uccsd_state(n_qubits, rho, DS, theta_list, det, ndim1,
                       SpinProj=False):
    """Function
    Prepare a UCC state based on theta_list.
    The initial determinant 'det' contains the base-10 integer
    specifying the bit string for occupied orbitals.

    Author(s):  Yuto Mori, Takashi Tsuchimochi
    """
    ### Form RHF bits
    state = QuantumState(n_qubits)
    state.set_computational_basis(det)
    occ_list, vir_list = get_occvir_lists(n_qubits, det)

    theta_list_rho = theta_list/rho
    circuit = set_circuit_uccsdX(n_qubits, DS,
                                 theta_list_rho, occ_list, vir_list, ndim1)
    for i in range(rho):
        circuit.update_quantum_state(state)

    if SpinProj:
        from .phflib import S2Proj
        state = S2Proj(Quket, state)
    return state


def create_sauccsd_state(n_qubit_system, noa, nva, rho, DS, theta_list, det,
                         ndim1):
    """Function
    Prepare a UCC state based on theta_list.
    The initial determinant 'det' contains the base-10 integer
    specifying the bit string for occupied orbitals.

    Author(s):  Yuto Mori, Takashi Tsuchimochi
    """
    ### Form RHF bits

    state = QuantumState(n_qubit_system)
    state.set_computational_basis(det)
    theta_list_rho = theta_list/rho
    circuit = set_circuit_sauccsd(n_qubit_system, noa, nva, DS, theta_list_rho,
                                  ndim1)
    for i in range(rho):
        circuit.update_quantum_state(state)
    return state


def ucc_singles_g(circuit, no, nv, theta_list, ndim2=0):
    """Function:
    Construct circuit for UCC singles in the spin-generalized
            prod_ai exp[theta(a!i - i!a)]

    Author(s): Takashi Tsuchimochi
    """
    ia = ndim2
    ### alpha ###
    for a in range(nv):
        for i in range(no):
            #single_ope(a2, i2, circuit, theta_list[ia])
            single_ope_Pauli(a+no, i, circuit, theta_list[ia])
            ia += 1
