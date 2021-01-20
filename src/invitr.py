"""
#######################
#        quket        #
#######################

invitr.py

Functions preparing inverse-iteration gates and circuits.

"""

import sys
import numpy as np
import math
import time
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import PauliRotation, merge
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import QubitOperator
from . import config as cf
from . import mpilib as mpi
from .fileio import (
    SaveTheta,
    print_state,
    print_amplitudes,
    print_amplitudes_spinfree,
    prints,
)


# ゲートを作る関数
def make_gate(n, index, pauli_id):
    circuit = QuantumCircuit(n)
    for i in range(len(index)):
        gate_number = index[i]
        if pauli_id[i] == 1:
            circuit.add_X_gate(gate_number)
        elif pauli_id[i] == 2:
            circuit.add_Y_gate(gate_number)
        elif pauli_id[i] == 3:
            circuit.add_Z_gate(gate_number)
    return circuit


# |chi_i> とnとpauliとdbとjを与えられたら|chi_i>  = h[i]|chi_dash>を計算
def multiply_Hpauli(chi_i, n, pauli, db, j):
    coef = pauli.get_coef()
    circuit = make_gate(n, pauli.get_index_list(), pauli.get_pauli_id_list())
    # |chi_i>  = h[i]|chi_dash>を計算
    circuit.update_quantum_state(chi_i)
    chi_i.multiply_coef(coef)
    chi_i.multiply_coef(-db / j)  # -db/jをかける
    return chi_i


def exp_iht(h, t):
    """
    任意のハミルトニアン項 h[i] = h_i * pauli[i] （ただしh_iは実の係数、pauli[i]はパウリのテンソル積）に対して
    量子ゲート exp[-i h[i] t] を作る
    """
    coef = h.get_coef()
    target_list = h.get_index_list()
    pauli_id = h.get_pauli_id_list()
    return PauliRotation(target_list, pauli_id, -t * coef.real)


def exp_iHt(H, t, n_qubit=None):
    """
    ハミルトニアンH = sum_i h[i] に対して、一次のTrotter近似
    Exp[-iHt] ~ Prod_i  Exp[-i h[i] t]
    を行う量子回路を生成する
    """
    nterms = H.get_term_count()
    if n_qubit == None:
        n_qubit = H.get_qubit_count()
    Circuit = QuantumCircuit(n_qubit)
    for i in nterms:
        h = H.get_term(i)
        Circuit.add_gate(exp_iht(h, t))
    return Circuit


def make_pauli_id(num, n, active):
    """
    4ビットの場合
    [0 0 0 0]
    [0 0 0 1]
    [0 0 0 2]
    [0 0 0 3]
    [0 0 1 0]
    [0 0 1 1]
    .....
    となるように作る
    """
    id = []

    quo = num
    for i in range(len(active)):
        rem = quo % 4
        quo = (quo - rem) // 4
        id.append(rem)
    id.reverse()
    full_id = np.zeros(n, dtype=int)
    j = 0
    for i in range(len(active)):
        full_id[active[i]] = id[j]
        j += 1
    return full_id


def make_index(n):
    index = []
    for i in range(n):
        index.append(i)
    return index
