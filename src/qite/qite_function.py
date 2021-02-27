import numpy as np
import itertools
from pprint import pprint

from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.utils import hermitian_conjugated
from qulacs import QuantumCircuit
from qulacs.gate import PauliRotation
from qulacs.state import inner_product

from .. import mpilib as mpi
from .. import config as cf
from ..opelib import QubitOperator_to_Observable
from ..fileio import prints
from ..init import get_occvir_lists


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


def multiply_Hpauli(chi_i, n, pauli, db, j):
    coef = pauli.get_coef()
    circuit = make_gate(n, pauli.get_index_list(), pauli.get_pauli_id_list())
    circuit.update_quantum_state(chi_i)
    chi_i.multiply_coef(coef)
    chi_i.multiply_coef(-db / j)
    return chi_i


def make_pauli_id(num, n, active):
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


def calc_delta(psi, observable, n, db):
    """Function
    Form delta
        delta = (exp[-db H] - 1) psi
    where exp[-db H] is Taylor-expanded.
    """
    chi = psi.copy()
    chi_dash = psi.copy()
    expHpsi = psi.copy()  # Will hold exp[-db H] psi after while statement
    nterms = observable.get_term_count()
    d = 10.0
    j = 1
    while d > 1e-8:
        chi.multiply_coef(0)
        for i in range(nterms):
            chi_i = chi_dash.copy()
            pauli = observable.get_term(i)
            chi_i = multiply_Hpauli(chi_i, n, pauli, db, j)
            chi.add_state(chi_i)
        chi_dash = chi.copy()
        # chi  = H.psi  ->  1/2 H.(H.psi) -> 1/3 H.(1/2 H.(H.psi)) ...
        expHpsi.add_state(chi)
        j += 1
        # check chi =  (1/j!) H^j psi  is small enough
        d = chi.get_squared_norm()
        d = np.sqrt(d)
    norm = expHpsi.get_squared_norm()
    expHpsi.normalize(norm)
    psi0 = psi.copy()
    psi0.multiply_coef(-1)
    delta = expHpsi.copy()
    delta.add_state(psi0)
    return delta


def calc_msqite_delta(psi_dash, observable, n, db):
    chi = psi_dash.copy()
    chi_dash = psi_dash.copy()
    psi_dash_copy = psi_dash.copy()
    nterms = observable.get_term_count()
    d = 10.0
    j = 1
    while d > 1e-8:
        chi.multiply_coef(0)
        for i in range(nterms):
            chi_i = chi_dash.copy()
            pauli = observable.get_term(i)
            chi_i = multiply_Hpauli(chi_i, n, pauli, db, j)
            chi.add_state(chi_i)
        chi_dash = chi.copy()
        psi_dash.add_state(chi)
        j += 1
        d = chi.get_squared_norm()
        d = np.sqrt(d)
    psi = psi_dash_copy.copy()
    psi.multiply_coef(-1)
    delta = psi_dash.copy()
    delta.add_state(psi)
    return delta


def calc_psi(psi_dash, n, index, a, active):
    circuit = QuantumCircuit(n)
    for i in range(len(a)):
        if abs(a[i]) > 0:
            pauli_id = make_pauli_id(i, n, active)
            gate = PauliRotation(index, pauli_id, a[i] * -2)
            circuit.add_gate(gate)
    circuit.update_quantum_state(psi_dash)
    norm = psi_dash.get_squared_norm()
    psi_dash.normalize(norm)
    return psi_dash


def calc_psi_lessH(psi_dash, n, index, a, id_set):
    circuit = QuantumCircuit(n)
    for i in range(len(a)):
        if abs(a[i]) > 0:
            pauli_id = id_set[i]
            gate = PauliRotation(index, pauli_id, a[i] * -2)
            circuit.add_gate(gate)
    circuit.update_quantum_state(psi_dash)
    norm = psi_dash.get_squared_norm()
    psi_dash.normalize(norm)
    return psi_dash


def fermi_to_str_heisenberg(fermionic_hamiltonian):
    """
    フェルミ演算子をstring型のリストに変える
    """
    string = str(fermionic_hamiltonian).replace("]", "")
    string = string.replace("(", "")
    string = string.replace("+0j)", "")
    hamiltonian_list = [
        x.strip().split("[") for x in string.split("+") if not string.strip() == ""
    ]
    return hamiltonian_list


def fermi_to_str(fermionic_hamiltonian, threshold=0):
    """
    フェルミ演算子をstring型のリストに変える
    """
    if type(fermionic_hamiltonian) is FermionOperator:
        string = str(fermionic_hamiltonian).replace("]", "")
    else:
        string = str(get_fermion_operator(fermionic_hamiltonian)).replace("]", "")

    hamiltonian_list = [
        x.strip().split("[") for x in string.split("+") if not string.strip() == ""
    ]
    hamiltonian_list = sort_hamiltonian_list(hamiltonian_list, threshold)
    return hamiltonian_list


def sort_hamiltonian_list(hamiltonian_list, threshold):
    """Function
    Sort hamiltonian_list in descending order.
    If the coefficient is less than threshold in aboslute value, truncate.
    """
    hamiltonian_list_tmp = []
    if cf.debug and mpi.main_rank:
        pprint(hamiltonian_list)
    len_list = len(hamiltonian_list)
    for i in range(len_list):
        coef = float(hamiltonian_list[i][0])
        if abs(coef) > threshold:
            hamiltonian_list_tmp.append([coef, hamiltonian_list[i][1]])

    hamiltonian_list = sorted(
        hamiltonian_list_tmp, reverse=True, key=lambda x: abs(x[0])
    )
    new_len_list = len(hamiltonian_list)
    if threshold > 0:
        prints(
            "Truncation threshold = {} :  A total of {} terms truncated.".format(
                threshold, len_list - new_len_list
            )
        )
    return hamiltonian_list


def conv_id(str):
    """
    X,Y,Zを数字1,2,3に置き換える
    """
    if "X" in str[0]:
        num = 1
    elif "Y" in str[0]:
        num = 2
    elif "Z" in str[0]:
        num = 3
    else:
        num = 0
    return num


def conv_id2XYZ(pauli_id):
    """Function
    Set pauli_id ([0, 0, 1, 3], etc.) to pauli string, X2 Z3
    """
    n = len(pauli_id)
    pauli_str = ""
    for k in range(n):
        if pauli_id[k] == 1:
            pauli_str += " X" + str(k)
        elif pauli_id[k] == 2:
            pauli_str += " Y" + str(k)
        elif pauli_id[k] == 3:
            pauli_str += " Z" + str(k)
    return pauli_str


def conv_anti(hamiltonian_list):
    """
    string型にしたフェルミ演算子から
    反エルミートな演算子を作る
    """
    op = 0 * QubitOperator("")
    nterms = len(hamiltonian_list)
    for i in range(nterms):
        fop = FermionOperator(hamiltonian_list[i][1])
        anti_fop = fop - hermitian_conjugated(fop)
        anti_tmp = str(jordan_wigner(anti_fop)).replace("]", "")
        anti_list = [
            x.strip().split("[")
            for x in anti_tmp.split("+")
            if not anti_tmp.strip() == ""
        ]
        if len(anti_list) > 1:
            for j in range(len(anti_list)):
                qop = QubitOperator(anti_list[j][1])
                op += qop
    return op


def anti_to_base(op, n):
    """
    反エルミートから基底に使うパウリ演算子
    を取り出す
    """
    id_set = []
    op_list = [x.strip().split("[") for x in str(op).replace("]", "").split("+")]
    if len(op_list[0]) == 1:
        return id_set, 0
    for i in range(len(op_list)):
        id = [0] * n
        op = op_list[i][1].split(" ")
        for j in range(len(op)):
            # if j == len(op)-1:
            #    op[j] = op[j].replace(']', '')
            # for k in range(n):
            #    if str(k) in op[j]:
            #        id[k] = conv_id(op[j])
            for k in range(n):
                if int(op[j][1:]) == k:
                    id[k] = conv_id(op[j])
        id_set.append(id)
        if cf.debug:
            prints("i = ", i, "   op   ", op)
            prints(id)
    size = len(id_set)
    return id_set, size

def make_state1(i, n, active_qubit, index, psi_dash):
    pauli_id = make_pauli_id(i, n, active_qubit)
    circuit = make_gate(n, index, pauli_id)
    state = psi_dash.copy()
    circuit.update_quantum_state(state)
    return state


def calc_inner1(i, j, n, active_qubit, index, psi_dash):
    s_i = make_state1(i, n, active_qubit, index, psi_dash)
    s_j = make_state1(j, n, active_qubit, index, psi_dash)
    s = inner_product(s_j, s_i)
    return s


def uccsd(nspin, det):
    """
    uccsdのfermionic_hamiltonianをつくる
    """

    occ_list, vir_list = get_occvir_lists(2 * nspin, det)
    fermionic_hamiltonian_1 = ucc_singles(occ_list, vir_list)
    fermionic_hamiltonian_2 = ucc_doubles(occ_list, vir_list)
    fermionic_hamiltonian = fermionic_hamiltonian_1 + fermionic_hamiltonian_2
    return fermionic_hamiltonian


def ucc_singles(occ_list, vir_list, ndim2=0):
    """Function
    Prepare a Quantum Circuit for the single exictation part of a Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ia = ndim2
    occ_list_a = [i for i in occ_list if i % 2 == 0]
    occ_list_b = [i for i in occ_list if i % 2 == 1]
    vir_list_a = [i for i in vir_list if i % 2 == 0]
    vir_list_b = [i for i in vir_list if i % 2 == 1]
    ### alpha ###
    fermionic_hamiltonian = FermionOperator()
    for a in vir_list_a:
        for i in occ_list_a:
            fermionic_hamiltonian += FermionOperator(((a, 1), (i, 0)), 1.0)
    ### beta ###
    for a in vir_list_b:
        for i in occ_list_b:
            fermionic_hamiltonian += FermionOperator(((a, 1), (i, 0)), 1.0)
    return fermionic_hamiltonian


def ucc_doubles(occ_list, vir_list, ndim1=0):
    """Function
    Prepare a Quantum Circuit for the double exictation part of a Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ijab = ndim1
    occ_list_a = [i for i in occ_list if i % 2 == 0]
    occ_list_b = [i for i in occ_list if i % 2 == 1]
    vir_list_a = [i for i in vir_list if i % 2 == 0]
    vir_list_b = [i for i in vir_list if i % 2 == 1]
    fermionic_hamiltonian = FermionOperator()
    ### aa -> aa ###
    for [a, b] in itertools.combinations(vir_list_a, 2):
        for [i, j] in itertools.combinations(occ_list_a, 2):
            fermionic_hamiltonian += FermionOperator(
                ((a, 1), (b, 1), (i, 0), (j, 0)), 1.0
            )
    ### ab -> ab ###
    for b in vir_list_b:
        for a in vir_list_a:
            for j in occ_list_b:
                for i in occ_list_a:
                    fermionic_hamiltonian += FermionOperator(
                        ((a, 1), (b, 1), (i, 0), (j, 0)), 1.0
                    )
    ### bb -> bb ###
    for [a, b] in itertools.combinations(vir_list_b, 2):
        for [i, j] in itertools.combinations(occ_list_b, 2):
            fermionic_hamiltonian += FermionOperator(
                ((a, 1), (b, 1), (i, 0), (j, 0)), 1.0
            )
    return fermionic_hamiltonian


def uccgsd(nspin, det):
    """
    uccgsdのfermionic_hamiltonianをつくる
    """

    all_list = [i for i in range(2 * nspin)]
    fermionic_hamiltonian_1 = uccg_singles(all_list)
    fermionic_hamiltonian_2 = uccg_doubles(all_list)
    fermionic_hamiltonian = fermionic_hamiltonian_1 + fermionic_hamiltonian_2
    return fermionic_hamiltonian


def uccg_singles(all_list, ndim2=0):
    """Function
    Prepare a Quantum Circuit for the single exictation part of a Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ia = ndim2
    all_list_a = [i for i in all_list if i % 2 == 0]
    all_list_b = [i for i in all_list if i % 2 == 1]
    l = len(all_list_a)
    ### alpha ###
    fermionic_hamiltonian = FermionOperator()
    for a in range(l):
        for i in range(a):
            fermionic_hamiltonian += FermionOperator(
                ((all_list_a[a], 1), (all_list_a[i], 0)), 1.0
            )
    ### beta ###
    for a in range(l):
        for i in range(a):
            fermionic_hamiltonian += FermionOperator(
                ((all_list_b[a], 1), (all_list_b[i], 0)), 1.0
            )
    return fermionic_hamiltonian


def uccg_doubles(all_list, ndim1=0):
    """Function
    Prepare a Quantum Circuit for the double exictation part of a Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ijab = ndim1
    all_list_a = [i for i in all_list if i % 2 == 0]
    all_list_b = [i for i in all_list if i % 2 == 1]
    fermionic_hamiltonian = FermionOperator()
    l = len(all_list_a)
    """
    [0 2 4 6]
    [0 2] [4 6]
    [0 4] [2 6]
    
    [0 2 4 6 8]
    [0 2] [4 6]/[4 8]/[6 8]
    [0 4] [2 6]/[2 8]/[6 8]
    [0 6] [2 8]/[4 8]
    [2 4] [4 6]/[4 8]/[6 8]
    [2 6] [4 8]
    """
    ### aa -> aa ###
    import copy

    for [a, b] in itertools.combinations(all_list_a, 2):
        all_list_r = copy.copy(all_list_a)
        all_list_r.remove(a)
        all_list_r.remove(b)
        if a < all_list_a[l - 2]:
            for [i, j] in itertools.combinations(all_list_r, 2):
                if a < i and b < j:
                    fermionic_hamiltonian += FermionOperator(
                        ((a, 1), (b, 1), (i, 0), (j, 0)), 1.0
                    )

    ### ab -> ab ###
    for b in range(l):
        for a in range(l):
            for j in range(b):
                for i in range(a):
                    fermionic_hamiltonian += FermionOperator(
                        (
                            (all_list_a[a], 1),
                            (all_list_b[b], 1),
                            (all_list_a[i], 0),
                            (all_list_b[j], 0),
                        ),
                        1.0,
                    )
    ### bb -> bb ###
    for [a, b] in itertools.combinations(all_list_b, 2):
        all_list_r = copy.copy(all_list_b)
        all_list_r.remove(a)
        all_list_r.remove(b)
        if a < all_list_a[l - 2]:
            for [i, j] in itertools.combinations(all_list_r, 2):
                if a < i and b < j:
                    fermionic_hamiltonian += FermionOperator(
                        ((a, 1), (b, 1), (i, 0), (j, 0)), 1.0
                    )

    return fermionic_hamiltonian

def upccgsd(nspin, det):
    """
    upccgsdのfermionic_hamiltonianをつくる
    """

    all_list = [i for i in range(2 * nspin)]
    fermionic_hamiltonian_1 = uccg_singles(all_list)
    fermionic_hamiltonian_2 = upccg_doubles(all_list)
    fermionic_hamiltonian = fermionic_hamiltonian_1 + fermionic_hamiltonian_2
    return fermionic_hamiltonian


def upccg_doubles(all_list, ndim1=0):
    ijab = ndim1
    fermionic_hamiltonian = FermionOperator()
    l = len(all_list)
    ### aa -> aa ###
    import copy

    ### ab -> ab ###
    for a in range(l):
        for i in range(a):
            fermionic_hamiltonian += FermionOperator(
                        (
                            (2*all_list[a]+1, 1),
                            (2*all_list[a],   1),
                            (2*all_list[i]+1, 0),
                            (2*all_list[i],   0),
                        ),
                        1.0,
            )
    return fermionic_hamiltonian



def qite_s_operators(id_set, n):
    """Function
    Given id_set, which contains sigma_i, return the UNIQUE set of sigma_i(dag) * sigma_j.

    Args:
        id_set[k] ([int]): Either 0, 1, 2, 3 to represent I, X, Y, Z at k-th qubit
        n (int): number of qubits
    Returns:
        sigma_list ([Observable]): list of unique qulacs observables
        sigma_ij_index ([int]): tells which unique sigma in sigma_list should be used for sigma_i * sigma_j
        sigma_ij_coef ([[complex]): phase of sigma_i * sigma_j, either 1, -1, 1j, -1j
    """
    size = len(id_set)
    import time

    #T1 = time.time()
    # Old serial version
    #len_list = 0
    #sigma_list = []
    #sigma_ij_index = []
    #sigma_ij_coef = []
    #for i in range(size):
    #    pauli_i = conv_id2XYZ(id_set[i])
    #    sigma_i = QubitOperator(pauli_i)
    #    for j in range(i):
    #        pauli_j = conv_id2XYZ(id_set[j])
    #        sigma_j = QubitOperator(pauli_j)
    #        sigma_ij = sigma_i * sigma_j
    #
    #        coef, pauli_ij = separate_pauli(sigma_ij)

    #        if pauli_ij not in sigma_list:
    #            sigma_list.append(pauli_ij)
    #            sigma_ij_index.append(len_list)
    #            len_list += 1
    #        else:
    #            ind = sigma_list.index(pauli_ij)
    #            sigma_ij_index.append(ind)

    #        sigma_ij_coef.append(coef)

    my_sigma_list = []
    sizeT = size * (size - 1) // 2
    ipos, my_ndim = mpi.myrange(sizeT)
    ij = 0
    for i in range(size):
        pauli_i = conv_id2XYZ(id_set[i])
        sigma_i = QubitOperator(pauli_i)
        for j in range(i):
            if ij in range(ipos, ipos + my_ndim):
                pauli_j = conv_id2XYZ(id_set[j])
                sigma_j = QubitOperator(pauli_j)
                sigma_ij = sigma_i * sigma_j
                coef, pauli_ij = separate_pauli(sigma_ij)
                if pauli_ij not in my_sigma_list:
                    my_sigma_list.append(pauli_ij)
            ij += 1

    #T2 = time.time()
    data = mpi.comm.gather(my_sigma_list, root=0)
    if mpi.rank == 0:
        sigma_list = [x for l in data for x in l]
        sigma_list = list(set(sigma_list))
    else:
        sigma_list = None
    sigma_list = mpi.comm.bcast(sigma_list, root=0)
    len_list = len(sigma_list)

    #T3 = time.time()
    ij = 0
    my_sigma_ij_index = np.zeros(sizeT, dtype=int)
    my_sigma_ij_coef = np.zeros(sizeT, dtype=complex)
    sigma_ij_index = np.zeros(sizeT, dtype=int)
    sigma_ij_coef = np.zeros(sizeT, dtype=complex)
    for i in range(size):
        pauli_i = conv_id2XYZ(id_set[i])
        sigma_i = QubitOperator(pauli_i)
        for j in range(i):
            if ij in range(ipos, ipos + my_ndim):
                pauli_j = conv_id2XYZ(id_set[j])
                sigma_j = QubitOperator(pauli_j)
                sigma_ij = sigma_i * sigma_j
                coef, pauli_ij = separate_pauli(sigma_ij)

                ind = sigma_list.index(pauli_ij)
                my_sigma_ij_index[ij] = ind
                my_sigma_ij_coef[ij] = coef
            ij += 1

    mpi.comm.Allreduce(my_sigma_ij_index, sigma_ij_index, mpi.MPI.SUM)
    mpi.comm.Allreduce(my_sigma_ij_coef, sigma_ij_coef, mpi.MPI.SUM)
    #T4 = time.time()

    # Redefine sigma_list as qulacs.Observable
    for i in range(len_list):
        ope = QubitOperator(sigma_list[i])
        sigma_list[i] = QubitOperator_to_Observable(ope, n)

    #T5 = time.time()
    #prints("T1 -> T2 ", T2 - T1)
    #prints("T2 -> T3 ", T3 - T2)
    #prints("T3 -> T4 ", T4 - T3)
    #prints("T4 -> T5 ", T5 - T4)
    return sigma_list, sigma_ij_index, sigma_ij_coef


def separate_pauli(sigma):
    """Function
    Extract coefficient and pauli word from a single QubitOperator 'sigma'

    Args:
        sigma (QubitOperator):
    Returns:
        coef (complex): coefficient
        pauli (str): pauli word
    """
    tmp = str(sigma).replace("]", "").split("[")
    return complex(tmp[0]), tmp[1]
