"""
#######################
#        quket        #
#######################

opelib.py

Library for operators.

"""


from . import mpilib as mpi
from . import config as cf
import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermion.utils import s_squared_operator, commutator
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from qulacs.state import inner_product


from .fileio import error, prints, openfermion_print_state
from .mod import run_pyscf_mod


def create_1body_operator(mo_coeff, XA, XB=None, const=0, ao=False, n_active_orbitals=None):
    """Function
    Given XA (=XB) as a (n_orbitals x n_orbitals) matrix,
    return FermionOperator in OpenFermion Format.
    For active-space calculations, zero electron part (const) may be added.

    If ao is True, XA is ao basis.

    Author(s): Takashi Tsuchimochi
    """
    mo = np.copy(XA)
    core = const
    if n_active_orbitals == None:
        n_active_orbitals = mo.shape[0]
    if ao:
        ### XA is in AO basis. Transform to MO.
        n_core_orbitals = mo.shape[0] - n_active_orbitals
        mo = mo_coeff.T @ mo @ mo_coeff
        core = 0
        for i in range(n_core_orbitals):
            core += 2 * mo[0, 0]
            mo = np.delete(mo, 0, 0)
            mo = np.delete(mo, 0, 1)
    ### XA is in MO basis.
    Operator = FermionOperator("", core)
    for i in range(2 * n_active_orbitals):
        for j in range(2 * n_active_orbitals):
            string = str(j) + "^ " + str(i)
            ii = int(i / 2)
            jj = int(j / 2)
            if i % 2 == 0 and j % 2 == 0:  # Alpha-Alpha
                Operator += FermionOperator(string, mo[jj][ii])
            elif i % 2 == 1 and j % 2 == 1:  # Beta-Beta
                if XB is None:
                    Operator += FermionOperator(string, mo[jj][ii])
                else:
                    error(
                        "Currently, UHF basis is not supported in create_1body_operator."
                    )
                    # Operator += FermionOperator(string,XB[jj][ii])
    return Operator


def single_operator_gradient(p, q, jordan_wigner_hamiltonian, state, n_qubit):
    """Function
    Compute gradient d<H>/dXpq

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    # 与えられたpqからフェルミ演算子a_p!q-a_q!pを生成する
    # ダミーを作って後で引く
    dummy = FermionOperator(str(n_qubit - 1) + "^ " + str(n_qubit - 1), 1.0)
    fermi = FermionOperator(str(p) + "^ " + str(q), 1.0) + FermionOperator(
        str(q) + "^ " + str(p), -1.0
    )
    # フェルミ演算子をjordan_wigner変換する
    jordan_wigner_fermi = jordan_wigner(fermi)
    jordan_wigner_dummy = jordan_wigner(dummy)
    # 交換子を用いてエネルギーの傾きを求める準備を行う
    jordan_wigner_gradient = (
        commutator(jordan_wigner_fermi, jordan_wigner_hamiltonian) + jordan_wigner_dummy
    )
    # オブザーバブルクラスに変換
    observable_gradient = create_observable_from_openfermion_text(
        str(jordan_wigner_gradient)
    )
    observable_dummy = create_observable_from_openfermion_text(str(jordan_wigner_dummy))
    # オブザーバブルを用いてエネルギーの傾きを求める
    gradient = observable_gradient.get_expectation_value(
        state
    ) - observable_dummy.get_expectation_value(state)

    return gradient


def FermionOperator_to_Observable(operator, n_qubit):
    """Function
    Create qulacs observable from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    str_jw = str(jordan_wigner(operator))
    string = "(0.0000000000000000+0j) [Z" + str(n_qubit - 1) + "]"
    if str_jw == "0":
        str_jw = string
    else:
        str_jw += " + \n" + string
    return create_observable_from_openfermion_text(str_jw)

def QubitOperator_to_Observable(operator, n_qubit):
    """Function
    Create qulacs observable from OpenFermion QubitOperator `operator`.

    Author(s): Takashi Tsuchimochi
    """
    str_jw = str(operator)
    string = "(0.0000000000000000+0j) [Z" + str(n_qubit - 1) + "]"
    if str_jw == "0":
        str_jw = string
    else:
        str_jw += " + \n" + string
    return create_observable_from_openfermion_text(str_jw)


def FermionOperator_to_Operator(operator, n_qubit):
    """Function
    Create qulacs general operator from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    str_jw = str(jordan_wigner(operator))
    string = "(0.0000000000000000+0j) [Z" + str(n_qubit - 1) + "]"
    if str_jw == "0":
        str_jw = string
    else:
        str_jw += " + \n" + string
    return create_quantum_operator_from_openfermion_text(str_jw)


def Orthonormalize(state0, state1, normalize=True):
    """Function
    Project out state 0 from state 1
    |1>  <= (1 - |0><0| ) |1>

    |1> is renormalized.

    Author(s): Takashi Tsuchimochi
    """
    S01 = inner_product(state0, state1)

    tmp = state0.copy()
    tmp.multiply_coef(-S01)
    state1.add_state(tmp)
    if normalize:
        # Normalize
        norm2 = state1.get_squared_norm()
        state1.normalize(norm2)
