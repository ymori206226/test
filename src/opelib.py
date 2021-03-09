"""
#######################
#        quket        #
#######################

opelib.py

Library for operators.

"""
import numpy as np
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from qulacs.state import inner_product
from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator
from openfermion.utils import commutator

from .fileio import error


def create_1body_operator(mo_coeff, XA,
                          XB=None, const=0, ao=False, n_active_orbitals=None):
    """Function
    Given XA (=XB) as a (n_orbitals x n_orbitals) matrix,
    return FermionOperator in OpenFermion Format.
    For active-space calculations, zero electron part (const) may be added.

    If ao is True, XA is ao basis.

    Author(s): Takashi Tsuchimochi
    """
    mo = XA.copy()
    core = const
    if n_active_orbitals is None:
        n_active_orbitals = mo.shape[0]
    if ao:
        ### XA is in AO basis. Transform to MO.
        n_core_orbitals = mo.shape[0] - n_active_orbitals
        mo = mo_coeff.T@mo@mo_coeff
        core = np.sum([2*mo[i, i] for i in range(n_core_orbitals)])
        mo = mo[n_core_orbitals:, n_core_orbitals:]

    ### XA is in MO basis.
    Operator = FermionOperator("", core)
    for i in range(2*n_active_orbitals):
        for j in range(2*n_active_orbitals):
            string = f"{j}^ {i}"
            ii = i//2
            jj = j//2
            if i%2 == 0 and j%2 == 0:  # Alpha-Alpha
                Operator += FermionOperator(string, mo[jj, ii])
            elif i%2 == 1 and j%2 == 1:  # Beta-Beta
                if XB is None:
                    Operator += FermionOperator(string, mo[jj, ii])
                else:
                    error("Currently, UHF basis is not supported "
                          "in create_1body_operator.")
                    #Operator += FermionOperator(string, XB[jj, ii])
    return Operator


def single_operator_gradient(p, q, jw_hamiltonian, state, n_qubits):
    """Function
    Compute gradient d<H>/dXpq

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    dummy = FermionOperator(f"{n_qubits-1}^ {n_qubits-1}", 1.)
    fermi = FermionOperator(f"{p}^ {q}", 1.) + FermionOperator(f"{q}^ {p}", -1.)

    jw_dummy = jordan_wigner(dummy)
    jw_fermi = jordan_wigner(fermi)
    jw_gradient = commutator(jw_fermi, jw_hamiltonian) + jw_dummy

    observable_dummy \
            = create_observable_from_openfermion_text(str(jw_dummy))
    observable_gradient \
            = create_observable_from_openfermion_text(str(jw_gradient))

    gradient = (observable_gradient.get_expectation_value(state) \
                - observable_dummy.get_expectation_value(state))
    return gradient


def FermionOperator_to_Observable(operator, n_qubits):
    """Function
    Create qulacs observable from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    str_jw = str(jordan_wigner(operator))
    string = "(0.0000000000000000+0j) [Z" + str(n_qubits - 1) + "]"
    if str_jw == "0":
        str_jw = string
    else:
        str_jw += " + \n" + string
    return create_observable_from_openfermion_text(str_jw)


def QubitOperator_to_Observable(operator, n_qubits):
    """Function
    Create qulacs observable from OpenFermion QubitOperator `operator`.

    Author(s): Takashi Tsuchimochi
    """
    str_jw = str(operator)
    string = "(0.0000000000000000+0j) [Z" + str(n_qubits - 1) + "]"
    if str_jw == "0":
        str_jw = string
    else:
        str_jw += " + \n" + string
    return create_observable_from_openfermion_text(str_jw)


def FermionOperator_to_Operator(operator, n_qubits):
    """Function
    Create qulacs general operator from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    str_jw = str(jordan_wigner(operator))
    string = "(0.0000000000000000+0j) [Z" + str(n_qubits - 1) + "]"
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
