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


def generate_operators(
    guess,
    geometry,
    basis,
    multiplicity,
    charge=0,
    n_active_electrons=None,
    n_active_orbitals=None,
):
    """Function
    Get fermion operators in OpenFermion format, and store in config.

    Author(s): Takashi Tsuchimochi
    """
    if mpi.main_rank:
        # Run electronic structure calculations
        molecule = run_pyscf_mod(
            guess,
            n_active_orbitals,
            n_active_electrons,
            MolecularData(geometry, basis, multiplicity, charge),
        )
        # Freeze core orbitals and truncate to active space
        if n_active_electrons is None:
            n_core_orbitals = 0
            occupied_indices = None
        else:
            n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
            occupied_indices = list(range(n_core_orbitals))

        if n_active_orbitals is None:
            active_indices = None
        else:
            active_indices = list(
                range(n_core_orbitals, n_core_orbitals + n_active_orbitals)
            )

        cf.Hamiltonian_operator = molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices, active_indices=active_indices
        )
        from openfermion.utils import number_operator, s_squared_operator

        cf.S2_operator = s_squared_operator(n_active_orbitals)
        cf.Number_operator = number_operator(n_active_orbitals)

        # Dipole operators from dipole integrals (AO)
        rx_operator = create_1body_operator(cf.rint[0], ao=True)
        ry_operator = create_1body_operator(cf.rint[1], ao=True)
        rz_operator = create_1body_operator(cf.rint[2], ao=True)
        cf.Dipole_operator = [rx_operator, ry_operator, rz_operator]

    # Broadcasting computed objects in open-fermion and pyscf
    cf.mo_coeff = mpi.comm.bcast(cf.mo_coeff, root=0)
    cf.natom = mpi.comm.bcast(cf.natom, root=0)
    cf.atom_charges = mpi.comm.bcast(cf.atom_charges, root=0)
    cf.atom_coords = mpi.comm.bcast(cf.atom_coords, root=0)
    cf.rint = mpi.comm.bcast(cf.rint, root=0)

    cf.Hamiltonian_operator = mpi.comm.bcast(cf.Hamiltonian_operator, root=0)
    cf.S2_operator = mpi.comm.bcast(cf.S2_operator, root=0)
    cf.Number_operator = mpi.comm.bcast(cf.Number_operator, root=0)
    cf.Dipole_operator = mpi.comm.bcast(cf.Dipole_operator, root=0)


def get_hubbard(hubbard_u, hubbard_nx, hubbard_ny, n_electrons, run_fci=1):
    """Function:
    Generate Hamiltonian for Hubbard.

    Author(s): Takashi Tsuchimochi
    """
    from openfermion.utils import QubitDavidson
    from openfermion.hamiltonians import fermi_hubbard
    from openfermion.transforms import jordan_wigner

    fermionic_hamiltonian = fermi_hubbard(hubbard_nx, hubbard_ny, 1, hubbard_u)
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    fermionic_s2 = s_squared_operator(hubbard_nx * hubbard_ny)
    jw_s2 = jordan_wigner(fermionic_s2)
    if run_fci == 1:
        n_qubit = hubbard_nx * hubbard_ny * 2
        jw_hamiltonian.compress()
        qubit_eigen = QubitDavidson(jw_hamiltonian, n_qubit)
        # Initial guess :  | 0000...00111111>
        #                             ~~~~~~ = n_electrons
        guess = np.zeros((2 ** n_qubit, 1))
        #
        guess[2 ** n_electrons - 1][0] = 1.0
        n_state = 1
        results = qubit_eigen.get_lowest_n(n_state, guess)
        prints("Convergence?           : ", results[0])
        prints("Ground State Energy    : ", results[1][0])
        prints("Wave function          : ")
        openfermion_print_state(results[2], n_qubit, 0)
    return jw_hamiltonian, jw_s2


def create_1body_operator(XA, XB=None, const=0, ao=False):
    """Function
    Given XA (=XB) as a (n_orbitals x n_orbitals) matrix,
    return FermionOperator in OpenFermion Format.
    For active-space calculations, zero electron part (const) may be added.

    If ao is True, XA is ao basis.

    Author(s): Takashi Tsuchimochi
    """
    mo = np.copy(XA)
    core = const
    if ao:
        ### XA is in AO basis. Transform to MO.
        n_core_orbitals = mo.shape[0] - cf.n_active_orbitals
        mo = cf.mo_coeff.T @ mo @ cf.mo_coeff
        core = 0
        for i in range(n_core_orbitals):
            core += 2 * mo[0, 0]
            mo = np.delete(mo, 0, 0)
            mo = np.delete(mo, 0, 1)
    ### XA is in MO basis.
    Operator = FermionOperator("", core)
    for i in range(2 * cf.n_active_orbitals):
        for j in range(2 * cf.n_active_orbitals):
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


def FermionOperator_to_Observable(operator):
    """Function
    Create qulacs observable from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    str_jw = str(jordan_wigner(operator))
    string = "(0.0000000000000000+0j) [Z" + str(2 * cf.n_active_orbitals - 1) + "]"
    if str_jw == "0":
        str_jw = string
    else:
        str_jw += " + \n" + string
    return create_observable_from_openfermion_text(str_jw)


def FermionOperator_to_Operator(operator):
    """Function
    Create qulacs general operator from OpenFermion FermionOperator `operator`.

    Author(s): Masaki Taii, Takashi Tsuchimochi
    """
    str_jw = str(jordan_wigner(operator))
    string = "(0.0000000000000000+0j) [Z" + str(2 * cf.n_active_orbitals - 1) + "]"
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
