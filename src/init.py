"""
#######################
#        quket        #
#######################

init.py

Initializing state.

"""
from typing import Any, List
from dataclasses import dataclass, field, InitVar, make_dataclass

import numpy as np
from qulacs import Observable
from qulacs.state import inner_product
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from openfermion.ops import InteractionOperator, QubitOperator
from openfermion.utils import (number_operator, s_squared_operator, commutator,
                               QubitDavidson)
from openfermion.transforms import jordan_wigner
from openfermion.hamiltonians import fermi_hubbard

from . import mpilib as mpi
from . import config as cf
from .mod import run_pyscf_mod
from .fileio import error, prints, openfermion_print_state, print_geom
from .phflib import weightspin, trapezoidal, simpson
#from .icmrucc import calc_num_ic_theta


@dataclass
class Operators():
    """
    Operator sections.

    Attributes:
        Hamiltonian (InteractionOperator): Hamiltonian operator.
        S2 (InteractionOperator): S2 operator.
        Number (InteractionOperator): Number operator.
        jw_Hamiltonian (QubitOperator): JW transformed hamiltonian.
        jw_S2 (QubitOperator): JW transformed S2.
        jw_Number (QubitOperator): JW transformed number.
        Dipole (list): Dipole moment.
    """
    Hamiltonian: InteractionOperator = None
    S2: InteractionOperator = None
    Number: InteractionOperator = None
    Dipole: np.ndarray = None

    jw_Hamiltonian: QubitOperator = field(init=False, default=None)
    jw_S2: QubitOperator = field(init=False, default=None)
    jw_Number: QubitOperator = field(init=False, default=None)

    def __post_init__(self, *args, **kwds):
        if self.Hamiltonian:
            self.jw_Hamiltonian = jordan_wigner(self.Hamiltonian)
        if self.S2:
            self.jw_S2 = jordan_wigner(self.S2)
        if self.Number:
            self.jw_Number = jordan_wigner(self.Number)


@dataclass
class Qulacs():
    """
    Qulacs section.

    Attributes:
        Hamiltonian (Observable): Quantum hamiltonian.
        S2 (Observable): Quansum S2.
    """
    jw_Hamiltonian: InitVar[QubitOperator] = None
    jw_S2: InitVar[QubitOperator] = None
    jw_Number: InitVar[QubitOperator] = None

    Hamiltonian: Observable = None
    S2: Observable = None
    Number: Observable = None

    def __post_init__(self, jw_Hamiltonian, jw_S2, jw_Number, *args, **kwds):
        if jw_Hamiltonian is not None:
            self.Hamiltonian = create_observable_from_openfermion_text(
                    str(jw_Hamiltonian))
        if jw_S2 is not None:
            self.S2 = create_observable_from_openfermion_text(str(jw_S2))
        if jw_Number is not None:
            self.Number = create_observable_from_openfermion_text(
                    str(jw_Number))


@dataclass
class Projection():
    """
    Symmetry-Projection section.

    Attributes:
        SpinProj (bool): Spin projection.
        NumberProj (bool): Number projection.
        spin (int): Target spin for spin projection.
        Ms (int): Same as multiplicity; multiplicity - 1.
        euler_ngrids (list): Grid points for spin projection.
        number_ngrids (int): Grid points for number projection.
    """
    ansatz: InitVar[str] = None

    Ms: int = None
    spin: int = None
    SpinProj: bool = False
    NumberProj: bool = False
    number_ngrids: int = 0
    euler_ngrids: List[int] = field(default_factory=lambda :[0, -1, 0])

    def __post_init__(self, ansatz, *args, **kwds):
        if ansatz is not None:
            if ansatz in ["phf", "suhf", "sghf", "opt_puccsd", "opt_pucccd"]:
                self.SpinProj = True

    def set_projection(self, trap=True):
        if self.SpinProj:
            prints(f"Projecting to spin space : "
                   f"s = {(self.spin-1)/2:.1f}    "
                   f"Ms = {self.Ms} ")
            prints(f"             Grid points : "
                   f"(alpha, beta, gamma) = ({self.euler_ngrids[0]}, "
                                           f"{self.euler_ngrids[1]}, "
                                           f"{self.euler_ngrids[2]})")

            self.sp_angle = []
            self.sp_weight = []
            # Alpha
            if self.euler_ngrids[0] > 1:
                if trap:
                    alpha, wg_alpha = trapezoidal(0, 2*np.pi,
                                                  self.euler_ngrids[0])
                else:
                    alpha, wg_alpha = simpson(0, 2*np.pi, self.euler_ngrids[0])
            else:
                alpha = [0]
                wg_alpha = [1]
            self.sp_angle.append(alpha)
            self.sp_weight.append(wg_alpha)

            # Beta
            if self.euler_ngrids[1] > 1:
                beta, wg_beta \
                        = np.polynomial.legendre.leggauss(self.euler_ngrids[1])
                beta = np.arccos(beta)
                beta = beta.tolist()
                self.dmm = weightspin(self.euler_ngrids[1], self.spin,
                                      self.Ms, self.Ms, beta)
            else:
                beta = [0]
                wg_beta = [1]
                self.dmm = [1]
            self.sp_angle.append(beta)
            self.sp_weight.append(wg_beta)

            # Gamma
            if self.euler_ngrids[2] > 1:
                if trap:
                    gamma, wg_gamma = trapezoidal(0, 2*np.pi,
                                                  self.euler_ngrids[2])
                else:
                    gamma, wg_gamma = simpson(0, 2*np.pi, self.euler_ngrids[2])
            else:
                gamma = [0]
                wg_gamma = [1]
            self.sp_angle.append(gamma)
            self.sp_weight.append(wg_gamma)

        if self.NumberProj:
            prints(f"Projecting to number space :  "
                   f"N = {self.number_ngrids}")

            self.np_angle = []
            self.np_weight = []

            # phi
            if self.number_ngrids > 1:
                if trap:
                    phi, wg_phi = trapezoidal(0, 2*np.pi, self.number_ngrids)
                else:
                    gamma, wg_gamma = simpson(0, 2*np.pi, self.number_ngrids)
            else:
                phi = [0]
                wg_phi = [1]
            self.np_angle = phi
            self.np_weight = wg_phi


@dataclass
class Multi():
    """
    Multi/Excited-State calculation section.

    Attributes:
        act2act_opt (bool): ??
        states (list): Initial determinants (bits)
                       for multi-state calculations; JM-UCC or ic-MRUCC.
        weights (list): Weight for state-average calculations;
                        usually 1 for all.
    """
    act2act_opt: bool = False

    states: List = field(default_factory=list)
    weights: List = field(default_factory=list)

    nstates: int = field(init=False)

    def __post_init__(self, *args, **kwds):
        self.nstates = len(self.weights) if self.weights else 0


@dataclass
class QuketData():
    """Data class for Quket.

    Attributes:
        method (str): Computation method; 'vqe' or 'qite'.
        model (str): Computation model; 'chemical', 'hubbard' or 'heisenberg'.
        ansatz (str): VQE or QITE ansatz; 'uccsd' and so on.
        det (int): A decimal value of the determinant of the quantum state.
        run_fci (bool): Whether run FCI or not.
        rho (int): Trotter number for related ansatz.
        DS (int): If 0/1, the operation order is;
                  Exp[T1] Exp[T2] / Exp[T2] Exp[T1]
        Do1RDM (bool): Whether do 1RDM or not.
        print_amp_thres (float): Threshold for printing VQE parameters.
        ftol (float): Convergence criterion based on energy (cost).
        gtol (float): Convergence criterion based on gradients.
        dt (flaot): Time-step for time-evolution.
        truncate (float): Truncation threshold for anti-symmetric hamiltonian.
        excited_states (list): Initial determinants
                               for excited state calculations.

    Author(s): Takashi Tsuchimochi, Yuma Shimomoto
    """
    #----------For QuketData----------
    method: str = "vqe"
    model: str = None
    ansatz: str = None
    det: int = None
    run_fci: bool = True
    rho: int = 1
    DS: int = 0
    Do1RDM: bool = False
    print_amp_thres: float = 1e-2
    ftol: float = 1e-9
    gtol: float = 1e-5
    dt: float = 0.
    truncate: float = 0.
    excited_states: List = field(default_factory=list)
    maxiter: int = 0

    operators: Operators = field(init=False, default=None)
    qulacs: Qulacs = field(init=False, default=None)
    projection: Projection = field(init=False, default=None)
    multi: Multi = field(init=False, default=None)

    def __post_init__(self, *args, **kwds):
        # Set some initialization.
        pass

    def initialize(self, *, pyscf_guess="minao", **kwds):
        """Function
        Run PySCF and initialize parameters.

        Args:
            pyscf_guess (str): PySCF guess.
        """
        ##################
        # Set Subclasses #
        ##################
        # Projection
        init_dict = get_func_kwds(Projection.__init__, kwds)
        self.projection = Projection(**init_dict)
        # Multi
        init_dict = get_func_kwds(Multi.__init__, kwds)
        self.multi = Multi(**init_dict)

        #############
        # Set model #
        #############
        if kwds["basis"] == "hubbard":
            self.model = "hubbard"
        elif "heisenberg" in kwds["basis"]:
            self.model = "heisenberg"
        else:
            self.model = "chemical"

        #######################
        # Create parent class #
        #######################
        if self.model == "hubbard":
            from .hubbard import Hubbard

            init_dict = get_func_kwds(Hubbard.__init__, kwds)
            obj = Hubbard(**init_dict)
        elif self.model == "heisenberg":
            from .heisenberg import Heisenberg

            init_dict = get_func_kwds(Heisenberg.__init__, kwds)
            obj = Heisenberg(**init_dict)
            if self.det is None:
                self.det = 1
            self.current_det = self.det
        elif self.model == "chemical":
            from .quket_molecule import QuketMolecule

            init_dict = get_func_kwds(QuketMolecule.__init__, kwds)
            obj = QuketMolecule(**init_dict)
# 全部の軌道と電子を使う？
            obj, pyscf_mol = run_pyscf_mod(pyscf_guess, obj.n_orbitals,
                                           obj.n_electrons, obj,
                                           run_casci=self.run_fci)

            if "n_electrons" in kwds:
                obj.n_active_electrons = kwds["n_electrons"]
            else:
                obj.n_active_electrons = obj.n_electrons
            if "n_orbitals" in kwds:
                obj.n_active_orbitals = kwds["n_orbitals"]
            else:
                obj.n_active_orbitals = obj.n_orbitals

        #######################################
        # Inherit parent class                #
        #   MAGIC: DYNAMIC CLASS INHERITANCE. #
        #######################################
        # Add attributes to myself
        for k, v in obj.__dict__.items():
            if k not in self.__dict__:
                self.__dict__[k] = v
        # Keep them under the control of dataclass and rename my class name.
        my_fields = []
        for k, v in self.__dict__.items():
            if isinstance(v, (dict, list, set)):
                my_fields.append((k, type(v), field(default_factory=v)))
            else:
                my_fields.append((k, type(v), v))
        self.__class__ = make_dataclass(f"{obj.__class__.__name__}QuketData",
                                        my_fields,
                                        bases=(QuketData, obj.__class__))
        # Add funcgtions and properties to myself.
        for k in dir(obj):
            if k not in dir(self):
                setattr(self, k, getattr(mol, k))

        #################
        # Get Operators #
        #################
        if self.model == "hubbard":
            # self.jw_Hamiltonian, self.jw_S2 = get_hubbard(
            #    hubbard_u,
            #    hubbard_nx,
            #    hubbard_ny,
            #    n_electrons,
            #    run_fci,
            # )
            Hamiltonian, S2, Number = obj.get_operators(guess=pyscf_guess)
            self.operators = Operators(Hamiltonian=Hamiltonian, S2=S2,
                                       Number=Number)
        elif self.model == "heisenberg":
            jw_Hamiltonian = obj.get_operators()
            self.operators = Operators()
            self.operators.jw_Hamiltonian = jw_Hamiltonian
            return
        elif self.model == "chemical":
            if cf._geom_update:
                # New geometry found. Run PySCF and get operators.
                Hamiltonian, S2, Number, Dipole \
                        = obj.get_operators(guess=pyscf_guess,
                                             pyscf_mol=pyscf_mol)
                cf._geom_update = False

                self.operators = Operators(Hamiltonian=Hamiltonian, S2=S2,
                                           Number=Number, Dipole=Dipole)

        # Initializing parameters
        self.n_qubits = self.n_orbitals * 2
#hubbardにmultiplicityないのでは？
        self.projection.Ms = self.multiplicity - 1
        self.n_qubits_anc = self.n_qubits + 1
        self.anc = self.n_qubits
        self.state = None

        # Check spin, multiplicity, and Ms
        if self.projection.spin is None:
            self.projection.spin = self.multiplicity  # Default
        if (self.projection.spin-self.multiplicity)%2 != 0 \
                or self.projection.spin < self.multiplicity:
            prints(f"Spin = {self.projection.spin}    "
                   f"Ms = {self.projection.Ms}")
            error("Spin and Ms not cosistent.")
        if (self.n_electrons+self.multiplicity-1)%2 != 0:
            prints(f"Incorrect specification for "
                   f"n_electrons = {self.n_electrons} "
                   f"and multiplicity = {self.multiplicity}")

        # NOA; Number of Occupied orbitals of Alpha.
        self.noa = (self.n_electrons+self.multiplicity-1)//2
        # NOB; Number of Occupied orbitals of Beta.
        self.nob = self.n_electrons - self.noa
        # NVA; Number of Virtual orbitals of Alpha.
        self.nva = self.n_orbitals - self.noa
        # NVB; Number of Virtual orbitals of Beta.
        self.nvb = self.n_orbitals - self.nob

        # Check initial determinant
        if self.det is None:
            # Initial determinant is RHF or ROHF
            self.det = set_initial_det(self.noa, self.nob)
        self.current_det = self.det

        # Excited states (orthogonally-constraint)
        self.nexcited = len(self.excited_states) if self.excited_states else 0
        self.lower_states = []

    def jw_to_qulacs(self):
        self.qulacs = Qulacs(jw_Hamiltonian=self.operators.jw_Hamiltonian,
                             jw_S2=self.operators.jw_S2,
                             jw_Number=self.operators.jw_Number)

    def set_projection(self, euler_ngrids=None, number_ngrids=None, trap=True):
        """Function
        Set the angles and weights for integration of
        spin-projection and number-projection.

        Spin-rotation operator;
                Exp[ -i alpha Sz ]   Exp[ -i beta Sy ]   Exp[ -i gamma Sz ]
        weight
                Exp[ i m' alpha]      d*[j,m',m]         Exp[ i m gamma]

        Angles alpha and gamma are determined by Trapezoidal quadrature,
        and beta is determined by Gauss-Legendre quadrature.

        Number-rotation operator
                Exp[ -i phi N ]
        weight
                Exp[  i phi Ne ]

        phi are determined by Trapezoidal quadrature.
        """
        if self.model =="heisenberg":
            # No spin or number symmetry in the model
            return

        if euler_ngrids is not None:
            self.projection.euler_ngrids = euler_ngrids
        if number_ngrids is not None:
            self.projection.number_ngrids = number_ngrids

        self.projection.set_projection(trap=trap)

#    def get_ic_ndim(self):
#        core_num = self.n_qubits
#        vir_index = 0
#        for istate in range(self.multi.nstates):
#            ### Read state integer and extract occupied/virtual info
#            occ_list_tmp = int2occ(self.multi.states[istate])
#            vir_tmp = occ_list_tmp[-1] + 1
#            for ii in range(len(occ_list_tmp)):
#                if ii == occ_list_tmp[ii]:
#                    core_tmp = ii + 1
#            vir_index = max(vir_index, vir_tmp)
#            core_num = min(core_num, core_tmp)
#        vir_num = self.n_qubits - vir_index
#        act_num = self.n_qubits - core_num - vir_num
#        self.multi.core_num = core_num
#        self.multi.act_num = act_num
#        self.multi.vir_num = vir_num
#        ndim1, ndim2 = calc_num_ic_theta(n_qubit_system, vir_num,
#                                         act_num, core_num)

    def print(self):
        if mpi.main_rank:
            formatstr = f"0{self.n_qubits}b"
            max_len = max(map(len, list(self.__dict__.keys())))
            for k, v in self.__dict__.items():
                if callable(v):
                    continue
                if k == "det":
                    print(f"{k.ljust(max_len)} : {format(v, formatstr)}")
                else:
                    print(f"{k.ljust(max_len)} : {v}")


#def set_initial_det(noa, nob):
#    """Function
#    Set the initial wave function to RHF/ROHF determinant.
#
#    Author(s): Takashi Tsuchimochi
#    """
#    det = 0
#    for i in range(noa):
#        det = det^(1 << 2*i)
#    for i in range(nob):
#        det = det^(1 << 2*i + 1)
#    return det


def set_initial_det(noa, nob):
    """ Function
    Set the initial wave function to RHF/ROHF determinant.

    Author(s): Takashi Tsuchimochi
    """
    # Note: r'~~' means that it is a regular expression.
    # a: Number of Alpha spin electrons
    # b: Number of Beta spin electrons
    if noa >= nob:
        # Here calculate 'a_ab' as follow;
        # |r'(01){a-b}(11){b}'> wrote by regular expression.
        # e.g.)
        #   a=1, b=1: |11> = |3>
        #   a=3, b=1: |0101 11> = |23> = |3 + 5/2*2^3>
        # r'(01){a-b}' = (1 + 4 + 16 + ... + 4^(a-b-1))/2
        #              = (4^(a-b) - 1)/3
        # That is, it is the sum of the first term '1'
        # and the geometric progression of the common ratio '4'
        # up to the 'a-b' term.
        base = nob*2
        a_ab = (4**(noa-nob) - 1)//3
    elif noa < nob:
        # Here calculate 'a_ab' as follow;
        # |r'(10){b-a}(11){a}'> wrote by regular expression.
        # e.g.)
        #   a=1, b=1: |11> = |3>
        #   a=1, b=3: |1010 11> = |43> = |3 + 5*2^3>
        # r'(10){b-a}' = 2 + 8 + 32 + ... + 2*4^(b-a-1)
        #              = 2 * (4^(b-a) - 1)/3
        # That is, it is the sum of the first term '2'
        # and the geometric progression of the common ratio '4'
        # up to the 'a-b' term.
        base = noa*2
        a_ab = 2*(4**(nob-noa) - 1)//3
    return 2**base-1 + (a_ab<<base)


#def int2occ(state_int):
#    """Function
#    Given an (base-10) integer, find the index for 1 in base-2 (occ_list)
#
#    Author(s): Takashi Tsuchimochi
#    """
#    occ_list = []
#    k = 0
#    while k < state_int:
#        kk = 1 << k
#        if kk & state_int > 0:
#            occ_list.append(k)
#        k += 1
#    return occ_list


def int2occ(state_int):
    """ Function
    Given an (base-10) integer, find the index for 1 in base-2 (occ_list)

    Author(s): Takashi Tsuchimochi
    """
    # Note: bin(23) = '0b010111'
    #       bin(23)[-1:1:-1] = '111010'
    # Same as; bin(23)[2:][::-1]
    # Occupied orbitals denotes '1'.
    occ_list = [i for i, k in enumerate(bin(state_int)[-1:1:-1]) if k == "1"]
    return occ_list


def get_occvir_lists(n_qubits, det):
    """Function
    Generate occlist and virlist for det (base-10 integer).

    Author(s): Takashi Tsuchimochi
    """
    occ_list = int2occ(det)
    vir_list = [i for i in range(n_qubits) if i not in occ_list]
    return occ_list, vir_list


def get_func_kwds(func, kwds):
    import inspect

    sig = inspect.signature(func).parameters
    init_dict = {s: kwds[s] for s in sig if s in kwds}
    return init_dict
