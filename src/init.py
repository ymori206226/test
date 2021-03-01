"""
#######################
#        quket        #
#######################

init.py

Initializing state.

"""
from typing import Any, List
from dataclasses import dataclass, field, InitVar

import numpy as np
from qulacs import Observable
from qulacs.state import inner_product
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from openfermion.ops import InteractionOperator, QubitOperator
from openfermion.utils import number_operator, s_squared_operator, commutator
from openfermion.transforms import jordan_wigner
from openfermion.hamiltonians import MolecularData

from . import mpilib as mpi
from . import config as cf
from .mod import run_pyscf_mod
from .fileio import error, prints, openfermion_print_state, print_geom
from .opelib import create_1body_operator
from .phflib import weightspin, trapezoidal, simpson
from .icmrucc import calc_num_ic_theta


@dataclass
class QuketData(MolecularData):
    __doc__ = MolecularData.__doc__ + \
    """-----------------------------
    Molecular data for Quket.

    Attributes:
        method (str): Computation method; 'vqe' or 'qite'.
        model (str): Computation model; 'chemical', 'hubbard' or 'heisenberg'.
        ansatz (str): VQE or QITE ansatz; 'uccsd' and so on.
        det (int): A decimal value of the determinant of the quantum state.
        hubbard_u (??): ??
        hubbard_nx (int): Number of hubbard sites for x-axis.
        hubbard_ny (int): Number of hubbard sites for y-axis.
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
    #----------For MolecularData----------
    geometry: List = None
    basis: str = None
    multiplicity: int = None
    charge: int = 0
    description: str = ""
    filename: str = ""
    data_directory: str = None
    #----------For QuketData----------
    method: str = "vqe"
    model: str = None
    ansatz: str = None
    det: int = None
    hubbard_u: Any = None   # ??
    hubbard_nx: int = -1
    hubbard_ny: int = None
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

    operators: self.Operators = None
    qulacs: self.Qulacs = None
    projection: self.Projection = None
    multi: self.Multi = None

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
        jw_Hamiltonian: QubitOperator = None
        jw_S2: QubitOperator = None
        jw_Number: QubitOperator = None
        Dipole: List[float] = field(default_factory=list)

    @dataclass
    class Qulacs():
        """
        Qulacs section.

        Attributes:
            Hamiltonian (Observable): Quantum hamiltonian.
            S2 (Observable): Quansum S2.
        """
        Hamiltonian: Observable = None
        S2: Observable = None

    @dataclass
    class Projection():
        """
        Projection section.

        Attributes:
            SpinProj (bool): Spin projection.
            NumberProj (bool): Number projection.
            spin (int): Target spin for spin projection.
            Ms (int): Same as multiplicity; multiplicity - 1.
            euler_ngrids (list): Grid points for spin projection.
            number_ngrids (int): Grid points for number projection.
        """
        SpinProj: bool = False
        NumberProj: bool = False
        spin: int = None
        Ms: int = None
        euler_ngrids: List[int] = [0, -1, 0]
        number_ngrids: int = 0

    @dataclass
    class Multi():
        """
        Multi section.

        Attributes:
            states (list): Initial determinants (bits)
                           for multi-state calculations; JM-UCC or ic-MRUCC.
            weights (list): Weight for state-average calculations;
                            usually 1 for all.
        """
        states: List = field(default=list)
        weights: List = field(default=list)

    def __post_init__(self, *args, **kwds):
        # Set each variables of MolecularData.
        super().__init__(geometry=self.geometry, basis=self.basis,
                         multiplicity=self.multiplicity, charge=self.charge,
                         description=self.description, filename=self.filename,
                         data_directory=self.data_directory)
        self.operators = self.Operators()
        self.qulacs = self.Qulacs()
        self.projection = self.Projection()
        self.multi = self.Multi()

    def initialize(self, pyscf_guess="minao"):
        """Function
        Run PySCF and initialize parameters.
        """
        if self.basis == "hubbard":
            self.model = "hubbard"
        elif "heisenberg" in self.basis:
            self.model = "heisenberg"
        else:
            self.model = "chemical"
            if self.basis is None or self.geometry is None:
                raise ValueError("Basis and geometry have to be specified"
                                 "for chemical Hamiltonian")
        if self.n_electrons is None and self.model in ("hubbard", "chemical"):
            raise ValueError("No electron number")

        if self.model == "hubbard":
            if self.hubbard_u is None or self.hubbard_nx is None:
                raise ValueError("For hubbard, hubbard_u and hubbard_nx"
                                 "have to be given")
            self.n_orbitals = self.hubbard_nx * self.hubbard_ny
            # self.jw_Hamiltonian, self.jw_S2 = get_hubbard(
            #    hubbard_u,
            #    hubbard_nx,
            #    hubbard_ny,
            #    n_electrons,
            #    run_fci,
            # )
            self.get_operators()
            self.operators.jw_Hamiltonian \
                    = jordan_wigner(self.operators.Hamiltonian)
            self.operators.jw_S2 = jordan_wigner(self.operators.S2)
            self.operators.jw_Number = jordan_wigner(self.operators.Number)
            self.operators.Dipole = None

            # Initializing parameters
            prints(f"Hubbard model: nx = {self.hubbard_nx}"
                   f"ny = {self.hubbard_ny}"
                   f"U = {self.hubbard_u:2.2f}")
        elif self.model == "heisenberg":
            nspin = self.n_orbitals
            sx = [QubitOperator(f"X{i}") for i in range(nspin)]
            sy = [QubitOperator(f"Y{i}") for i in range(nspin)]
            sz = [QubitOperator(f"Z{i}") for i in range(nspin)]
            jw_Hamiltonian = 0 * QubitOperator("")
            if "lr" in self.basis:
                for i in range(nspin):
                    j = (i+1)%nspin
                    jw_Hamiltonian += 0.5*(sx[i]*sx[j]
                                           + sy[i]*sy[j]
                                           + sz[i]*sz[j])
                for i in range(2):
                    j = i+2
                    jw_Hamiltonian += 0.333333333333*(sx[i]*sx[j]
                                                      + sy[i]*sy[j]
                                                      + sz[i]*sz[j])
            else:
                for i in range(nspin):
                    j = (i+1)%nspin
                    jw_Hamiltonian += sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j]
            self.operators.jw_Hamiltonian = jw_Hamiltonian

            self.n_qubits = self.n_orbitals
            if self.det is None:
                self.det = 1
            self.current_det = self.det

            return
        elif self.model == "chemical":
            if cf._geom_update:
                # New geometry found. Run PySCF and get operators.
                self.get_operators(guess=pyscf_guess)
                # Set Jordan-Wigner Hamiltonian and S2 operators using PySCF and Open-Fermion
                self.operators.jw_Hamiltonian \
                        = jordan_wigner(self.operators.Hamiltonian)
                self.operators.jw_S2 = jordan_wigner(self.operators.S2)
                self.operators.jw_Number = jordan_wigner(self.operators.Number)
                cf._geom_update = False

        # Initializing parameters
        self.n_qubits = self.n_orbitals * 2
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
        # Number of occupied orbitals of alpha
        self.noa = (self.n_electrons+self.multiplicity-1)//2
        # Number of occupied orbitals of beta
        self.nob = self.n_electrons - self.noa
        # Number of virtual orbitals of alpha
        self.nva = self.n_orbitals - self.noa
        # Number of virtual orbitals of beta
        self.nvb = self.n_orbitals - self.nob

        # Check initial determinant
        if self.det is None:
            # Initial determinant is RHF or ROHF
            self.det = set_initial_det(self.noa, self.nob)
        self.current_det = self.det
        if self.ansatz in ("phf", "suhf", "sghf", "opt_puccsd", "opt_puccd"):
            self.projection.SpinProj = True

        # Excited states (orthogonally-constraint)
        self.nexcited = len(self.excited_states)
        self.lower_states = []

        # Multi states
        self.multi.nstates = len(self.multi.weights)

    def get_operators(self, guess="minao"):
        if self.model == "chemical":
            if mpi.main_rank:
                # Run electronic structure calculations
                molecule, pyscf_molecule \
                        = run_pyscf_mod(guess, self.n_orbitals,
                                        self.n_electrons, self,
                                        run_fci=self.run_fci)
                # Freeze core orbitals and truncate to active space
                if self.n_electrons is None:
                    n_core_orbitals = 0
                    occupied_indices = None
                else:
                    n_core_orbitals = (molecule.n_electrons-self.n_electrons)//2
                    occupied_indices = list(range(n_core_orbitals))

                if self.n_orbitals is None:
                    active_indices = None
                else:
                    active_indices = list(
                            range(n_core_orbitals,
                                  n_core_orbitals+self.n_orbitals))

                hf_energy = molecule.hf_energy
                fci_energy = molecule.fci_energy

                mo_coeff = molecule.canonical_orbitals.astype(float)
                natom = pyscf_molecule.natm
                atom_charges = pyscf_molecule.atom_charges().reshape(-1, 1)
                atom_coords = pyscf_molecule.atom_coords()
                rint = pyscf_molecule.intor("int1e_r")

                Hamiltonian = molecule.get_molecular_hamiltonian(
                    occupied_indices=occupied_indices,
                    active_indices=active_indices)

                # Dipole operators from dipole integrals (AO)
                rx = create_1body_operator(mo_coeff, rint[0], ao=True,
                                           n_active_orbitals=self.n_orbitals)
                ry = create_1body_operator(mo_coeff, rint[1], ao=True,
                                           n_active_orbitals=self.n_orbitals)
                rz = create_1body_operator(mo_coeff, rint[2], ao=True,
                                           n_active_orbitals=self.n_orbitals)
                Dipole = np.array([rx, ry, rz])
            else:
                Hamiltonian = None
                Dipole = None
                hf_energy = None
                fci_energy = None
                mo_coeff = None
                natom = None
                atom_charges = None
                atom_coords = None

            # MPI broadcasting
            Hamiltonian = mpi.comm.bcast(Hamiltonian, root=0)
            Dipole = mpi.comm.bcast(Dipole, root=0)
            hf_energy = mpi.comm.bcast(hf_energy, root=0)
            fci_energy = mpi.comm.bcast(fci_energy, root=0)
            mo_coeff = mpi.comm.bcast(mo_coeff, root=0)
            natom = mpi.comm.bcast(natom, root=0)
            atom_charges = mpi.comm.bcast(atom_charges, root=0)
            atom_coords = mpi.comm.bcast(atom_coords, root=0)

            # Put values in self
            self.operators.Hamiltonian = Hamiltonian
            self.operators.Dipole = Dipole
            self.hf_energy = hf_energy
            self.fci_energy = fci_energy
            self.mo_coeff = mo_coeff
            self.natom = natom
            self.atom_charges = atom_charges
            self.atom_coords = atom_coords

            # Print out some results
            print_geom(self.geometry)
            prints("E[FCI] = ", fci_energy)
            prints("E[HF]  = ", hf_energy)
            prints("")
        elif self.model == "hubbard":
            from openfermion.utils import QubitDavidson
            from openfermion.transforms import jordan_wigner
            from openfermion.hamiltonians import fermi_hubbard

            self.operators.Hamiltonian = fermi_hubbard(self.hubbard_nx,
                                                       self.hubbard_ny,
                                                       1, self.hubbard_u)
            self.operators.jw_Hamiltonian \
                    = jordan_wigner(self.operators.Hamiltonian)

            self.hf_energy = None
            self.fci_energy = None

            if self.run_fci:
                n_qubits = self.hubbard_nx * self.hubbard_ny * 2
                self.operators.jw_Hamiltonian.compress()
                qubit_eigen = QubitDavidson(self.operators.jw_Hamiltonian,
                                            n_qubits)
                # Initial guess :  | 0000...00111111>
                #                             ~~~~~~ = n_electrons
                guess = np.zeros((2**n_qubits, 1))
                guess[2**self.n_electrons - 1][0] = 1.0
                n_state = 1
                results = qubit_eigen.get_lowest_n(n_state, guess)
                prints("Convergence?           : ", results[0])
                prints("Ground State Energy    : ", results[1][0])
                self.fci_energy = results[1][0]
                #prints("Wave function          : ")
                #openfermion_print_state(results[2], n_qubits, 0)

            self.mo_coeff = None
            self.natom = self.hubbard_nx * self.hubbard_ny
            self.atom_charges = None
            self.atom_coords = None

        self.operators.S2 = s_squared_operator(self.n_orbitals)
        self.operators.Number = number_operator(self.n_orbitals)

    def jw_to_qulacs(self):
        if self.operators.jw_Hamiltonian is not None:
            self.qulacs.Hamiltonian = create_observable_from_openfermion_text(
                    str(self.operators.jw_Hamiltonian))
        else:
            self.qulacs.Hamiltonian = None

        if self.operators.jw_S2 is not None:
            self.qulacs.S2 = create_observable_from_openfermion_text(
                    str(self.operators.jw_S2))
        else:
            self.qulacs.S2 = None

        if self.operators.jw_Number is not None:
            self.qulacs.Number = create_observable_from_openfermion_text(
                    str(self.operators.jw_Number))
        else:
            self.qulacs.Number = None

    def set_projection(self, euler_ngrids=None, number_ngrids=None):
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

        trap = True
        if euler_ngrids is not None:
            self.projection.euler_ngrids = euler_ngrids
        if number_ngrids is not None:
            self.projection.number_ngrids = number_ngrids
        if self.projection.SpinProj:
            prints(f"Projecting to spin space : "
                   f"s = {(self.projection.spin-1)/2:.1f}    "
                   f"Ms = {self.projection.Ms} ")
            prints(f"             Grid points :  "
                   f"(alpha,beta,gamma) = ({self.projection.euler_ngrids[0]}, "
                                         f"{self.projection.euler_ngrids[1]}, "
                                         f"{self.projection.euler_ngrids[2]})")
            self.projection.sp_angle = []
            self.projection.sp_weight = []

            # Alpha
            if self.projection.euler_ngrids[0] > 1:
                if trap:
                    alpha, wg_alpha \
                            = trapezoidal(0, 2*np.pi,
                                          self.projection.euler_ngrids[0])
                else:
                    alpha, wg_alpha \
                            = simpson(0, 2*np.pi,
                                      self.projection.euler_ngrids[0])
            else:
                alpha = [0]
                wg_alpha = [1]
            self.projection.sp_angle.append(alpha)
            self.projection.sp_weight.append(wg_alpha)

            # Beta
            if self.projection.euler_ngrids[1] > 1:
                beta, wg_beta = np.polynomial.legendre.leggauss(
                        self.projection.euler_ngrids[1])
                beta = np.arccos(beta)
                beta = beta.tolist()
                self.projection.dmm = weightspin(
                        self.projection.euler_ngrids[1],
                        self.projection.spin,
                        self.projection.Ms,
                        self.projection.Ms,
                        beta)
            else:
                beta = [0]
                wg_beta = [1]
                self.projection.dmm = [1]
            self.projection.sp_angle.append(beta)
            self.projection.sp_weight.append(wg_beta)

            # Gamma
            if self.projection.euler_ngrids[2] > 1:
                if trap:
                    gamma, wg_gamma \
                            = trapezoidal(0, 2*np.pi,
                                          self.projection.euler_ngrids[2])
                else:
                    gamma, wg_gamma \
                            = simpson(0, 2*np.pi,
                                      self.projection.euler_ngrids[2])
            else:
                gamma = [0]
                wg_gamma = [1]
            self.projection.sp_angle.append(gamma)
            self.projection.sp_weight.append(wg_gamma)

        if self.projection.NumberProj:
            self.projection.number_ngrids = number_ngrids
            prints(f"Projecting to number space :  "
                   f"N = {self.projection.number_ngrids}")

            self.projection.np_angle = []
            self.projection.np_weight = []

            # phi
            if self.projection.number_ngrids > 1:
                if trap:
                    phi, wg_phi \
                            = trapezoidal(0, 2*np.pi,
                                          self.projection.number_ngrids)
                else:
                    gamma, wg_gamma \
                            = simpson(0, 2*np.pi,
                                      self.projection.number_ngrids)
            else:
                phi = [0]
                wg_phi = [1]
            self.projection.np_angle = phi
            self.projection.np_weight = wg_phi

    def get_ic_ndim(self):
        core_num = self.n_qubits
        vir_index = 0
        for istate in range(self.multi.nstates):
            ### Read state integer and extract occupied/virtual info
            occ_list_tmp = int2occ(self.multi.states[istate])
            vir_tmp = occ_list_tmp[-1] + 1
            for ii in range(len(occ_list_tmp)):
                if ii == occ_list_tmp[ii]:
                    core_tmp = ii + 1
            vir_index = max(vir_index, vir_tmp)
            core_num = min(core_num, core_tmp)
        vir_num = self.n_qubits - vir_index
        act_num = self.n_qubits - core_num - vir_num
        self.multi.core_num = core_num
        self.multi.act_num = act_num
        self.multi.vir_num = vir_num
        ndim1, ndim2 = calc_num_ic_theta(n_qubits_system, vir_num,
                                         act_num, core_num)

    def print(self):
        if mpi.main_rank:
            formatstr = f"0{self.n_qubits}b"
            print(f"method       : {self.method}")
            print(f"model        : {self.model}")
            print(f"ansatz       : {self.ansatz}")
            print(f"n_electrons  : {self.n_electrons}")
            print(f"n_orbitals   : {self.n_orbitals}")
            print(f"n_qubits     : {self.n_qubits}")
            print(f"spin         : {self.spin}")
            print(f"multiplicity : {self.multiplicity}")
            print(f"noa          : {self.noa}")
            print(f"nob          : {self.nob}")
            print(f"nva          : {self.nva}")
            print(f"nvb          : {self.nvb}")
            print(f"det          : |{format(self.det, formatstr)}>")


def set_initial_det(noa, nob):
    """Function
    Set the initial wave function to RHF/ROHF determinant.

    Author(s): Takashi Tsuchimochi
    """
    det = 0
    for i in range(noa):
        det = det^(1 << 2*i)
    for i in range(nob):
        det = det^(1 << 2*i + 1)
    return det


def int2occ(state_int):
    """Function
    Given an (base-10) integer, find the index for 1 in base-2 (occ_list)

    Author(s): Takashi Tsuchimochi
    """
    occ_list = []
    k = 0
    while k < state_int:
        kk = 1 << k
        if kk & state_int > 0:
            occ_list.append(k)
        k += 1
    return occ_list


def get_occvir_lists(n_qubits, det):
    """Function
    Generate occlist and virlist for det (base-10 integer).

    Author(s): Takashi Tsuchimochi
    """
    occ_list = int2occ(det)
    vir_list = [i for i in range(n_qubits) if i not in occ_list]
    return occ_list, vir_list
