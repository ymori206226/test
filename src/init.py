"""
#######################
#        quket        #
#######################

init.py

Initializing state.

"""
from . import mpilib as mpi
from . import config as cf
import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.utils import s_squared_operator, commutator
from qulacs.observable import create_observable_from_openfermion_text
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from qulacs.state import inner_product


from .fileio import error, prints, openfermion_print_state, print_geom
from .mod import run_pyscf_mod
from .opelib import create_1body_operator


class QuketData(object):
    class operators:
        def __init__(self):
            return

    class qulacs:
        def __init__(self):
            return

    class projection:
        def __init__(self):
            return

    class multi:
        def __init__(self):
            return

    def __init__(self):
        self.method = "vqe"  # method = (vqe, qite)
        self.model = None  # model = (chemical, hubbard, heisenberg), to be determined
        self.ansatz = None  # ansatz = vqe or qite ansatz like uccsd
        self.basis = None  # basis set
        self.geometry = None  # molecular geometry
        self.det = None  # initial determinant (bit)
        self.multiplicity = None  # spin multiplicity, defined here as Nalpha - Nbeta
        self.charge = 0  # Charge
        self.n_electron = None  # Number of electrons to be considered
        self.n_orbital = None  # Number of orbitals to be considered
        self.hubbard_u = None  # Hubbard U strength (only needed for basis = hubbard)
        self.hubbard_nx = None  # Number of Hubbard sites for x-axis
        self.hubbard_ny = 1  # Number of Hubbard sites for y-axis
        self.run_fci = True  # Whether fci is performed
        self.rho = 1  # Trotter number for related ansatz
        self.DS = 0  # If 0/1, the operation order is Exp[T1] Exp[T2] / Exp[T2] Exp[T1]
        self.excited_states = []  # Initial determinants for excited state calculations
        self.Do1RDM = False  # Whether 1RDM is computed
        self.print_amp_thres = 0.01  # Threshold for printing VQE parameters.
        ### convergence profile
        self.ftol = 1e-9  # Convergence criterion based on energy (cost)
        self.gtol = 1e-5  # Convergence criterion based on gradients

        ### projection section ###
        self.projection.SpinProj = False  # Spin Projection
        self.projection.NumberProj = False  # Number Projection
        self.projection.spin = None  # Target spin for spin projection
        self.projection.Ms = None  # Same as multiplicity
        self.projection.euler_ngrids = [0, -1, 0]  # Grid points for spin projection
        self.projection.number_ngrids = 0  # Grid points for number projection
        ### multi section ###
        self.multi.states = (
            []
        )  # Initial determinants (bits) for multi-state calculatinos (JM-UCC, ic-MRUCC)
        self.multi.weights = (
            []
        )  # Weight for state-average calculations: usually 1 for all
        ### (imaginary) time-evolution section ###
        self.dt = 0  # Time-step for time-evolution
        self.truncate = 0 # Truncation threshold for anti-symmetric hamiltonian

    def initialize(
        self,
        pyscf_guess="minao",
    ):
        """Function
        Run PySCF and initialize parameters.
        """
        if self.basis == "hubbard":
            self.model = "hubbard"
        elif "heisenberg" in self.basis:
            self.model = "heisenberg"
        else:
            self.model = "chemical"
            if self.basis == None or self.geometry == None:
                raise ValueError(
                    "Basis and geometry have to be specified" "for chemical Hamiltonian"
                )
        if self.n_electron == None and self.model in ("hubbard", "chemical"):
            raise ValueError("No electron number")

        if self.model == "hubbard":
            if self.hubbard_u == None or self.hubbard_nx == None:
                raise ValueError(
                    "For hubbard, hubbard_u and hubbard_nx" "have to be given"
                )
            self.n_orbital = self.hubbard_nx * self.hubbard_ny
            # self.jw_Hamiltonian, self.jw_S2 = get_hubbard(
            #    hubbard_u,
            #    hubbard_nx,
            #    hubbard_ny,
            #    n_electron,
            #    run_fci,
            # )
            self.get_operators()
            self.operators.jw_Hamiltonian = jordan_wigner(self.operators.Hamiltonian)
            self.operators.jw_S2 = jordan_wigner(self.operators.S2)
            self.operators.jw_Number = jordan_wigner(self.operators.Number)
            self.operators.Dipole = None

            # Initializing parameters
            prints(
                "Hubbard model: nx = %d  " % self.hubbard_nx,
                "ny = %d  " % self.hubbard_ny,
                "U = %2.2f" % self.hubbard_u,
            )
        elif self.model == "heisenberg":
            nspin = self.n_orbital
            sx = []
            sy = []
            sz = []
            for i in range(nspin):
                sx.append(QubitOperator('X'+str(i)))
                sy.append(QubitOperator('Y'+str(i)))
                sz.append(QubitOperator('Z'+str(i)))
            jw_Hamiltonian = 0*QubitOperator('')
            if "lr" in self.basis:
                for i in range(nspin):
                    j = (i+1) % nspin
                    jw_Hamiltonian += 0.5*(sx[i]*sx[j]+sy[i]*sy[j]+sz[i]*sz[j])
                for i in range(2):
                    j = i+2
                    jw_Hamiltonian += 0.333333333333 * \
                        (sx[i]*sx[j]+sy[i]*sy[j]+sz[i]*sz[j])
            else:
                for i in range(nspin):
                    j = (i+1) % nspin
                    jw_Hamiltonian += sx[i]*sx[j]+sy[i]*sy[j]+sz[i]*sz[j]
            self.operators.jw_Hamiltonian = jw_Hamiltonian
            self.operators.jw_S2 = None
            self.operators.jw_Number = None
            self.operators.Hamiltonian = None
            self.operators.S2 = None
            self.operators.Number = None
            self.n_qubit = self.n_orbital
            if self.det == None:
                self.det = 1
            self.current_det = self.det
            return
        elif self.model == "chemical":
            if cf._geom_update:
                # New geometry found. Run PySCF and get operators.
                self.get_operators(
                    guess=pyscf_guess,
                )
                # Set Jordan-Wigner Hamiltonian and S2 operators using PySCF and Open-Fermion
                self.operators.jw_Hamiltonian = jordan_wigner(
                    self.operators.Hamiltonian
                )
                self.operators.jw_S2 = jordan_wigner(self.operators.S2)
                self.operators.jw_Number = jordan_wigner(self.operators.Number)
                cf._geom_update = False

        # Initializing parameters
        self.n_qubit = self.n_orbital * 2
        self.projection.Ms = self.multiplicity - 1
        self.n_qubit_anc = self.n_qubit + 1
        self.anc = self.n_qubit
        self.state = None

        #    # Check spin, multiplicity, and Ms
        if self.projection.spin == None:
            self.projection.spin = self.multiplicity  # Default
        if (
            self.projection.spin - self.multiplicity
        ) % 2 != 0 or self.projection.spin < self.multiplicity:
            prints(
                "Spin = {}    Ms = {}".format(self.projection.spin, self.projection.Ms)
            )
            error("Spin and Ms not cosistent.")
        if (self.n_electron + self.multiplicity - 1) % 2 != 0:
            prints(
                "Incorrect specification for n_electrons = {} and multiplicity = {}.".format(
                    self.n_electron, self.multiplicity
                )
            )
        # Number of occupied orbitals of alpha
        self.noa = (self.n_electron + self.multiplicity - 1) // 2
        # Number of occupied orbitals of beta
        self.nob = self.n_electron - self.noa
        # Number of virtual orbitals of alpha
        self.nva = self.n_orbital - self.noa
        # Number of virtual orbitals of beta
        self.nvb = self.n_orbital - self.nob
        # Check initial determinant
        if self.det == None:
            # Initial determinant is RHF or ROHF
            self.det = set_initial_det(self.noa, self.nob)
        self.current_det = self.det
        #
        if self.ansatz in ("phf", "suhf", "sghf", "opt_puccsd", "opt_puccd"):
            self.projection.SpinProj = True

        # Excited states (orthogonally-constraint)
        self.nexcited = len(self.excited_states)
        self.lower_states = []

        # Multi states
        self.multi.nstates = len(self.multi.weights)

    def get_operators(
        self,
        guess="minao",
    ):
        from . import mpilib as mpi

        if self.model == "chemical":
            if mpi.main_rank:
                # Run electronic structure calculations
                molecule, pyscf_molecule = run_pyscf_mod(
                    guess,
                    self.n_orbital,
                    self.n_electron,
                    MolecularData(
                        self.geometry, self.basis, self.multiplicity, self.charge
                    ),
                )
                # Freeze core orbitals and truncate to active space
                if self.n_electron is None:
                    n_core_orbitals = 0
                    occupied_indices = None
                else:
                    n_core_orbitals = (molecule.n_electrons - self.n_electron) // 2
                    occupied_indices = list(range(n_core_orbitals))

                if self.n_orbital is None:
                    active_indices = None
                else:
                    active_indices = list(
                        range(n_core_orbitals, n_core_orbitals + self.n_orbital)
                    )

                hf_energy = molecule.hf_energy
                fci_energy = molecule.fci_energy

                mo_coeff = molecule.canonical_orbitals.astype(float)
                natom = pyscf_molecule.natm
                atom_charges = []
                atom_coords = []
                for i in range(natom):
                    atom_charges.append(pyscf_molecule.atom_charges()[i])
                    atom_coords.append(pyscf_molecule.atom_coords()[i])

                rint = pyscf_molecule.intor("int1e_r")

                Hamiltonian = molecule.get_molecular_hamiltonian(
                    occupied_indices=occupied_indices, active_indices=active_indices
                )
                #
                #                # Dipole operators from dipole integrals (AO)
                rx = create_1body_operator(
                    mo_coeff, rint[0], ao=True, n_active_orbitals=self.n_orbital
                )
                ry = create_1body_operator(
                    mo_coeff, rint[1], ao=True, n_active_orbitals=self.n_orbital
                )
                rz = create_1body_operator(
                    mo_coeff, rint[2], ao=True, n_active_orbitals=self.n_orbital
                )
                Dipole = [rx, ry, rz]
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
            from openfermion.hamiltonians import fermi_hubbard
            from openfermion.transforms import jordan_wigner

            self.operators.Hamiltonian = fermi_hubbard(
                self.hubbard_nx, self.hubbard_ny, 1, self.hubbard_u
            )
            self.operators.jw_Hamiltonian = jordan_wigner(self.operators.Hamiltonian)
            self.hf_energy = None
            self.fci_energy = None
            if self.run_fci == 1:
                n_qubit = self.hubbard_nx * self.hubbard_ny * 2
                self.operators.jw_Hamiltonian.compress()
                qubit_eigen = QubitDavidson(self.operators.jw_Hamiltonian, n_qubit)
                # Initial guess :  | 0000...00111111>
                #                             ~~~~~~ = n_electrons
                guess = np.zeros((2 ** n_qubit, 1))
                #
                guess[2 ** self.n_electron - 1][0] = 1.0
                n_state = 1
                results = qubit_eigen.get_lowest_n(n_state, guess)
                prints("Convergence?           : ", results[0])
                prints("Ground State Energy    : ", results[1][0])
                self.fci_energy = results[1][0]
            #                prints("Wave function          : ")
            #                openfermion_print_state(results[2], n_qubit, 0)
            #

            self.mo_coeff = None
            self.natom = self.hubbard_nx * self.hubbard_ny
            self.atom_charges = None
            self.atom_coords = None
        from openfermion.utils import number_operator, s_squared_operator

        self.operators.S2 = s_squared_operator(self.n_orbital)
        self.operators.Number = number_operator(self.n_orbital)

    def jw_to_qulacs(self):
        if self.operators.jw_Hamiltonian is not None:
            self.qulacs.Hamiltonian = create_observable_from_openfermion_text(
                str(self.operators.jw_Hamiltonian)
            )
        else:
            self.qulacs.Hamiltonian = None
        if self.operators.jw_S2 is not None:
            self.qulacs.S2 = create_observable_from_openfermion_text(
                str(self.operators.jw_S2)
            )
        else:
            self.qulacs.S2 = None
        if self.operators.jw_Number is not None:
            self.qulacs.Number = create_observable_from_openfermion_text(
                str(self.operators.jw_Number)
            )
        else:
            self.qulacs.Number = None

    def set_projection(self, euler_ngrids=None, number_ngrids=None):
        """Function
        Set the angles and weights for integration of spin-projection and number-projection.

        Spin-rotation operator     Exp[ -i alpha Sz ]   Exp[ -i beta Sy ]   Exp[ -i gamma Sz ]
        weight                     Exp[ i m' alpha]      d*[j,m',m]         Exp[ i m gamma]

        Angles alpha and gamma are determined by Trapezoidal quadrature, and beta is determined by Gauss-Legendre quadrature.

        Number-rotation operator   Exp[ -i phi N ]
        weight                     Exp[  i phi Ne ]

        phi are determined by Trapezoidal quadrature.

        """
        from .phflib import weightspin, trapezoidal, simpson
        if self.model =="heisenberg":
            # No spin or number symmetry in the model
            return
        trap = True
        if euler_ngrids is not None:
            self.projection.euler_ngrids = euler_ngrids
        if number_ngrids is not None:
            self.projection.number_ngrids = number_ngrids
        if self.projection.SpinProj:
            prints(
                "Projecting to spin space :  s = {:.1f}    Ms = {} ".format(
                    (self.projection.spin - 1) / 2, self.projection.Ms
                )
            )
            prints(
                "             Grid points :  (alpha,beta,gamma) = ({},{},{})".format(
                    self.projection.euler_ngrids[0],
                    self.projection.euler_ngrids[1],
                    self.projection.euler_ngrids[2],
                )
            )
            self.projection.sp_angle = []
            self.projection.sp_weight = []

            # Alpha

            if self.projection.euler_ngrids[0] > 1:
                if trap:
                    alpha, wg_alpha = trapezoidal(
                        0, 2 * np.pi, self.projection.euler_ngrids[0]
                    )
                else:
                    alpha, wg_alpha = simpson(
                        0, 2 * np.pi, self.projection.euler_ngrids[0]
                    )
            else:
                alpha = [0]
                wg_alpha = [1]

            self.projection.sp_angle.append(alpha)
            self.projection.sp_weight.append(wg_alpha)

            # Beta

            if self.projection.euler_ngrids[1] > 1:
                beta, wg_beta = np.polynomial.legendre.leggauss(
                    self.projection.euler_ngrids[1]
                )
                beta = np.arccos(beta)
                beta = beta.tolist()
                self.projection.dmm = weightspin(
                    self.projection.euler_ngrids[1],
                    self.projection.spin,
                    self.projection.Ms,
                    self.projection.Ms,
                    beta,
                )
            else:
                beta = [0]
                wg_beta = [1]
                self.projection.dmm = [1]

            self.projection.sp_angle.append(beta)
            self.projection.sp_weight.append(wg_beta)

            # Gamma

            if self.projection.euler_ngrids[2] > 1:
                if trap:
                    gamma, wg_gamma = trapezoidal(
                        0, 2 * np.pi, self.projection.euler_ngrids[2]
                    )
                else:
                    gamma, wg_gamma = simpson(
                        0, 2 * np.pi, self.projection.euler_ngrids[2]
                    )
            else:
                gamma = [0]
                wg_gamma = [1]
            self.projection.sp_angle.append(gamma)
            self.projection.sp_weight.append(wg_gamma)

        if self.projection.NumberProj:
            self.projection.number_ngrids = number_ngrids
            prints(
                "Projecting to number space :  N = {}".format(
                    self.projection.number_ngrids
                )
            )

            self.projection.np_angle = []
            self.projection.np_weight = []
            # phi

            if self.projection.number_ngrids > 1:
                if trap:
                    phi, wg_phi = trapezoidal(
                        0, 2 * np.pi, self.projection.number_ngrids
                    )
                else:
                    gamma, wg_gamma = simpson(
                        0, 2 * np.pi, self.projection.number_ngrids
                    )
            else:
                phi = [0]
                wg_phi = [1]
            self.projection.np_angle = phi
            self.projection.np_weight = wg_phi

    def get_ic_ndim(self):
        core_num = self.n_qubit
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
        vir_num = self.n_qubit - vir_index
        act_num = self.n_qubit - core_num - vir_num
        self.multi.core_num = core_num
        self.multi.act_num = act_num
        self.multi.vir_num = vir_num
        from .icmrucc import calc_num_ic_theta

        ndim1, ndim2 = calc_num_ic_theta(n_qubit_system, vir_num, act_num, core_num)

    def print(self):
        if mpi.main_rank:
            print("method       : ", self.method)
            print("model        : ", self.model)
            print("ansatz       : ", self.ansatz)
            print("n_electron   : ", self.n_electron)
            print("n_orbital    : ", self.n_orbital)
            print("n_qubit      : ", self.n_qubit)
            print("spin         : ", self.spin)
            print("multiplicity : ", self.multiplicity)
            print("noa          : ", self.noa)
            print("nob          : ", self.nob)
            print("nva          : ", self.nva)
            print("nvb          : ", self.nvb)
            print(
                "det          : |{}>".format(
                    (format(self.det, "0" + str(self.n_qubit) + "b"))
                )
            )


def set_initial_det(noa, nob):
    """Function
    Set the initial wave function to RHF/ROHF determinant.

    Author(s): Takashi Tsuchimochi
    """
    det = 0
    for i in range(noa):
        det = det ^ (1 << 2 * i)
    for i in range(nob):
        det = det ^ (1 << 2 * i + 1)
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


def get_occvir_lists(n_qubit, det):
    """Function
    Generate occlist and virlist for det (base-10 integer).

    Author(s): Takashi Tsuchimochi
    """
    occ_list = int2occ(det)
    vir_list = [i for i in range(n_qubit) if i not in occ_list]
    return occ_list, vir_list
