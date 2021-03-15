from dataclasses import dataclass, field, InitVar

from numpy import ndarray
from openfermion.utils import number_operator, s_squared_operator
from openfermion.hamiltonians import fermi_hubbard

from .fileio import prints, error


@dataclass
class Hubbard():
    """Hubbard model class.

    Attributes:
        hubbard_u (float): ??
        hubbard_nx (int): Number of hubbard sites for x-axis.
        hubbard_ny (int): Number of hubbard sites for y-axis.
        natom (int): Number of atoms.
        n_orbitals (int): Number of spatial orbitals.
        hf_energy (float): HF energy.
        fci_energy (float): FCI energy.
        mo_coeff (float): Orbital coefficients.
        atom_charges (ndarray): Nuclear effective charge of the atoms.
        atom_coords (ndarray): Coordinates of the atoms.

    Author(s): Yuma Shhimomoto
    """
    # Note; rename 'n_orbitals' to 'n_active_orbitals' when read input file.
    n_active_electrons: InitVar[int] = None

    basis: str = "hubbard"
    multiplicity: int = None
    hubbard_u: float = None   # ??
    hubbard_nx: int = None
    hubbard_ny: int = 1
    n_electrons: int = None

    natom: int = field(init=False)
    n_orbitals: int = field(init=False)

    n_qubits: int = field(init=False, default=None)
    hf_energy: float = field(init=False, default=None)
    fci_energy: float = field(init=False, default=None)
    mo_coeff: ndarray = field(init=False, default=None)
    atom_charges: ndarray = field(init=False, default=None)
    atom_coords: ndarray = field(init=False, default=None)

    def __post_init__(self, n_active_electrons, *args, **kwds):
        if self.hubbard_u is None or self.hubbard_nx is None:
            error("For hubbard, hubbard_u and hubbard_nx have to be given")
        if n_active_electrons is None:
            error("No electron number")
        if self.hubbard_nx <= 0:
            error("Hubbard model but hubbard_nx is not defined!")
        if n_active_electrons <= 0:
            error("# electrons <= 0 !")
        self.n_electrons = n_active_electrons

        self.natom = self.hubbard_nx*self.hubbard_ny
        self.n_orbitals = self.hubbard_nx*self.hubbard_ny
        self.n_qubits = self.n_orbitals*2

        # Initializing parameters
        prints(f"Hubbard model: nx = {self.hubbard_nx}  "
               f"ny = {self.hubbard_ny}  "
               f"U = {self.hubbard_u:2.2f}")

    @property
    def n_active_electrons(self):
        return self.n_electrons

    @property
    def n_active_orbitals(self):
        return self.n_orbitals

    def get_operators(self, guess="minao", run_fci=True, jw_hamiltonian=None):
        Hamiltonian = fermi_hubbard(self.hubbard_nx, self.hubbard_ny,
                                    1, self.hubbard_u)
        S2 = s_squared_operator(self.n_orbitals)
        Number = number_operator(self.n_orbitals)
        return Hamiltonian, S2, Number
