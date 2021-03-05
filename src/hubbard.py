from dataclasses import dataclass, field

from numpy import ndarray
from openfermion.hamiltonians import fermi_hubbard


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

    def __post_init__(self, *args, **kwds):
        if self.hubbard_u is None or self.hubbard_nx is None:
            raise ValueError("For hubbard, hubbard_u and hubbard_nx"
                             "have to be given")
        if self.n_electrons is None:
            raise ValueError("No electron number")

        self.natom = self.hubbard_nx * self.hubbard_ny
        self.n_orbitals = self.hubbard_nx * self.hubbard_ny

        # Initializing parameters
        prints(f"Hubbard model: nx = {self.hubbard_nx}"
               f"ny = {self.hubbard_ny}"
               f"U = {self.hubbard_u:2.2f}")

    def get_operators(self, guess="minao"):
        Hamiltonian = fermi_hubbard(self.hubbard_nx, self.hubbard_ny,
                                    1, self.hubbard_u)
        S2 = s_squared_operator(self.n_orbitals)
        Number = number_operator(self.n_orbitals)

       self.hf_energy = None
       self.fci_energy = None
       if self.run_fci:
           self.operators.jw_Hamiltonian.compress()
           qubit_eigen = QubitDavidson(self.operators.jw_Hamiltonian,
                                       self.n_qubits)
           # Initial guess :  | 0000...00111111>
           #                             ~~~~~~ = n_electrons
           guess = np.zeros((2**self.n_qubits, 1))
           guess[2**self.n_electrons - 1][0] = 1.0
           n_state = 1
           results = qubit_eigen.get_lowest_n(n_state, guess)
           prints("Convergence?           : ", results[0])
           prints("Ground State Energy    : ", results[1][0])
           self.fci_energy = results[1][0]
           #prints("Wave function          : ")
           #openfermion_print_state(results[2], n_qubits, 0)


        return Hamiltonian, S2, Number
