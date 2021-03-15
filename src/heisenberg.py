from dataclasses import dataclass, field, InitVar

from openfermion.ops import QubitOperator

from .fileio import error


@dataclass
class Heisenberg():
    """Heisenberg model class.

    Attributes:
        nspin (int): Number of spin.
        n_qubits (int): Number of qubits.

    Author(s): Yuma Shimomoto
    """
    # Note; rename 'n_orbitals' to 'n_active_orbitals' when read input file.
    n_active_orbitals: InitVar[int] = None

    basis: str = "lr-heisenberg"
    n_orbitals: int = None

    nspin: int = field(init=False)
    n_qubits: int = field(init=False)

    def __post_init__(self, n_active_orbitals, *args, **kwds):
        if n_active_orbitals is None:
            error("'n_orbitals' is None.")
        if n_active_orbitals <= 0:
            error("# orbitals <= 0!")

        self.nspin = self.n_qubits = self.n_orbitals = n_active_orbitals

    @property
    def n_active_orbitals(self):
        return self.n_orbitals

    def get_operators(self):
        sx = []
        sy = []
        sz = []
        for i in range(self.nspin):
            sx.append(QubitOperator(f"X{i}"))
            sy.append(QubitOperator(f"Y{i}"))
            sz.append(QubitOperator(f"Z{i}"))

        jw_Hamiltonian = 0*QubitOperator("")
        if "lr" in self.basis:
            for i in range(self.nspin):
                j = (i+1)%self.nspin
                jw_Hamiltonian += 0.5*(sx[i]*sx[j]
                                       + sy[i]*sy[j]
                                       + sz[i]*sz[j])
            for i in range(2):
                j = i+2
                jw_Hamiltonian += 1./3.*(sx[i]*sx[j]
                                         + sy[i]*sy[j]
                                         + sz[i]*sz[j])
        else:
            for i in range(self.nspin):
                j = (i+1)%self.nspin
                jw_Hamiltonian += sx[i]*sx[j] + sy[i]*sy[j] + sz[i]*sz[j]
        return jw_Hamiltonian
