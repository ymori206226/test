from dataclasses import dataclass, field

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
    basis: str = "lr-heisenberg"
    n_orbitals: int = None

    nspin: int = field(init=False)
    n_qubits: int = field(init=False)

    def __post_init__(self, *args, **kwds):
        if self.n_orbitals is None:
            error("'n_orbitals' is None.")
        self.nspin = self.n_qubits = self.n_orbitals

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
