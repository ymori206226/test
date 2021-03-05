from dataclasses import dataclass, field, InitVar


@dataclass
class Heisenberg():
    """Heisenberg model class.

    Attributes:
        nspin (int): Number of spin.
        n_qubits (int): Number of qubits.

    Author(s): Yuma Shimomoto
    """
    # Instance variables for __post_init__.
    n_orbitals: InitVar[int]

    nspin: int = field(init=False)
    n_qubits: int = field(init=False)

    def __post_init__(self, n_orbitals, *args, **kwds):
        self.nspin = n_orbitals
        self.n_qubits = n_orbitals

    def get_operators(self, basis):
        sx = sy = sz = []
        for i in range(self.nspin):
            sx.append(QubitOperator(f"X{i}"))
            sy.append(QubitOperator(f"Y{i}"))
            sz.append(QubitOperator(f"Z{i}"))

        jw_Hamiltonian = 0 * QubitOperator("")
        if "lr" in basis:
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
