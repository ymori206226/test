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
