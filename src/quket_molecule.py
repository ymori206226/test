from typing import List
from dataclasses import dataclass, field

from openfermion.hamiltonians import MolecularData

from . import mpilib as mpi


@dataclass
class QuketMolecule(MolecularData):
    geometry: List = field(default_factory=list)
    basis: str = None
    multiplicity: int = None
    charge: int = None
    description: str = ""
    filename: str = ""
    data_directory: str = None

    def __post_init__(self, *args, **kwds):
        super().__init__(geometry=self.geometry, basis=self.basis,
                         multiplicity=self.multiplicity, charge=self.charge,
                         description=self.description, filename=self.filename,
                         data_directory=self.data_directory)

    def get_operators(self, guess="minao", pyscf_mol=None):
        if mpi.main_rank:
            # Run electronic structure calculations
            if pyscf_mol is None:
                self, pyscf_mol = run_pyscf_mod(guess, self.n_orbitals,
                                                self.n_electrons, self,
                                                run_fci=self.run_fci)

            # 'n_electrons' and 'n_orbitals' must not be 'None'.
            n_core_orbitals = (self.n_electrons-self.n_electrons)//2
            occupied_indices = list(range(n_core_orbitals))
            active_indices = list(range(n_core_orbitals,
                                        n_core_orbitals+self.n_orbitals))

            hf_energy = self.hf_energy
            fci_energy = self.fci_energy

            mo_coeff = self.canonical_orbitals.astype(float)
            natom = pyscf_mol.natm
            atom_charges = pyscf_mol.atom_charges().reshape(-1, 1)
            atom_coords = pyscf_mol.atom_coords()
            rint = pyscf_mol.intor("int1e_r")

            Hamiltonian = self.get_molecular_hamiltonian(
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

        S2 = s_squared_operator(self.n_orbitals)
        Number = number_operator(self.n_orbitals)

        return Hamiltonian, S2, Number, Dipole
