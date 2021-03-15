from typing import List
from dataclasses import dataclass, field

import numpy as np
from openfermion.utils import number_operator, s_squared_operator
from openfermion.hamiltonians import MolecularData

from . import config as cf
from . import mpilib as mpi
from .mod import run_pyscf_mod
from .opelib import create_1body_operator
from .fileio import prints, print_geom


@dataclass
class Chemical(MolecularData):
    geometry: List = field(default_factory=list)
    basis: str = None
    multiplicity: int = None
    charge: int = None
    description: str = ""
    filename: str = ""
    data_directory: str = None
    n_active_orbitals: int = None
    n_active_electrons: int = None

    n_qubits: int = field(init=False, default=None)

    def __post_init__(self, *args, **kwds):
        if self.n_active_orbitals is not None:
            if self.n_active_orbitals <= 0:
                error(f"# orbitals = {self.n_active_orbitals}!")
        if self.n_active_electrons is not None:
            if self.n_active_electrons <= 0:
                error(f"# electrons = {self.n_active_electrons}!")

        super().__init__(geometry=self.geometry, basis=self.basis,
                         multiplicity=self.multiplicity, charge=self.charge,
                         description=self.description, filename=self.filename,
                         data_directory=self.data_directory)

        self._guess = cf.pyscf_guess
        self, self._pyscf_mol = run_pyscf_mod(cf.pyscf_guess,
                                              self.n_active_orbitals,
                                              self.n_active_electrons,
                                              self)

        if self.n_active_orbitals is None:
            self.n_active_orbitals = self.n_orbitals
        if self.n_active_electrons is None:
            self.n_active_electrons = self.n_electrons

        self.n_qubits = self.n_active_orbitals*2


    def get_operators(self, guess="minao", run_fci=True):
        if mpi.main_rank:
            # Run electronic structure calculations
            if guess != self._guess:
                self, self._pyscf_mol = run_pyscf_mod(guess,
                                                      self.n_active_orbitals,
                                                      self.n_active_electrons,
                                                      self,
                                                      run_fci=run_fci)

            # Number of core orbitals is 0.
            occupied_indices = list(range(0))
            active_indices = list(range(self.n_active_orbitals))

            hf_energy = self.hf_energy
            fci_energy = self.fci_energy

            mo_coeff = self.canonical_orbitals.astype(float)
            natom = self._pyscf_mol.natm
            atom_charges = self._pyscf_mol.atom_charges().reshape(1, -1)
            atom_coords = self._pyscf_mol.atom_coords()
            rint = self._pyscf_mol.intor("int1e_r")

            Hamiltonian = self.get_molecular_hamiltonian(
                    occupied_indices=occupied_indices,
                    active_indices=active_indices)

            # Dipole operators from dipole integrals (AO)
            rx = create_1body_operator(mo_coeff, rint[0], ao=True,
                                       n_active_orbitals=self.n_active_orbitals)
            ry = create_1body_operator(mo_coeff, rint[1], ao=True,
                                       n_active_orbitals=self.n_active_orbitals)
            rz = create_1body_operator(mo_coeff, rint[2], ao=True,
                                       n_active_orbitals=self.n_active_orbitals)
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

        S2 = s_squared_operator(self.n_active_orbitals)
        Number = number_operator(self.n_active_orbitals)

        return Hamiltonian, S2, Number, Dipole
