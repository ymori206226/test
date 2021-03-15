"""
#######################
#        quket        #
#######################

mod.py

Modified versions of OpenFermionPySCF routines
to enable active space calculations.

"""
# いる?
from __future__ import absolute_import

#from typing import Dict
#from dataclasses import dataclass

#import numpy as np
#from pyscf import gto, scf, ao2mo, ci, cc, fci, mp
from pyscf import ci, cc, fci, mp, mcscf
from openfermion import MolecularData
from openfermionpyscf import prepare_pyscf_molecule
from openfermionpyscf._run_pyscf import (compute_scf, compute_integrals,
                                         PyscfMolecularData)

from . import config as cf
from . import mpilib as mpi
from .fileio import error


#def prepare_pyscf_molecule_mod(molecule):
#    """Function
#    This function creates and saves a pyscf input file.
#    Args:
#        molecule: An instance of the MolecularData class.
#    Returns:
#        pyscf_molecule: A pyscf molecule instance.
#
#    Author(s): Takashi Tsuchimochi
#    """
#    pyscf_molecule = gto.Mole()
#    pyscf_molecule.atom = molecule.geometry
#    pyscf_molecule.basis = molecule.basis
#    pyscf_molecule.spin = molecule.multiplicity - 1
#    pyscf_molecule.charge = molecule.charge
#    pyscf_molecule.symmetry = False
#    pyscf_molecule.build()
#    from pprint import pprint
#    pprint(pyscf_molecule.__dict__)
#    return pyscf_molecule


#def compute_scf_mod(pyscf_molecule):
#    """Function
#    Perform a Hartree-Fock calculation.
#    Args:
#        pyscf_molecule: A pyscf molecule instance.
#    Returns:
#        pyscf_scf: A PySCF "SCF" calculation object.
#
#    Author(s): Takashi Tsuchimochi
#    """
#    if pyscf_molecule.spin:
#        pyscf_scf = scf.ROHF(pyscf_molecule)
#    else:
#        pyscf_scf = scf.RHF(pyscf_molecule)
#    return pyscf_scf


#def compute_integrals_mod(pyscf_molecule, pyscf_scf):
#    """Function
#    Compute the 1-electron and 2-electron integrals.
#    Args:
#        pyscf_molecule: A pyscf molecule instance.
#        pyscf_scf: A PySCF "SCF" calculation object.
#    Returns:
#        one_electron_integrals: An N by N array storing h_{pq}
#        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
#
#    Author(s): Takashi Tsuchimochi
#    """
#    # Get one electrons integrals.
#    n_orbitals = pyscf_scf.mo_coeff.shape[1]
#    one_electron_compressed \
#            = pyscf_scf.mo_coeff.T@pyscf_scf.get_hcore()@pyscf_scf.mo_coeff
#    one_electron_integrals \
#            = one_electron_compressed.reshape(n_orbitals, n_orbitals)
#
#    # Get two electron integrals in compressed format.
#    two_electron_compressed = ao2mo.kernel(pyscf_molecule, pyscf_scf.mo_coeff)
#
#    two_electron_integrals = ao2mo.restore(
#        1, two_electron_compressed, n_orbitals  # no permutation symmetry
#    )
#    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
#    # h[p,q,r,s] = (ps|qr)
#    two_electron_integrals = np.asarray(
#        two_electron_integrals.transpose(0, 2, 3, 1), order="C"
#    )
#
#    # Return.
#    return one_electron_integrals, two_electron_integrals


### modify generate_molecular_hamiltonian to be able to use chkfile
def generate_molecular_hamiltonian_mod(guess, geometry, basis, multiplicity,
                                       charge=0, n_active_electrons=None,
                                       n_active_orbitals=None):
    """Function
    Old subroutine to get molecular hamiltonian by using pyscf.

    Author(s): Takashi Tsuchimochi
    """

    # Run electronic structure calculations
    molecule, pyscf_mol \
            = run_pyscf_mod(guess, n_active_orbitals, n_active_electrons,
                            MolecularData(geometry, basis, multiplicity, charge,
                                          data_directory=cf.input_dir))

    # Freeze core orbitals and truncate to active space
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons-n_active_electrons)//2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(range(n_core_orbitals,
                                    n_core_orbitals+n_active_orbitals))

    return molecule.get_molecular_hamiltonian(occupied_indices=occupied_indices,
                                              active_indices=active_indices)


def run_pyscf_mod(guess, n_active_orbitals, n_active_electrons, molecule,
                  spin=None, run_mp2=False, run_cisd=False,
                  run_ccsd=False, run_fci=False, run_casci=True, verbose=False):
    """Function
    This function runs a pyscf calculation.

    Args:
        molecule: An instance of the MolecularData or PyscfMolecularData class.
        run_mp2: Optional boolean to run MP2 calculation.
        run_cisd: Optional boolean to run CISD calculation.
        run_ccsd: Optional boolean to run CCSD calculation.
        run_fci: Optional boolean to FCI calculation.
        verbose: Boolean whether to print calculation results to screen.
    Returns:
        molecule: The updated PyscfMolecularData object. Note the attributes
        of the input molecule are also updated in this function.

    Author(s): Takashi Tsuchimochi
    """
    # Prepare pyscf molecule.
    #pyscf_molecule = prepare_pyscf_molecule_mod(molecule)
    pyscf_molecule = prepare_pyscf_molecule(molecule)
    molecule.n_orbitals = int(pyscf_molecule.nao_nr())
    molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

    # Run SCF.
    #pyscf_scf = compute_scf_mod(pyscf_molecule)
    pyscf_scf = compute_scf(pyscf_molecule)
    pyscf_scf.verbose = 0
# chkファイルのパスを指定すると並列計算時にOSErrorが出る
    #pyscf_scf.run(chkfile=cf.chk, init_guess=guess,
    #              conv_tol=1e-12, conv_tol_grad=1e-12)
    pyscf_scf.run(init_guess=guess, conv_tol=1e-12, conv_tol_grad=1e-12)
    molecule.hf_energy = float(pyscf_scf.e_tot)

    # Set number of active electrons/orbitals.
    if n_active_electrons is None:
        n_active_electrons = molecule.n_electrons
    if n_active_orbitals is None:
        n_active_orbitals = molecule.n_orbitals

    # Hold pyscf data in molecule. They are required to compute density
    # matrices and other quantities.
    molecule._pyscf_data = {}
    pyscf_data = {}
    pyscf_data["mol"] = pyscf_molecule
    pyscf_data["scf"] = pyscf_scf

    # Populate fields.
    molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)

    # Get integrals.
    #one_body_integrals, two_body_integrals \
    #        = compute_integrals_mod(pyscf_molecule, pyscf_scf)
    one_body_integrals, two_body_integrals \
            = compute_integrals(pyscf_molecule, pyscf_scf)
    molecule.one_body_integrals = one_body_integrals
    molecule.two_body_integrals = two_body_integrals
    molecule.overlap_integrals = pyscf_scf.get_ovlp()

    if run_mp2:
        if molecule.multiplicity != 1:
            error("WARNING: RO-MP2 is not available in PySCF.")
        else:
            pyscf_mp2 = mp.MP2(pyscf_scf)
            pyscf_mp2.verbose = 0
            #pyscf_mp2.run(chkfile=cf.chk, init_guess=guess,
            #              conv_tol=1e-12, conv_tol_grad=1e-12)
            pyscf_mp2.run(init_guess=guess, conv_tol=1e-12, conv_tol_grad=1e-12)
            # molecule.mp2_energy = pyscf_mp2.e_tot  # pyscf-1.4.4 or higher
            molecule.mp2_energy = pyscf_scf.e_tot + pyscf_mp2.e_corr
            pyscf_data["mp2"] = pyscf_mp2
            if verbose:
                print(f"MP2 energy for {molecule.name} "
                      f"({molecule.n_electrons} electrons) is "
                      f"{molecule.mp2_energy}.")

    # Run CISD.
    if run_cisd:
        pyscf_cisd = ci.CISD(pyscf_scf)
        pyscf_cisd.verbose = 0
        #pyscf_cisd.run(chkfile=cf.chk, init_guess=guess,
        #               conv_tol=1e-12, conv_tol_grad=1e-12)
        pyscf_cisd.run(init_guess=guess, conv_tol=1e-12, conv_tol_grad=1e-12)
        molecule.cisd_energy = pyscf_cisd.e_tot
        pyscf_data["cisd"] = pyscf_cisd
        if verbose:
            print(f"CISD energy for {molecule.name} "
                  f"({molecule.n_electrons} electrons) is "
                  f"{molecule.cisd_energy}.")

    # Run CCSD.
    if run_ccsd:
        pyscf_ccsd = cc.CCSD(pyscf_scf)
        pyscf_ccsd.verbose = 0
        #pyscf_ccsd.run(chkfile=cf.chk, init_guess=guess,
        #               conv_tol=1e-12, conv_tol_grad=1e-12)
        pyscf_ccsd.run(init_guess=guess, conv_tol=1e-12, conv_tol_grad=1e-12)
        molecule.ccsd_energy = pyscf_ccsd.e_tot
        pyscf_data["ccsd"] = pyscf_ccsd
        if verbose:
            print(f"CCSD energy for {molecule.name} "
                  f"({molecule.n_electrons} electrons) is "
                  f"{molecule.ccsd_energy}.")

    # Run FCI.
    if run_fci:
        pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
        pyscf_fci.verbose = 0
        molecule.fci_energy = pyscf_fci.kernel()[0]
        pyscf_data["fci"] = pyscf_fci
        if verbose:
            print(f"FCI energy for {molecule.name} "
                  f"({molecule.n_electrons} electrons) is "
                  f"{molecule.fci_energy}.")

    # CASCI (FCI)
    ### Change the spin ... (S,Ms) = (spin, multiplicity)
    if run_casci:
        if spin is None:
            spin = molecule.multiplicity
        pyscf_molecule.spin = spin - 1
        pyscf_casci = mcscf.CASCI(pyscf_scf,
                                  n_active_orbitals, n_active_electrons)
        #pyscf_scf = compute_scf_mod(pyscf_molecule)
        #pyscf_scf.run()
        #pyscf_scf = compute_scf(pyscf_molecule)
        #pyscf_scf.run(chkfile=cf.chk, init_guess=guess,
        #              conv_tol=1e-12, conv_tol_grad=1e-12)
        ### reload mo coeffictions
        #pyscf_scf.mo_coeff = molecule.canonical_orbitals
        #pyscf_scf.mo_energy = molecule.orbital_energies
        #Efci = pyscf_scf.CASCI(n_active_orbitals, n_active_electrons).kernel()
        Efci = pyscf_casci.casci()[0]
        molecule.fci_energy = Efci

    #cf.fci_coeff = fci[2]
    #fci2qubit(n_active_orbitals,n_active_electrons,pyscf_molecule.spin,fci[2])

    # Return updated molecule instance.
    #pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    #pyscf_molecular_data.__dict__.update(molecule.__dict__)
    #pyscf_molecular_data.save()
    #return pyscf_molecular_data, pyscf_molecule
    return molecule, pyscf_molecule
