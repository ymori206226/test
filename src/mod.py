"""
#######################
#        quket        #
#######################

mod.py

Modified versions of OpenFermionPySCF routines
to enable active space calculations.

"""

from __future__ import absolute_import
from functools import reduce
import numpy 
import itertools
from openfermion.transforms import jordan_wigner
from openfermion.ops import QubitOperator
import pyscf
from pyscf import gto, ao2mo, ci, cc, fci, mp, mcscf, scf

from openfermion import MolecularData
from openfermionpyscf import PyscfMolecularData

from . import config


def prepare_pyscf_molecule_mod(molecule):
    """
    This function creates and saves a pyscf input file.
    Args:
        molecule: An instance of the MolecularData class.
    Returns:
        pyscf_molecule: A pyscf molecule instance.
    """
    pyscf_molecule = gto.Mole()
    pyscf_molecule.atom = molecule.geometry
    pyscf_molecule.basis = molecule.basis
    pyscf_molecule.spin = molecule.multiplicity - 1
    pyscf_molecule.charge = molecule.charge
    pyscf_molecule.symmetry = False
    pyscf_molecule.build()

    return pyscf_molecule


def compute_scf_mod(pyscf_molecule):
    """
    Perform a Hartree-Fock calculation.
    Args:
        pyscf_molecule: A pyscf molecule instance.
    Returns:
        pyscf_scf: A PySCF "SCF" calculation object.
    """
    if pyscf_molecule.spin:
        pyscf_scf = scf.ROHF(pyscf_molecule)
    else:
        pyscf_scf = scf.RHF(pyscf_molecule)
    return pyscf_scf


def compute_integrals_mod(pyscf_molecule, pyscf_scf):
    """
    Compute the 1-electron and 2-electron integrals.
    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.
    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(numpy.dot, (pyscf_scf.mo_coeff.T,
                                                 pyscf_scf.get_hcore(),
                                                 pyscf_scf.mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = numpy.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals


### modify generate_molecular_hamiltonian to be able to use chkfile
def generate_molecular_hamiltonian_mod(
        guess,
        geometry,
        basis,
        multiplicity,
        charge=0,
        n_active_electrons=None,
        n_active_orbitals=None
        ):

    # Run electronic structure calculations
    molecule = run_pyscf_mod(
            guess,
            n_active_orbitals,n_active_electrons,
            MolecularData(geometry, basis, multiplicity, charge)
    )
    # Freeze core orbitals and truncate to active space
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(range(n_core_orbitals,
                                    n_core_orbitals + n_active_orbitals))

    return molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices)


def run_pyscf_mod(
              guess,
              n_active_orbitals,
              n_active_electrons,
              molecule,
              run_scf=True,
              run_mp2=False,
              run_cisd=False,
              run_ccsd=False,
              run_fci=False,
              verbose=False
              ):
    """
    This function runs a pyscf calculation.
    Args:
        molecule: An instance of the MolecularData or PyscfMolecularData class.
        run_scf: Optional boolean to run SCF calculation.
        run_mp2: Optional boolean to run MP2 calculation.
        run_cisd: Optional boolean to run CISD calculation.
        run_ccsd: Optional boolean to run CCSD calculation.
        run_fci: Optional boolean to FCI calculation.
        verbose: Boolean whether to print calculation results to screen.
    Returns:
        molecule: The updated PyscfMolecularData object. Note the attributes
        of the input molecule are also updated in this function.
    """
    # Prepare pyscf molecule.
    pyscf_molecule = prepare_pyscf_molecule_mod(molecule)
    molecule.n_orbitals = int(pyscf_molecule.nao_nr())
    molecule.n_qubits = 2 * molecule.n_orbitals
    molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

    # Run SCF.
    pyscf_scf = compute_scf_mod(pyscf_molecule)
    pyscf_scf.verbose = 0
    pyscf_scf.run(chkfile=config.chk,init_guess=guess, conv_tol=1e-12,conv_tol_grad=1e-12)
    molecule.hf_energy = float(pyscf_scf.e_tot)
    if verbose:
        print('Hartree-Fock energy for {} ({} electrons) is {}.'.format(
            molecule.name, molecule.n_electrons, molecule.hf_energy))

    # Hold pyscf data in molecule. They are required to compute density
    # matrices and other quantities.
    molecule._pyscf_data = pyscf_data = {}
    pyscf_data['mol'] = pyscf_molecule
    pyscf_data['scf'] = pyscf_scf

    # Populate fields.
    molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)

    # Get integrals.
    one_body_integrals, two_body_integrals = compute_integrals_mod(
        pyscf_molecule, pyscf_scf)
    molecule.one_body_integrals = one_body_integrals
    molecule.two_body_integrals = two_body_integrals
    molecule.overlap_integrals = pyscf_scf.get_ovlp()
    # CASCI (FCI)
    if run_fci:
        molecule.fci_energy  = pyscf_scf.CASCI(n_active_orbitals,n_active_electrons).kernel()[0]
    # Return updated molecule instance.
    pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    pyscf_molecular_data.__dict__.update(molecule.__dict__)
    pyscf_molecular_data.save()
    return pyscf_molecular_data


def jordan_wigner0(iop, n_qubits=None):
    """ Function:
    
    """
    # Initialize qubit operator as constant.
    qubit_operator = QubitOperator((), iop.constant)

    # Transform diagonal one-body terms
    for p in range(n_qubits):
        coefficient = iop[(p, 1), (p, 0)]
        qubit_operator += jordan_wigner_one_body0(p, p, coefficient)

    # Transform other one-body terms and "diagonal" two-body terms
    for p, q in itertools.combinations(range(n_qubits), 2):
        # One-body
        coefficient = .5 * (iop[(p, 1), (q, 0)] + iop[(q, 1), (p, 0)].conjugate())
        qubit_operator += jordan_wigner_one_body0(p, q, coefficient)

        # Two-body
        coefficient = (iop[(p, 1), (q, 1), (p, 0), (q, 0)] -
                       iop[(p, 1), (q, 1), (q, 0), (p, 0)] -
                       iop[(q, 1), (p, 1), (p, 0), (q, 0)] +
                       iop[(q, 1), (p, 1), (q, 0), (p, 0)])
        qubit_operator += jordan_wigner_two_body0(p, q, p, q, coefficient)
    return qubit_operator


def jordan_wigner_one_body0(p, q, coefficient=1.):
    """Map the term a^\dagger_p a_q + h.c. to QubitOperator.
    Note that the diagonal terms are divided by a factor of 2
    because they are equal to their own Hermitian conjugate.
    """
    # Handle diagonal terms.
    qubit_operator = QubitOperator()
    if p == q:
        qubit_operator += QubitOperator((), .5 * coefficient)

    return qubit_operator


def jordan_wigner_two_body0(p, q, r, s, coefficient=1.):
    """Map the term a^\dagger_p a^\dagger_q a_r a_s + h.c. to QubitOperator.
    Note that the diagonal terms are divided by a factor of two
    because they are equal to their own Hermitian conjugate.
    """
    # Initialize qubit operator.
    qubit_operator = QubitOperator()

    # Return zero terms.
    if (p == q) or (r == s):
        return qubit_operator

    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:

        # Get coefficient.
        if p == s:
            coeff = -.25 * coefficient
        else:
            coeff = .25 * coefficient

        # Add terms.
        qubit_operator -= QubitOperator((), coeff)

    return qubit_operator



def jordan_wigner_fermion_operator_0(operator):
    transformed_operator = QubitOperator()
    for term in operator.terms:
        # Initialize identity matrix.
        transformed_term = QubitOperator((), operator.terms[term])
        # Loop through operators, transform and multiply.
        for ladder_operator in term:
            #print(ladder_operator[0])
            z_factors = tuple((index, 'Z') for
                              index in range(ladder_operator[0]))
            print(z_factors)
            pauli_x_component = QubitOperator(
                z_factors + ((ladder_operator[0], 'X'),), 0.5)
#                z_factors , 0.5)
            if ladder_operator[1]:
                pauli_y_component = QubitOperator(
                    z_factors + ((ladder_operator[0], 'Y'),), -0.5j)
                    #z_factors , -0.5j)
            else:
                pauli_y_component = QubitOperator(
                    z_factors + ((ladder_operator[0], 'Y'),), 0.5j)
                    #z_factors , 0.5j)
            transformed_term *= pauli_x_component + pauli_y_component
            print(transformed_term,'\n')
        transformed_operator += transformed_term
    return transformed_operator
