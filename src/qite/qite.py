"""
#######################
#        quket        #
#######################

qite.py

Main driver of QITE.

"""
from .qite_anti import make_antisymmetric_group, qite_anti
from .qite_function import uccsd_fermi, uccgsd_fermi, upccgsd_fermi
from .qite_exact import qite_exact
from .qite_inexact import qite_inexact
from .. import config as cf
from ..fileio import prints


def QITE_driver(Quket):
    model = Quket.model
    ansatz = Quket.ansatz
    det = Quket.det
    n_orbitals = Quket.n_orbitals
    n_qubits = Quket.n_qubits
    ftol = Quket.ftol
    truncate = Quket.truncate

    opt = f"0{n_qubits}b"
    prints(f"Performing QITE for {model} Hamiltonian")
    prints(f"Initial configuration: | {format(det, opt)} >")
    prints(f"Convergence criteria: ftol = {ftol:1.0E}")

    if ansatz == "inexact":
        qite_inexact(Quket, cf.nterm, cf.dimension)
    elif ansatz == "exact":
        qite_exact(Quket)
    else:
        ### Anti-symmetric group
        if ansatz == "hamiltonian":
            ansatz_operator = Quket.operators.Hamiltonian
        if ansatz == "hamiltonian2":
            ansatz_operator = Quket.operators.Hamiltonian
            ansatz_operator *= ansatz_operator
        if ansatz == "uccsd":
            ansatz_operator = uccsd_fermi(n_orbitals, det)
        if ansatz == "uccgsd":
            ansatz_operator = uccgsd_fermi(n_orbitals, det)
        if ansatz == "upccgsd":
            ansatz_operator = upccgsd_fermi(n_orbitals, det)
        if ansatz == "cite":
            ### Classical ITE
            id_set = []
            size = 0
        else:
            id_set, size \
                    = make_antisymmetric_group(
                            ansatz_operator,
                            Quket.operators.jw_Hamiltonian,
                            model,
                            n_qubits,
                            ansatz,
                            truncate)
        qite_anti(Quket, id_set, size)
