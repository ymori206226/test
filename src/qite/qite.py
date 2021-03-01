"""
#######################
#        quket        #
#######################

qite.py

Main driver of QITE.

"""

from .. import config as cf
import numpy as np
from ..fileio import error, prints

from .qite_anti import make_antisymmetric_group, qite_anti
from .qite_function import uccsd, uccgsd, upccgsd
from .qite_exact import qite_exact
from .qite_inexact import qite_inexact


def QITE_driver(Quket):
    prints("Performing QITE for {} Hamiltonian".format(Quket.model))
    prints(
        "Initial configuration: |",
        format(Quket.det, "0" + str(Quket.n_qubits) + "b"),
        ">",
    )
    prints("Convergence criteria:  ftol = {:1.0E} ".format(Quket.ftol))

    if Quket.ansatz == "inexact":
        qite_inexact(
            Quket,
            cf.nterm,
            cf.dimension,
        )
    elif Quket.ansatz == "exact":
        qite_exact(Quket)
    else:
        ### Anti-symmetric group
        if Quket.ansatz == "hamiltonian":
            ansatz_operator = Quket.operators.Hamiltonian
        if Quket.ansatz == "hamiltonian2":
            ansatz_operator = Quket.operators.Hamiltonian
            ansatz_operator *= ansatz_operator
        if Quket.ansatz == "uccsd":
            ansatz_operator = uccsd(Quket.n_orbitals, Quket.det)
        if Quket.ansatz == "uccgsd":
            ansatz_operator = uccgsd(Quket.n_orbitals, Quket.det)
        if Quket.ansatz == "upccgsd":
            ansatz_operator = upccgsd(Quket.n_orbitals, Quket.det)
        if Quket.ansatz == "cite":
            ### Classical ITE
            id_set = []
            size = 0
        else:
            id_set, size = make_antisymmetric_group(
                ansatz_operator,
                Quket.operators.jw_Hamiltonian,
                Quket.model,
                Quket.n_qubits,
                Quket.ansatz,
                Quket.truncate,
            )
        qite_anti(Quket, id_set, size)
