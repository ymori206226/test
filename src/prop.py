"""
#######################
#        quket        #
#######################

prop.py

Properties.

"""
import numpy as np
from . import config as cf
from . import mpilib as mpi
from .fileio import prints, printmat, SaveTheta
from .opelib import FermionOperator_to_Observable, FermionOperator_to_Operator
from openfermion.ops import FermionOperator


def dipole(QuketData):
    """Function
        Prepare the dipole operator and get expectation value.

    Author(s): Takashi Tsuchimochi
    """
    # Prepare Operators for Qulacs
    qulacs_dipole_x = FermionOperator_to_Observable(QuketData.operators.Dipole_operator[0],QuketData.n_qubits)
    qulacs_dipole_y = FermionOperator_to_Observable(QuketData.operators.Dipole_operator[1],QuketData.n_qubits)
    qulacs_dipole_z = FermionOperator_to_Observable(QuketData.operators.Dipole_operator[2],QuketData.n_qubits)
    dx = -qulacs_dipole_x.get_expectation_value(QuketData.state)
    dy = -qulacs_dipole_y.get_expectation_value(QuketData.state)
    dz = -qulacs_dipole_z.get_expectation_value(QuketData.state)
    d = [dx, dy, dz]
    d += QuketData.atom_charges*QuketData.atom_coords
    #for i in range(QuketData.natom):
    #    d += QuketData.atom_charges[i] * QuketData.atom_coords[i]

    d = d / 0.393456

    prints("\nDipole moment (in Debye) :")
    prints("x = {x:.5f}     y = {y:.5f}    z = {z:.5f}".format(x=d[0], y=d[1], z=d[2]))
    prints("| mu | = {:.5f}".format(np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)))


def get_1RDM(QuketData, print_level=1):
    """Function
    Compute 1RDM of QuantmState `state` in QuketData.

    Author(s): Taisei Nishimaki, Takashi Tsuchimochi
    """
    prints("\n === Computing 1RDM === ")
    n_qubits = QuketData.n_qubits
    norbs = QuketData.n_orbitals
    Daa = np.zeros((norbs, norbs))
    Dbb = np.zeros((norbs, norbs))
    Daa_mat = np.zeros((norbs, norbs))
    Dbb_mat = np.zeros((norbs, norbs))
    # MPI parallel
    ipos, my_n_qubits = mpi.myrange(n_qubits)
    # print("ipos = ",ipos,"  my_n_qubits = ",my_n_qubits, "   my_rank = ",mpi.rank)
    for i in range(ipos, ipos + my_n_qubits):
        # print("my_rank = ",mpi.rank,"   working on ",i)
        for j in range(n_qubits):
            ii = int(i / 2)
            jj = int(j / 2)
            # Fermionoperatorからjw変換、qulacsで読み込める形に変換して期待値を表示
            string = str(i) + "^ " + str(j)
            Epq = FermionOperator(string)
            Epq_qu = FermionOperator_to_Operator(Epq, n_qubits)
            Epq_expect = Epq_qu.get_expectation_value(QuketData.state).real
            Dpq = Epq_expect
            if i % 2 == 0 and j % 2 == 0:
                Daa[jj][ii] = Dpq
            elif i % 2 == 1 and j % 2 == 1:
                Dbb[jj][ii] = Dpq
    mpi.comm.Allreduce(Daa, Daa_mat, mpi.MPI.SUM)
    mpi.comm.Allreduce(Dbb, Dbb_mat, mpi.MPI.SUM)
    if print_level > 0:
        printmat(Daa_mat, name="Daa")
        printmat(Dbb_mat, name="Dbb")
    SaveTheta(norbs ** 2, Daa_mat.ravel(), cf.rdm1, opentype="w")
    SaveTheta(norbs ** 2, Dbb_mat.ravel(), cf.rdm1, opentype="a")
    return Daa_mat, Dbb_mat
