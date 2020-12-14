
"""
#######################
#        quket        #
#######################

prop.py

Properties.

"""
import sys
import time
import numpy as np
from . import config as cf
from . import mpilib as mpi
from .fileio import prints, printmat, SaveTheta
from .opelib  import FermionOperator_to_Observable, FermionOperator_to_Operator
import math
import pprint
import itertools
from openfermion.transforms import jordan_wigner
from openfermion.utils      import hermitian_conjugated
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import QubitOperator,FermionOperator

def dipole(state):
    """ Function
        Prepare the dipole operator and get expectation value.

    Author(s): Takashi Tsuchimochi
    """
    # Prepare Operators for Qulacs
    qulacs_dipole_x = FermionOperator_to_Observable(cf.Dipole_operator[0])
    qulacs_dipole_y = FermionOperator_to_Observable(cf.Dipole_operator[1])
    qulacs_dipole_z = FermionOperator_to_Observable(cf.Dipole_operator[2])
    dx = - qulacs_dipole_x.get_expectation_value(state) 
    dy = - qulacs_dipole_y.get_expectation_value(state) 
    dz = - qulacs_dipole_z.get_expectation_value(state) 
    d = [dx, dy, dz]
    for i in range(cf.natom):
        d += cf.atom_charges[i] * cf.atom_coords[i]
    
    d = d / 0.393456

    prints("\nDipole moment (in Debye) :")
    prints("x = {x:.5f}     y = {y:.5f}    z = {z:.5f}".format(x=d[0], y=d[1], z=d[2]))
    prints("| mu | = {:.5f}".format(np.sqrt(d[0]**2+d[1]**2+d[2]**2)))
        

def get_1RDM(state,print_level=1):
    """ Function
    Compute 1RDM of QuantmState `state`.

    Author(s): Taisei Nishimaki, Takashi Tsuchimochi
    """
    prints("\n === Computing 1RDM === ")
    n_qubit = state.get_qubit_count() 
    comp = 0+1.0j
    norbs = int(n_qubit/2)
    Daa = np.zeros((norbs,norbs))
    Dbb = np.zeros((norbs,norbs))
    Daa_mat = np.zeros((norbs,norbs))
    Dbb_mat = np.zeros((norbs,norbs))
    # MPI parallel
    ipos, my_n_qubit = mpi.myrange(n_qubit)
    #print("ipos = ",ipos,"  my_n_qubit = ",my_n_qubit, "   my_rank = ",mpi.rank)
    for i in range(ipos,ipos+my_n_qubit):
        #print("my_rank = ",mpi.rank,"   working on ",i)
        for j in range(n_qubit):
            ii = int(i/2)
            jj = int(j/2)
            #Fermionoperatorからjw変換、qulacsで読み込める形に変換して期待値を表示
            string = str(i)+'^ '+str(j)
            Epq = FermionOperator(string)
            Epq_qu = FermionOperator_to_Operator(Epq)
            Epq_expect = Epq_qu.get_expectation_value(state).real
            Dpq = Epq_expect
            if(i%2==0 and j%2==0):
                Daa[jj][ii] = Dpq
            elif(i%2==1 and j%2==1):
                Dbb[jj][ii] = Dpq
    mpi.comm.Allreduce(Daa, Daa_mat, mpi.MPI.SUM)
    mpi.comm.Allreduce(Dbb, Dbb_mat, mpi.MPI.SUM)
    if print_level > 0:
        printmat(Daa_mat,name="Daa")
        printmat(Dbb_mat,name="Dbb")
    SaveTheta(norbs**2,Daa_mat.ravel(),cf.rdm1,opentype='w')
    SaveTheta(norbs**2,Dbb_mat.ravel(),cf.rdm1,opentype='a')
    return Daa_mat,Dbb_mat
