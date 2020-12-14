"""
#######################
#        quket        #
#######################

init.py

Initializing state.

"""
import numpy as np
import time
from qulacs import QuantumState, QuantumCircuit
from qulacs.state import inner_product
from . import config as cf
from . import mpilib as mpi
from .fileio import SaveTheta, print_amplitudes, print_state, prints, printmat

def int2occ(state_int):
    """ Function
    Given an (base-10) integer, find the index for 1 in base-2 (occ_list)

    Author(s): Takashi Tsuchimochi
    """
    occ_list=[]
    k = 0
    while k < state_int:
        kk = 1 << k
        if kk & state_int >0:
            occ_list.append(k)
        k += 1
    return occ_list

def set_initial_det():
    """ Function
    Set the initial wave function to RHF/ROHF determinant. 

    Author(s): Takashi Tsuchimochi
    """
    cf.det = 0
    noa = int((cf.n_active_electrons + cf.multiplicity - 1)/2)
    nob = cf.n_active_electrons - noa 
    for i in range(noa): 
        cf.det = cf.det ^ (1 << 2*i)
    for i in range(nob): 
        cf.det = cf.det ^ (1 << 2*i+1)
    cf.current_det = cf.det


def get_occvir_lists(n_qubit,det):
    """ Function
    Generate occlist and virlist for det (base-10 integer).

    Author(s): Takashi Tsuchimochi
    """
    occ_list    = int2occ(det)
    vir_list    = [i for i in range(n_qubit) if i not in occ_list]
    return occ_list, vir_list




