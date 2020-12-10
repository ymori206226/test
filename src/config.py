"""
#######################
#        quket        #
#######################

config.py

Global arguments are stored in this code.

"""
import re
import sys
import os
import time

################################################################
#                   Setting for input and output               #
################################################################
len_argv = len(sys.argv)
if len_argv == 1:
   print("Error! No input loaded.")
   exit()

# First argument = Input file,  either "***.inp" or "***" is allowed.
input_file=sys.argv[1]
input_name,ext = os.path.splitext( os.path.basename(input_file))
if ext == '':
    ext = '.inp'
input_file = input_name+ext

# Define the names of other useful files 
theta_list_file = './' + input_name + '.theta'
tmp  = './' + input_name + '.tmp'
kappa_list_file = './' + input_name + '.kappa'
chk = './' + input_name + '.chk'

# If second argument also exits, that will be your log file name
if len_argv == 3:
    log_name=sys.argv[2]
    log = './' + log_name
else:
    log_name = input_name
    log = './' + log_name + '.log'

rdm1 = './' + input_name + '.1rdm'
################################################################

# PeriodicTable to check the input atoms are supported. 
PeriodicTable = ["H","He","Li","Be","B","C","N","O","F","Ne"]


# Multi-refernce (State-Average) strings and weights
multi_states  = []
multi_weights = []

# molecular data  and operators
mo_coeff = 0
natom = 0
atom_charges = [] 
atom_coords = []
n_active_orbitals = 0
n_active_electrons = 0

rint = 0
rint_mo = 0
Hamiltonian_operator = 0
Number_operator = 0
S2_operator = 0
Dipole_operator = []
jw_s4 = 0

# Lagrange multiplier for Spin-Constrained Calculation
constraint_lambda = 0

# cycle
icyc = 0

# Time
t_old = 0

# OMP_NUM_THREADS
nthreads="1"

# Options to compute RDMs
Do1RDM = 0
Do2RDM = 0

# VQE Quantum State(s)
States = None
