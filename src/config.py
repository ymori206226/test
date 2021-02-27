"""
#######################
#        quket        #
#######################

config.py

Global arguments are stored in this code.
Default values are set up.

"""

import os
import re
import sys
import time
################################################################
#                   Setting for input and output               #
################################################################
len_argv = len(sys.argv)
if len_argv == 1:
    print("Error! No input loaded.")
    exit()

# First argument = Input file,  either "***.inp" or "***" is allowed.
input_file = sys.argv[1]
input_name = os.path.basename(input_file)
ext = ".inp"
if input_name[-4:] == ext:
    input_name = input_name[:-4]
input_file = input_name + ext    

# Define the names of other useful files
theta_list_file = "./" + input_name + ".theta"
tmp = "./" + input_name + ".tmp"
kappa_list_file = "./" + input_name + ".kappa"
chk = "./" + input_name + ".chk"

# If second argument also exits, that will be your log file name
if len_argv == 3:
    log_name = sys.argv[2]
    log = "./" + log_name
else:
    log_name = input_name
    log = "./" + log_name + ".log"

rdm1 = "./" + input_name + ".1rdm"
#################################################################

## OMP_NUM_THREADS ##############################################
nthreads = "1"
f = open(input_file)
lines = f.readlines()
iline = 0
num_lines = sum(1 for line in open(input_file))
while iline < num_lines:
    line = lines[iline].replace("=", " ")
    line = line.replace(":", " ")
    line = line.replace(",", " ")
    line = line.replace("'", " ")
    line = line.replace("(", " ")
    line = line.replace(")", " ")
    words = [x.strip() for x in line.split() if not line.strip() == ""]
    len_words = len(words)
    if len_words > 0 and words[0][0] not in {"!", "#"}:
        if words[0].lower() == "npar":
            nthreads = words[1]
    iline += 1  # read next line...
f.close()
os.environ["OMP_NUM_THREADS"] = nthreads
#################################################################



# Method and Ansatz list
vqe_ansatz_list = [
    "uhf",
    "phf",
    "suhf",
    "sghf",
    "uccd",
    "uccsd",
    "sauccsd",
    "jmucc",
    "opt_puccd",
    "opt_puccsd",
    "puccsd",
    "opt_psauccd",
    "ic_mrucc"
]
qite_ansatz_list = [
    "exact",
    "inexact",
    "hamiltonian",
    "hamiltonian2",
    "cite",
    "uccsd",
    "uccgsd",
    "upccgsd",
]

# PeriodicTable to check whether the input atoms are supported.
PeriodicTable = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]

# PySCF guess
pyscf_guess = "minao"  # Guess for pyscf: 'minao', 'chkfile'

# Lagrange multiplier for Spin-Constrained Calculation
constraint_lambda = 0

# OMP_NUM_THREADS
nthreads = "1"
approx_exp = False

# qulacs (VQE part)
print_level = 1  # Printing level
print_fci = 0  # Whether fci is printed initially
mix_level = 0  # Number of pairs of orbitals to be mixed (to break symmetry)
kappa_guess = "zero"  # Guess for kappa: 'zero', 'read', 'mix', 'random'
theta_guess = "zero"  # Guess for T1 and T2 amplitudes: 'zero', 'read', 'mix', 'random'
Kappa_to_T1 = 0  # Flag to use ***.kappa file (T1-like) for initial guess of T1
fci_state = None
fci_coeff = None

# scipy.optimize
opt_method = "L-BFGS-B"  # Method for optimization
eps = 1e-6  # Numerical step
maxfun = 10000000000  # Maximum function evaluations. Virtual infinity.
#gtol = 1e-5  # Convergence criterion based on gradient
#ftol = 1e-9  # Convergence criterion based on energy (cost)
#maxiter = 1000  # Maximum iterations: if 0, skip VQE and only JW-transformation is carried out.

##################################
#    System related constants    #
##################################
# cycle
icyc = 0
# Time
t_old = 0
# Debugging
debug = False

### QITE temporary
nterm = 1
nspin = 0
dimension = 1
