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

os.environ["OMP_NUM_THREADS"] = "1"  ### Initial setting
os.environ["MKL_NUM_THREADS"] = "1"  ### Initial setting
os.environ["NUMEXPR_NUM_THREADS"] = "1"  ### Initial setting
################################################################
#                   Setting for input and output               #
################################################################
len_argv = len(sys.argv)
if len_argv == 1:
    print("Error! No input loaded.")
    exit()

# First argument = Input file,  either "***.inp" or "***" is allowed.
input_file = sys.argv[1]
input_name, ext = os.path.splitext(os.path.basename(input_file))
if ext == "":
    ext = ".inp"
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
################################################################

# Method_list
vqe_method_list = [
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
    "opt_uccd",
    "opt_uccsd",
]
# PeriodicTable to check the input atoms are supported.
PeriodicTable = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]


# Determinant and Multi-refernce (State-Average) strings and weights
det = -1
current_det = -1

# molecular data  and operators (pyscf and openfermion)
basis = "sto-3G"  # Gaussian Basis Set
multiplicity = 1  # Spin multiplicity (defined as Ms + 1)
Ms = 0  # Ms = Multiplicity - 1
charge = 0  # Electron charge (0 for neutral)
geometry = None
pyscf_guess = "minao"  # Guess for pyscf: 'minao', 'chkfile'
run_fci = 1
n_active_orbitals = 0
n_active_electrons = 0

# (Hubbard mode is entered when basis = hubbard, 2d-hubbard)
hubbard_u = 1  # Hubbard model (U interaction)
hubbard_nx = 0  # Number of lattices in x direction: not needed if n_orbitals is defined (see below)
hubbard_ny = 1  # Number of lattices in y direction (needed only for 2d-hubbard)

# Spin-Projection
SpinProj = False  # Whether or not to perform Spin Projection
euler_ngrids = [0, -1, 0]
dmm = 0

# Lagrange multiplier for Spin-Constrained Calculation
constraint_lambda = 0

# cycle
icyc = 0

# Time
t_old = 0

# OMP_NUM_THREADS
nthreads = "1"

# Options to compute RDMs
Do1RDM = 0
Do2RDM = 0

# VQE Quantum State(s)
States = None

# qulacs (VQE part)
print_level = 1  # Printing level
mix_level = 0  # Number of pairs of orbitals to be mixed (to break symmetry)
rho = 1  # Trotter number
kappa_guess = "zero"  # Guess for kappa: 'zero', 'read', 'mix', 'random'
theta_guess = "zero"  # Guess for T1 and T2 amplitudes: 'zero', 'read', 'mix', 'random'
Kappa_to_T1 = 0  # Flag to use ***.kappa file (T1-like) for initial guess of T1
spin = (
    -1
)  # Spin quantum number for spin-projection (spin will be Ms + 1 if not specified in input)
DS = 0  # Ordering of T1 and T2: 0 for Exp[T1]Exp[T2], 1 for Exp[T2]Exp[T1]
print_amp_thres = 1e-2  # Threshold for T amplitudes to be printed
fci_energy = 0
hf_energy = 0
excited_states = []
mo_coeff = None
natom = 0
atom_charges = 0
atom_coords = 0
rint = None
Hamiltonian_operator = None
S2_operator = None
Number_operator = None
Dipole_operator = None

# scipy.optimize
opt_method = "L-BFGS-B"  # Method for optimization
gtol = 1e-5  # Convergence criterion based on gradient
ftol = 1e-9  # Convergence criterion based on energy (cost)
eps = 1e-6  # Numerical step
maxiter = 1000  # Maximum iterations: if 0, skip VQE and only JW-transformation is carried out.
maxfun = 10000000000  # Maximum function evaluations. Virtual infinity.
