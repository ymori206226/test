import os
import re
import string

from . import config as cf
from .fileio import error, prints
from .utils import chkbool#, chkmethod

#############################
###    Default options    ###
#############################
# How to write; 'option's name': 'attribute's name'
integers = {
        #----- For Config -----
        "print_level": "print_level",
        "mix_level": "mix_level",
        "kappa_to_t1": "Kappa_to_T1",
        "nterm": "nterm",
        "dimension": "dimension",
        #----- For General -----
        "n_electrons": "n_active_electrons",
        "n_orbitals": "n_active_orbitals",
        "multiplicity": "multiplicity",
        "charge": "charge",
        "rho": "rho",
        "mix_level": "mix_level",
        "maxiter": "maxiter",
        "hubbard_nx": "hubbard_nx",
        "hubbard_ny": "hubbard_ny",
        #----- For VQE -----
        "ds": "DS",
        #----- For Symmetry-Projection -----
        "spin": "spin",
        }
floats = {
        #----- For Config -----
        "eps": "ops",
        "lambda": "constraint_lambda",
        #----- For General -----
        "hubbard_u": "hubbard_u",
        "gtol": "gtol",
        "ftol": "ftol",
        "print_amp_thres": "print_amp_thres",
        #----- For QITE -----
        "timestep": "dt", "db": "dt", "dt": "dt",
        "truncate": "truncate",
        #----- For Adapt-VQE -----
        "adapt_eps": "eps",
        "adapt_max": "max"
        }
bools = {
        #----- For Config -----
        "print_fci": "print_fci",
        "approx_exp": "approx_exp",
        "debug": "debug",
        #----- For General -----
        "run_fci": "run_fci",
        #----- For VQE -----
        "1rdm": "Do1RDM",
        #----- Symmetry-Projection -----
        "spinproj": "SpinProj",
        #----- For Multi/Excited-state -----
        "act2act": "act2act_opt",
        }
strings = {
        #----- For Config -----
        "opt_method": "opt_method",
        "pyscf_guess": "pyscf_guess",
        "kappa_guess": "kappa_guess",
        "theta_guess": "theta_guess",
        "npar": "npar",
        #----- For General -----
        "method": "method",
        "ansatz": "ansatz",
        }


def read_input():
    """Function:
    Open ***.inp and read options.
    The read options are stored as global variables in config.py.

    Return:
        List: Whether input is read to EOF

    Notes:
        Here is the list of allowed options:

        method (str):           Name of method.
        multiplicity (int):     Spin-multiplicity
        charge (int):           Charge
        rho (int):              Trotter-steps
        run_fci (bool):         Whether to run fci with pyscf
        print_level (int):      Pritinging level
        opt_method (str):       Optimization method for VQE
        mix_level (int):        Number of orbitals to break spin-symmetry
        eps (float):            Step-size for numerical gradients
        gtol (float):           Convergence criterion of VQE (grad)
        ftol (float):           Convergence criterion of VQE (energy)
        print_amp_thres (int):  Printing level of amplitudes (theta_list, etc)
        maxiter (int):          Maximum iteration number of VQE
        pyscf_guess (str):      Guess for pyscf calculation (miao, read)
        kappa_guess (str):      Guess for initial kappa (zero, mix, random, read)
        theta_guess (str):      Guess for initial theta (zero, random, read)
        1rdm (bool):            Whether to compute 1RDM
        kappa_to_t1 (bool):     Whether to use kappa_list for T1 amplitudes.
        n_electrons (int):      Number of electrons
        n_orbitals (int):       Number of orbitals (x2 = number of qubits)
        spinproj (bool):        Whether to perform spin-projection
        spin (int):             Target spin state of spin-projection
        euler (ints):           Euler angles for spin-projection
        ds (bool):              Ordering of Singles/Doubles. If True,
                                Exp[T2][T1], if False, Exp[T1][T2]
        lambda (float):         Lagrange multiplier for spin-constrained calculation
        geometry (dict):        Standard cartesian geometry
        det, determinant (int): Initial determinant (like 000111)
        multi (ints):           Determinants for multi-determinantal calculation
        excited (ints):         Initial determinants for excited calculations
        npar, nthreads (int):   Number of threads
        hubbard_u (float):      Hubbard U
        hubbard_nx (int):       Hubbard lattice size for x-direction
        hubbard_ny (int):       Hubbard lattice size for y-direction

        Putting '@@@' in lines separates multiple jobs in one input file.
        (the options from previous jobs will be used unless redefined)
    """
    # Read input lines.
    with open(cf.input_file, "r") as f:
        lines = f.readlines()

    def get_line(line):
        # Removal of certain symbols and convert multi-space to single-space.
        repstr = ":,'()"
        line = line.translate(str.maketrans(repstr, " "*len(repstr), ""))
        line = re.sub(r" +", r" ", line).rstrip("\n").strip()
        if len(line) == 0:
            # Ignore blank line.
            return
        if line[0] in ["!", "#"]:
            # Ignore comment line.
            return

        if "=" not in line:
            return line

        key, value = line.split("=")
        key = key.strip().lower()
        value = value.strip()
        value = value.split(" ")
        if len(value) == 1:
            return key, value[0]
        else:
            return key, value

    ######################################
    ###    Start reading input file    ###
    ######################################
    kwds_list = []
    kwds = {}
    # Values forced to be initialized
    cf.lower_states = []
    # Add 'data_directory' for 'chemical'.
    kwds["data_directory"] = cf.input_dir
    i = 0
    while i < len(lines):
        line = get_line(lines[i])
        if isinstance(line, str):
            key = line.strip()
            value = ""
        elif isinstance(line, tuple):
            key, value = line
        else:
            i += 1
            continue

        ################
        # Easy setting #
        ################
        if key in integers:
            kwds[integers[key]] = int(value)
        elif key in floats:
            kwds[floats[key]] = float(value)
        elif key in bools:
            kwds[bools[key]] = chkbool(value)
        elif key in strings:
            kwds[strings[key]] = value
        ###########
        # General #
        ###########
        elif key == "basis":
            if len(value.split(" ")) == 1:
                # e.g.) basis = sto-3g
                kwds[key] = value
            else:
                # e.g.) basis = H sto-3g, O 6-31g
                atoms = value[::2]
                atom_basis = value[1::2]
                kwds[key] = dict(zip(atoms, atom_basis))
        elif key == "geometry":
            # e.g.) geometry:
            #       H 0 0 0
            #       H 0 0 0.74
            geom = []
            for j in range(i+1, len(lines)):
                next_line = get_line(lines[j])
                if not isinstance(next_line, str):
                    break
                atom_info = next_line.split(" ")
                if len(atom_info) != 4:
                    break
                atom = atom_info[0]
                xyz = tuple(map(float, atom_info[1:4]))
                if atom in cf.PeriodicTable:
                    geom.append((atom, xyz))
            # Set geometry and skip molecule's information lines.
            kwds[key] = geom
            i = j - 1
        elif key in ["det", "determinant"]:
            # e.g.) 000011
            if value.isdecimal() and int(value) >= 0:
                kwds["det"] = int(f"0b{value}", 2)
            else:
                prints(f"Invalid determinant description '{value}'")
        #######################
        # Symmetry-Projection #
        #######################
        elif key == "euler":
            # e.g.) euler = -1
            # e.g.) euler = 1 1
            # e.g.) euler = 1 -2 3
            x = 0
            y = -1
            z = 0
            if len(value) == 1:
                y = int(value)
            elif len(value) == 2:
                x, y = map(int, value)
            elif len(value) == 3:
                x, y, z = map(int, value)
            else:
                error("Format for Euler angle is wrong")
            kwds["euler_ngrids"] = [x, y, z]
        elif key == "nproj":
            kwds["number_ngrids"] = int(value)
            kwds["NumberProj"] = cf.number_ngrids > 1
        ######################
        # Multi/Excite-state #
        ######################
        elif key == "multi":
            # e.g.) multi:
            #       000011 0.25
            #       000110 0.25
            #       001001 0.25
            #       001100 0.25
            states = []
            weights = []
            for j in range(i+1, len(lines)):
                next_line = get_line(lines[j])
                if not isinstance(next_line, str):
                    break
                if len(next_line.split(" ")) != 2:
                    break
                state, weight = next_line.split(" ")
                states.append(int(f"0b{state}", 2))
                weights.append(float(weight))
            # Set states/weights and skip multi's information lines.
            kwds["states"] = states
            kwds["weights"] = weights
            i = j - 1
        elif key == "excited":
            # e.g.) excited:
            #       000110
            #       001001
            #       001100
            excited_states = []
            for j in range(i+1, len(lines)):
                next_line = get_line(lines[j])
                if not isinstance(next_line, str):
                    break
                state = next_line.strip()
                excited_states.append(int(f"0b{state}", 2))
            # Set excited_states and skip excited_states' information lines.
            kwds["excited_states"] = excited_states
            i = j - 1
        #############
        # Otherwise #
        #############
        elif key == "@@@":
            # Go next job input.
            # Set current 'kwds' to 'kwds_list'.
            # Note; 'kwds' may be overwritten at next job
            #       for the sake of simplicity in the input file.
            kwds_list.append(kwds.copy())
        else:
            if value != "":
                error(f"No option '{key}'")

        # Go next line
        i += 1

    kwds_list.append(kwds)
    return kwds_list


def set_config(kwds, Quket):
    for key, value in kwds.items():
        #----- For General -----
        if key == "print_level":
            cf.print_level = int(value)
        elif key == "mix_level":
            cf.mix_level = int(value)
        elif key == "eps":
            cf.eps = float(value)
        elif key == "print_fci":
            cf.print_fci = chkbool(value)
        elif key == "opt_method":
            cf.opt_method = value
        elif key == "pyscf_guess":
            cf.pyscf_guess = "chkfile" if value == "read" else value
        #----- For VQE -----
        elif key == "Kappa_to_T1":
            cf.Kappa_to_T1 = int(value)
        elif key == "constraint_lambda":
            cf.constraint_lambda = float(value)
        elif key == "approx_exp":
            cf.approx_exp = chkbool(value)
        elif key == "kappa_guess":
            cf.kappa_guess = value
        elif key == "theta_guess":
            cf.theta_guess = value
        #----- For QITE -----
        elif key == "nterm":
            cf.nterm = int(value)
        elif key == "dimension":
            cf.dimension = int(value)
        #----- For Sysmte -----
        elif key == "debug":
            cf.debug = chkbool(value)
        elif key == "nthreads":
            cf.nthreads = value
            os.environ["OMP_NUM_THREADS"] = value

    if cf.opt_method == "L-BFGS-B":
        cf.opt_options = {"disp": True,
                          "maxiter": Quket.maxiter,
                          "gtol": Quket.gtol,
                          "ftol": Quket.ftol,
                          "eps": cf.eps,
                          "maxfun": cf.maxfun}
    elif opt_method == "BFGS":
        cf.opt_options = {"disp": True,
                          "maxiter": Quket.maxiter,
                          "gtol": Quket.gtol,
                          "eps": cf.eps}
