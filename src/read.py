import os
import re
import string

from . import config as cf
from .fileio import error, prints
from .utils import chkbool, chkmethod

#############################
###    Default options    ###
#############################


def read_input(Quket, job_no):
    """Function:

    Open ***.inp and read options.
    The read options are stored as global variables in config.py.

    Args:
        job_no (int):  Job number. Read input by job_no '@@@'

    Return:
        bool: Whether input is read to EOF

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
    job_k = 0
    with open(cf.input_file):
        lines = f.readlines()

    # Values forced to be initialized
    cf.lower_states = []

    def get_line(line):
        # Removal of certain symbols and convert multi-space to single-space.
        repstr = ":,'()"
        line = lines[i].translate(str.marketrans(repstr, " "*len(repstr), ""))
        line = re.sub(r" +", r" ", line)
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
    kwds = {}
    for i in range(len(lines)):
        line = get_line(lines[i])
        if isinstance(line, str):
            key = line
        elif isinstance(line, tuple):
            key, value = line
        else:
            continue


        # How to write; 'option's name': 'attribute's name'
        integers = {
                #----- For General -----
                "n_electrons": "n_electrons",
                "n_orbitals": "n_orbitals",
                "multiplicity": "multiplicity",
                "charge": "charge",
                "rho": "rho",
                "mix_level": "mix_level",
                "maxiter": "maxiter",
                "hubbard_nx": "hhubbard_nx",
                "hubbard_ny": "hhubbard_ny",
                #----- For VQE -----
                "ds": "DS",
                #----- For Symmetry-Projection -----
                "spin": "spin",
                }
        floats = {
                #----- For General -----
                "hubbard_u": "hubbard_u",
                "gtol": "gtol",
                "ftol": "ftol",
                "print_amp_thres": "print_amp_thres",
                #-----For QITE -----
                "timestep": "dt", "db": "dt", "dt": "dt",
                "truncate": "truncate",
                }
        bools = {
                #----- For General -----
                "run_fci": "run_fci",
                #----- For VQE -----
                "1rdm": "Do1RDM",
                #----- Symmetry-Projection -----
                "spinproj": "SpinProj",
                #----- For Multi/Excited-state -----
                "act2act": "act2act_ops",
                }
        strings = {
                #----- For General -----
                "method": "method",
                "ansatz": "ansatz",
                }

        ##############
        # for config #
        ##############
        #----- For General -----
        if key == "print_level":
            cf.print_level = int(value)
        elif key == "opt_method":
            cf.opt_method = value
        elif key == "mix_level":
            cf.mix_level = int(value)
        elif key == "eps":
            cf.eps = float(value)
        elif key == "pyscf_guess":
            cf.pyscf_guess = value
        elif key == "print_fci":
            cf.print_fci = chkbool(value)
        #----- For VQE -----
        elif key == "kappa_guess":
            cf.kappa_guess = value
        elif key == "theta_guess":
            cf.theta_guess = value
        elif key == "kappa_to_t1":
            cf.Kappa_to_T1 = int(value)
        elif key == "lambda":
            cf.constraint_lambda = float(value)
        elif key == "approx_exp":
            cf.approx_exp = chkbool(value)
        #----- For QITE -----
        elif key == "nterm":
            cf.nterm = int(value)
        elif key == "dimension":
            cf.dimension = int(value)
        #----- For Sysmte -----
        elif key == "npar":
            cf.nthreads = value
            os.environ["OMP_NUM_THREADS"] = value
        elif key == "debug":
            cf.debug = chkbool(value)
        ################
        # Easy setting #
        ################
        elif key in integers:
            kwds[key] = int(value)
        elif key in floats:
            kwds[key] = float(value)
        elif key in bools:
            kwds[key] = chkbool(value)
        elif key in strings:
            kwds[key] = value
        ###########
        # General #
        ###########
        elif key == "basis":
            if len(value) == 1:
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
            if job_k == job_no - 1:
                cf._geom_update = True

            geom = []
            for j in range(i+1, len(lines)):
                next_line = get_line(lines[j])
                if not isinstance(next_line, str):
                    break
                atom_info = next_line.split(" ")
                atom = atom_info[0]
                xyz = tuple(atom_info[1:4])
                if atom in cf.PeriodicTable:
                    geom.append((atom, xyz))
            # Set geometry and update line number.
            kwds[key] == geom
            i = j
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
                x = int(value[0])
                y = int(value[1])
            elif len(value) == 3:
                x = int(value[0])
                y = int(value[1])
                z = int(value[2])
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
                state, weight = next_line.split(" ")
                states.append(int(f"0b{state}", 2))
                weights.append(float(weight))
            # Set states/weights and update line number.
            kwds["states"] = states
            kwds["weights"] = weights
            i = j
        elif key == "excited":
            # e.g.) excited:
            #       000110
            #       001001
            #       001100
            excited_states = []
            for j in range(i+1, len(lines)):
                next_line = get_line(lines[i])
                if not isinstance(next_line, str):
                    break
                state = next_line.strip()
                excited_states.append(int(f"0b{state}", 2))
            # Set excited_states and update line number.
            kwds["excited_states"] = excited_states
            i = j
        ##############
        # End of Job #
        ##############
        elif key == "@@@":
            job_k += 1
            if job_k == job_no:
                Finish = False
                break
            else:
                error(f"No option '{value}'")

    if "method" not in kwds or "ansatz" not in kwds:
        error(f"Unspecified method or ansatz.")
    if not chkmethod(kwds["method"], kwds["ansatz"]):
        error(f"No methhod option {kwds['method']} "
              f"with {kwds['ansatz']} available.")

    # Add 'data_directory' for 'chemical'.
    kwds["data_directory"] = cf.input_dir

    return Finish, kwds


#    num_lines = sum(
#        1 for line in open(cf.input_file)
#    )  # number of lines in the input file
#    job_k = 0
#    finish = true
#    f = open(cf.input_file)
#    lines = f.readlines()
#    iline = 0
#    while iline < num_lines:
#        line = lines[iline].replace("=", " ")
#        line = line.replace(":", " ")
#        line = line.replace(",", " ")
#        line = line.replace("'", " ")
#        line = line.replace("(", " ")
#        line = line.replace(")", " ")
#        words = [x.strip() for x in line.split() if not line.strip() == ""]
#        len_words = len(words)
#        if len_words > 0 and words[0][0] not in ["!", "#"]:
#### general
#            if words[0].lower() == "basis":
#                if len_words > 2:
#                    atoms = []
#                    atom_basis = []
#                    for i in range(0, int(len_words / 2)):
#                        atoms.append(words[2 * i + 1])
#                        atom_basis.append(words[2 * i + 2])
#                    quket.basis = dict(zip(atoms, atom_basis))
#                else:
#                    quket.basis = words[1].lower()
#            elif words[0].lower() == "geometry":
#                # reading Geometry ...
#                if job_k == job_no - 1:
#                    cf._geom_update = True
#
#                iatom = 0
#                quket.geometry = []
#                while iatom >= 0:
#                    iline += 1
#                    if iline == num_lines:
#                        return True
#                    geom = lines[iline].replace("=", " ")
#                    geom = [x.strip() for x in geom.split() if not geom.strip() == ""]
#
#                    if len(geom) > 0:
#                        if geom[0] in cf.PeriodicTable:
#                            xyz = []
#                            xyz.append(float(geom[1]))
#                            xyz.append(float(geom[2]))
#                            xyz.append(float(geom[3]))
#                            atom = []
#                            atom.append(geom[0])
#                            atom.append(xyz)
#                            Quket.geometry.append(atom)
#                        else:
#                            iatom = -1
#                            iline += -1
#                    else:
#                        iatom = -1
#                        iline += -1
#            elif words[0].lower() == "hubbard_u":
#                quket.hubbard_u = float(words[1])
#            elif words[0].lower() == "hubbard_nx":
#                quket.hubbard_nx = int(words[1])
#            elif words[0].lower() == "hubbard_ny":
#                quket.hubbard_ny = int(words[1])
#            elif words[0].lower() == "method":
#                quket.method = words[1]
#            elif words[0].lower() == "ansatz":
#                quket.ansatz = words[1]
#            elif words[0].lower() == "n_electrons":
#                quket.n_electrons = int(words[1])
#            elif words[0].lower() == "n_orbitals":
#                quket.n_orbitals = int(words[1])
#            elif words[0].lower() == "multiplicity":
#                quket.multiplicity = int(words[1])
#            elif words[0].lower() == "charge":
#                quket.charge = int(words[1])
#            elif words[0].lower() in ("det", "determinant"):
#                if words[1].isdecimal() and float(words[1]) >= 0:
#                    base2 = str("0b" + words[1])
#                    base10 = int(base2, 2)
#                    quket.det = base10
#                    # cf.current_det = base10
#                else:
#                    prints("Invalid determinant description '{}' ".fortmat(words[1]))
#            elif words[0].lower() == "rho":
#                quket.rho = int(words[1])
#            elif words[0].lower() == "run_fci":
#                quket.run_fci = int(words[1])
#            elif words[0].lower() == "print_level":
#                cf.print_level = int(words[1])
#            elif words[0].lower() == "opt_method":
#                cf.opt_method = words[1]
#            elif words[0].lower() == "mix_level":
#                cf.mix_level = int(words[1])
#            elif words[0].lower() == "eps":
#                cf.eps = float(words[1])
#            elif words[0].lower() == "gtol":
#                quket.gtol = float(words[1])
#            elif words[0].lower() == "ftol":
#                quket.ftol = float(words[1])
#            elif words[0].lower() == "print_amp_thres":
#                quket.print_amp_thres = float(words[1])
#            elif words[0].lower() == "maxiter":
#                quket.maxiter = int(words[1])
#            elif words[0].lower() == "pyscf_guess":
#                cf.pyscf_guess = words[1]
#            elif words[0].lower() == "print_fci":
#                cf.print_fci = chkbool(words[1])
#### vqe related
#            elif words[0].lower() == "kappa_guess":
#                cf.kappa_guess = words[1]
#            elif words[0].lower() == "theta_guess":
#                cf.theta_guess = words[1]
#            elif words[0].lower() == "1rdm":
#                quket.do1RDM = chkbool(words[1])
#            elif words[0].lower() == "kappa_to_t1":
#                cf.kappa_to_T1 = int(words[1])
#            elif words[0].lower() == "ds":
#                quket.ds = int(words[1])
#            elif words[0].lower() == "lambda":
#                cf.constraint_lambda = float(words[1])
#            elif words[0].lower() == "approx_exp":
#                cf.approx_exp = chkbool(words[1])
#### symmetry-projection
#            elif words[0].lower() == "spinproj":
#                quket.projection.SpinProj = chkbool(words[1])
#            elif words[0].lower() == "spin":
#                quket.projection.spin = int(words[1])
#            elif words[0].lower() == "euler":
#                if len_words == 2:
#                    # only beta
#                    quket.projection.euler_ngrids[1] = int(words[1])
#                elif len_words == 3:
#                    # only alpha and beta
#                    quket.projection.euler_ngrids[0] = int(words[1])
#                    quket.projection.euler_ngrids[1] = int(words[2])
#                elif len_words == 4:
#                    # alpha, beta, gamma
#                    quket.projection.euler_ngrids[0] = int(words[1])
#                    quket.projection.euler_ngrids[1] = int(words[2])
#                    quket.projection.euler_ngrids[2] = int(words[3])
#                else:
#                    error("Format for Euler angle is wrong")
#            elif words[0].lower() == "nproj":
#                quket.projection.number_ngrids = int(words[1])
#                quket.projection.NumberProj =  cf.number_ngrids > 1
#### multi/excited-state calculation section
#            elif words[0].lower() == "multi":
#                # reading determinants and weights ...
#                quket.multi.states = []
#                quket.multi.weights = []
#                ilabel = 0
#                while ilabel >= 0:
#                    iline += 1
#                    if iline == num_lines:
#                        return True
#                    multi = lines[iline].replace("=", " ")
#                    multi = [
#                        x.strip() for x in multi.split() if not multi.strip() == ""
#                    ]
#                    if len(multi) > 0:
#                        if multi[0].isdecimal() and float(multi[1]) >= 0:
#                            # Transform string to integer ...
#                            base2 = str("0b" + multi[0])
#                            base10 = int(base2, 2)
#                            Quket.multi.states.append(base10)
#                            Quket.multi.weights.append(float(multi[1]))
#                        else:
#                            ilabel = -1
#                            iline += -1
#                    else:
#                        ilabel = -1
#                        iline += -1
#
#            elif words[0].lower() == "excited":
#                # reading determinants ...
#                quket.excited_states = []
#                ilabel = 0
#                while ilabel >= 0:
#                    iline += 1
#                    if iline == num_lines:
#                        return True
#                    excited = lines[iline].replace("=", " ")
#                    excited = [
#                        x.strip() for x in excited.split() if not excited.strip() == ""
#                    ]
#                    if len(excited) > 0:
#                        if excited[0].isdecimal():
#                            # Transform string to integer ...
#                            base2 = str("0b" + excited[0])
#                            base10 = int(base2, 2)
#                            Quket.excited_states.append(base10)
#                        else:
#                            ilabel = -1
#                            iline += -1
#                    else:
#                        ilabel = -1
#                        iline += -1
#
#            elif words[0].lower() == "act2act":
#                Quket.multi.act2act_ops = chkbool(words[1])
#
#### QITE
#            elif words[0].lower() in ("timestep","db","dt"):
#                Quket.dt = float(words[1])
#            elif words[0].lower() in ("truncate"):
#                Quket.truncate = float(words[1])
#            # Temporary
#            elif words[0].lower() == "nterm":
#                cf.nterm = int(words[1])
#            elif words[0].lower() == "dimension":
#                cf.dimension = int(words[1])
#### System related
#            elif words[0].lower() == "npar":
#                cf.nthreads = words[1]
#            elif words[0].lower() == "debug":
#                cf.debug = chkbool(words[1])
#### End of Job
#            elif words[0].lower() == "@@@":
#                job_k += 1
#                if job_k == job_no:
#                    Finish = False
#                    break
#            else:
#                error('No option "%s"' % words[0])
#
#        iline += 1  # read next line...
#
#    f.close()
#
#    if not chkmethod(Quket.method, Quket.ansatz):
#        error("No method option {} with {} available.".format(Quket.method, Quket.ansatz))
#
#    return Finish
