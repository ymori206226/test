import os
from . import config as cf
from .fileio import error, prints
from .utils import chkbool, chkmethod

#############################
###    Default options    ###
#############################


def read_input(job_no):
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
        ds (bool):              Ordering of Singles/Doubles. If True, Exp[T2][T1], if False, Exp[T1][T2]
        lambda (float):         Lagrange multiplier for spin-constrained calculation
        geometry (dict):        Standard cartesian geometry
        det, determinant (int): Initial determinant (like 000111)
        multi (ints):           Determinants for multi-determinantal calculation
        excited (ints):         Initial determinants for excited calculations
        npar, nthreads (int):   Number of threads
        hubbard_u (float):      Hubbard U
        hubbard_nx (int):       Hubbard lattice size for x-direction
        hubbard_ny (int):       Hubbard lattice size for y-direction

        act2act (bool):         For ic-MRUCC, act-to-act

        Putting '@@@' in lines separates multiple jobs in one input file.
        (the options from previous jobs will be used unless redefined)

    """

    # Values forced to be initialized
    cf.lower_states = []

    ######################################
    ###    Start reading input file    ###
    ######################################
    num_lines = sum(
        1 for line in open(cf.input_file)
    )  # number of lines in the input file
    job_k = 0
    Finish = True
    f = open(cf.input_file)
    lines = f.readlines()
    iline = 0
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
            if words[0].lower() == "basis":
                if len_words > 2:
                    atoms = []
                    atom_basis = []
                    for i in range(0, int(len_words / 2)):
                        atoms.append(words[2 * i + 1])
                        atom_basis.append(words[2 * i + 2])
                    cf.basis = dict(zip(atoms, atom_basis))
                else:
                    cf.basis = words[1].lower()
            elif words[0].lower() == "method":
                cf.method = words[1]
                if not chkmethod(cf.method):
                    error("No method option {} available.".format(cf.method))
            elif words[0].lower() == "multiplicity":
                cf.multiplicity = int(words[1])
                cf.Ms = cf.multiplicity - 1
            elif words[0].lower() == "charge":
                cf.charge = int(words[1])
            elif words[0].lower() == "rho":
                cf.rho = int(words[1])
            elif words[0].lower() == "run_fci":
                cf.run_fci = int(words[1])
            elif words[0].lower() == "print_level":
                cf.print_level = int(words[1])
            elif words[0].lower() == "opt_method":
                cf.opt_method = words[1]
            elif words[0].lower() == "mix_level":
                cf.mix_level = int(words[1])
            elif words[0].lower() == "eps":
                cf.eps = float(words[1])
            elif words[0].lower() == "gtol":
                cf.gtol = float(words[1])
            elif words[0].lower() == "ftol":
                cf.ftol = float(words[1])
            elif words[0].lower() == "print_amp_thres":
                cf.print_amp_thres = float(words[1])
            elif words[0].lower() == "maxiter":
                cf.maxiter = int(words[1])
            elif words[0].lower() == "pyscf_guess":
                cf.pyscf_guess = words[1]
            elif words[0].lower() == "kappa_guess":
                cf.kappa_guess = words[1]
            elif words[0].lower() == "theta_guess":
                cf.theta_guess = words[1]
            elif words[0].lower() == "1rdm":
                cf.Do1RDM = chkbool(words[1])
            elif words[0].lower() == "kappa_to_t1":
                cf.Kappa_to_T1 = int(words[1])
            elif words[0].lower() == "n_electrons":
                cf.n_active_electrons = int(words[1])
            elif words[0].lower() == "n_orbitals":
                cf.n_active_orbitals = int(words[1])
            elif words[0].lower() == "spinproj":
                cf.SpinProj = chkbool(words[1])
            elif words[0].lower() == "spin":
                cf.spin = int(words[1])
            elif words[0].lower() == "euler":
                if len_words == 2:
                    # Only beta
                    cf.euler_ngrids[1] = int(words[1])
                elif len_words == 3:
                    # Only alpha and beta
                    cf.euler_ngrids[0] = int(words[1])
                    cf.euler_ngrids[1] = int(words[2])
                elif len_words == 4:
                    # alpha, beta, gamma
                    cf.euler_ngrids[0] = int(words[1])
                    cf.euler_ngrids[1] = int(words[2])
                    cf.euler_ngrids[2] = int(words[3])
                else:
                    error("Format for Euler angle is wrong")
            elif words[0].lower() == "nproj":
                cf.number_ngrids = int(words[1])
                cf.NumberProj =  cf.number_ngrids > 1
            elif words[0].lower() == "ds":
                cf.DS = int(words[1])
            elif words[0].lower() == "lambda":
                cf.constraint_lambda = float(words[1])
            elif words[0].lower() == "geometry":
                # Reading Geometry ...
                if job_k == job_no - 1:
                    cf.geom_update = True

                iatom = 0
                cf.geometry = []
                while iatom >= 0:
                    iline += 1
                    if iline == num_lines:
                        return True
                    geom = lines[iline].replace("=", " ")
                    geom = [x.strip() for x in geom.split() if not geom.strip() == ""]

                    if len(geom) > 0:
                        if geom[0] in cf.PeriodicTable:
                            xyz = []
                            xyz.append(float(geom[1]))
                            xyz.append(float(geom[2]))
                            xyz.append(float(geom[3]))
                            atom = []
                            atom.append(geom[0])
                            atom.append(xyz)
                            cf.geometry.append(atom)
                        else:
                            iatom = -1
                            iline += -1
                    else:
                        iatom = -1
                        iline += -1
            elif words[0].lower() in ("det", "determinant"):
                if words[1].isdecimal() and float(words[1]) >= 0:
                    base2 = str("0b" + words[1])
                    base10 = int(base2, 2)
                    cf.det = base10
                    # cf.current_det = base10
                else:
                    prints("Invalid determinant description '{}' ".fortmat(words[1]))
            elif words[0].lower() == "multi":
                # Reading determinants and weights ...
                cf.multi_states = []
                cf.multi_weights = []
                ilabel = 0
                while ilabel >= 0:
                    iline += 1
                    if iline == num_lines:
                        return True
                    multi = lines[iline].replace("=", " ")
                    multi = [
                        x.strip() for x in multi.split() if not multi.strip() == ""
                    ]
                    if len(multi) > 0:
                        if multi[0].isdecimal() and float(multi[1]) >= 0:
                            # Transform string to integer ...
                            base2 = str("0b" + multi[0])
                            base10 = int(base2, 2)
                            cf.multi_states.append(base10)
                            cf.multi_weights.append(float(multi[1]))
                        else:
                            ilabel = -1
                            iline += -1
                    else:
                        ilabel = -1
                        iline += -1

            elif words[0].lower() == "excited":
                # Reading determinants ...
                cf.excited_states = []
                ilabel = 0
                while ilabel >= 0:
                    iline += 1
                    if iline == num_lines:
                        return True
                    excited = lines[iline].replace("=", " ")
                    excited = [
                        x.strip() for x in excited.split() if not excited.strip() == ""
                    ]
                    if len(excited) > 0:
                        if excited[0].isdecimal():
                            # Transform string to integer ...
                            base2 = str("0b" + excited[0])
                            base10 = int(base2, 2)
                            cf.excited_states.append(base10)
                        else:
                            ilabel = -1
                            iline += -1
                    else:
                        ilabel = -1
                        iline += -1

            elif words[0].lower() == "npar":
                cf.nthreads = words[1]
                os.environ["OMP_NUM_THREADS"] = words[1]
            elif words[0].lower() == "@@@":
                job_k += 1
                if job_k == job_no:
                    Finish = False
                    break
            elif words[0].lower() == "hubbard_u":
                cf.hubbard_u = float(words[1])
            elif words[0].lower() == "hubbard_nx":
                cf.hubbard_nx = int(words[1])
            elif words[0].lower() == "hubbard_ny":
                cf.hubbard_ny = int(words[1])
            elif words[0].lower() == "print_fci":
                cf.print_fci = chkbool(words[1])
            elif words[0].lower() == "approx_exp":
                cf.approx_exp = chkbool(words[1])
            elif words[0].lower() == "act2act":
                cf.act2act_ops = chkbool(words[1])
            else:
                error('No option "%s"' % words[0])

        iline += 1  # read next line...

    f.close()

    return Finish
