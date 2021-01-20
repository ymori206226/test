import os

import datetime

from src import config as cf
from src import mpilib as mpi
from src.fileio import error, print_geom, prints
from src.read import read_input

prints(
    "///////////////////////////////////////////////////////////////////////////////////",
    opentype="w",
)
prints(
    "///                                                                             ///"
)
prints(
    "///                                                                             ///"
)
prints(
    "///              QQQ       UUU  UUU    KKK   KK    EEEEEEE    TTTTTTT           ///"
)
prints(
    "///             Q   Q       u    U      K   K       E    E    T  T  T           ///"
)
prints(
    "///            Q     Q      U    U      K  K        E  E         T              ///"
)
prints(
    "///            Q     Q      U    U      KKK         EEEE         T              ///"
)
prints(
    "///            Q QQQ Q      U    U      K  K        E  E         T              ///"
)
prints(
    "///             Q   Q       U    U      K   K       E    E       T              ///"
)
prints(
    "///              QQQ QQ      UUUU      KKK   KK    EEEEEEE      TTT             ///"
)
prints(
    "///                                                                             ///"
)
prints(
    "///                      Quantum Computing Simulator Ver Beta                   ///"
)
prints(
    "///                                                                             ///"
)
prints(
    "///        Copyright 2019-2020                                                  ///"
)
prints(
    "///        QC Project Team, Ten-no Research Group                               ///"
)
prints(
    "///        All rights Reserved.                                                 ///"
)
prints(
    "///                                                                             ///"
)
prints(
    "///////////////////////////////////////////////////////////////////////////////////"
)
prints("Start at ", datetime.datetime.now())  # time stamp

######################################
###    Start reading input file    ###
######################################

Finish = False
job_no = 0
cf.geom_update = False
while Finish is False:
    job_no += 1
    Finish = read_input(job_no)
    if cf.n_active_electrons == 0:
        error("# electrons = 0 !")
    if cf.basis != "hubbard":
        if cf.n_active_orbitals == 0:
            error("# orbitals = 0 !")
    else:
        if cf.hubbard_nx == 0:
            error("Hubbard model but hubbard_nx is not defined!")
        cf.n_orbitals = cf.hubbard_nx * cf.hubbard_ny

    if cf.opt_method == "L-BFGS-B":
        opt_options = {
            "disp": True,
            "maxiter": cf.maxiter,
            "gtol": cf.gtol,
            "ftol": cf.ftol,
            "eps": cf.eps,
            "maxfun": cf.maxfun,
        }
    elif cf.opt_method == "BFGS":
        opt_options = {
            "disp": True,
            "maxiter": cf.maxiter,
            "gtol": cf.gtol,
            "eps": cf.eps,
        }

    if cf.pyscf_guess == "read":
        cf.pyscf_guess = "chkfile"

    from openfermion.transforms import jordan_wigner

    from src.init import set_initial_det
    from src.opelib import generate_operators, get_hubbard
    from src.phflib import set_projection
    from src.vqe import VQE_driver

    prints("+-------------+")
    prints("|  Job # %3d  |" % job_no)
    prints("+-------------+")
    prints(
        "{} processes  x  {} threads  =  Total {} cores".format(
            mpi.nprocs, cf.nthreads, mpi.nprocs * int(cf.nthreads)
        )
    )

    if cf.basis != "hubbard":
        if cf.geometry is None:
            error("No geometry specified.")
        elif cf.geom_update:
            # Set Jordan-Wigner Hamiltonian and S2 operators using PySCF and Open-Fermion
            generate_operators(
                cf.pyscf_guess,
                cf.geometry,
                cf.basis,
                cf.multiplicity,
                cf.charge,
                cf.n_active_electrons,
                cf.n_active_orbitals,
            )
            jw_hamiltonian = jordan_wigner(cf.Hamiltonian_operator)
            jw_s2 = jordan_wigner(cf.S2_operator)
            cf.geom_update = False
            print_geom(cf.geometry)
            prints("E[FCI] = ", cf.fci_energy)
            prints("E[HF]  = ", cf.hf_energy)
            if cf.print_level > 2:
                prints("jw_hamiltonian:\n", jw_hamiltonian)

    elif cf.basis == "hubbard":
        if mpi.main_rank:
            jw_hamiltonian, jw_s2 = get_hubbard(
                cf.hubbard_u,
                cf.hubbard_nx,
                cf.hubbard_ny,
                cf.n_active_electrons,
                cf.run_fci,
            )
            prints(
                "Hubbard model: nx = %d  " % cf.hubbard_nx,
                "ny = %d  " % cf.hubbard_ny,
                "U = %2.2f" % cf.hubbard_u,
            )
        else:
            jw_hamiltonian = None
            jw_s2 = None
        jw_hamiltonian = mpi.comm.bcast(jw_hamiltonian, root=0)
        jw_s2 = mpi.comm.bcast(jw_s2, root=0)

    # If maxiter = 0, skip the VQE part. This option is useful to do PySCF for different geometries
    # (to read and utilize initial guess HF orbitals, which sometimes can change by occupying the wrong orbitals).
    if cf.maxiter == 0:
        continue

    # Check spin, multiplicity, and Ms
    if cf.spin == -1:
        cf.spin = cf.Ms + 1  # Default
    if (cf.spin - cf.Ms - 1) % 2 != 0 or cf.spin < cf.Ms + 1:
        prints("Spin = {}    Ms = {}".format(cf.spin, cf.Ms))
        error("Spin and Ms not cosistent.")
    if (cf.n_active_electrons + cf.multiplicity - 1) % 2 != 0:
        prints(
            "Incorrect specification for n_electrons = {} and multiplicity = {}.".format(
                cf.n_active_electrons, cf.multiplicity
            )
        )

    # Check initial determinant
    if cf.det == -1:
        # Initial determinant is RHF or ROHF
        set_initial_det()
    cf.current_det = cf.det

    if cf.method in ("phf", "suhf", "sghf", "opt_puccsd", "opt_puccd"):
        cf.SpinProj = True
    if cf.SpinProj:
        if cf.method not in (
            "uccsd",
            "uccd",
            "jmucc",
            "uhf",
            "phf",
            "suhf",
            "sghf",
            "opt_puccsd",
            "opt_puccd",
        ):
            prints("Spin-Projection is not yet available for {}.".format(cf.method))
        elif cf.method in ("uccd", "uccsd"):
            cf.method = "p" + cf.method

        set_projection()

    # VQE part
    VQE_driver(
        jw_hamiltonian,
        jw_s2,
        cf.method,
        cf.kappa_guess,
        cf.theta_guess,
        cf.mix_level,
        cf.rho,
        cf.DS,
        cf.opt_method,
        opt_options,
        cf.print_level,
        cf.maxiter,
        cf.Kappa_to_T1,
        cf.print_amp_thres,
    )

    # post VQE for excited states
    nexcited = len(cf.excited_states)
    for istate in range(nexcited):
        cf.lower_states.append(cf.States)
        prints(
            "Performing VQE for excited states: {}/{} states".format(
                istate + 1, nexcited
            )
        )
        cf.current_det = cf.excited_states[istate]
        VQE_driver(
            jw_hamiltonian,
            jw_s2,
            cf.method,
            "zero",
            cf.theta_guess,
            cf.mix_level,
            cf.rho,
            cf.DS,
            cf.opt_method,
            opt_options,
            cf.print_level,
            cf.maxiter,
            False,
            cf.print_amp_thres,
        )
    prints("End at ", datetime.datetime.now())  # time stamp
    if mpi.main_rank and os.path.exists(cf.tmp):
        os.remove(cf.tmp)
    # VQE part done, go to the next job.
