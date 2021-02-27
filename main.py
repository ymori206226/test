import os

import datetime

from src import config as cf
from src import mpilib as mpi
from src.fileio import error, prints
from src.read import read_input
from src.init import QuketData

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
    "///                      Quantum Computing Simulator Ver 0.4 dev                ///"
)
prints(
    "///                                                                             ///"
)
prints(
    "///        Copyright 2019-2021                                                  ///"
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
    Quket = QuketData()
    job_no += 1
    Finish = read_input(Quket, job_no)
    # Finish = read_input(job_no)
    if Quket.n_electron == 0:
        error("# electrons = 0 !")
    if Quket.basis != "hubbard":
        if Quket.n_orbital == 0:
            error("# orbitals = 0 !")
    else:
        if Quket.hubbard_nx == 0:
            error("Hubbard model but hubbard_nx is not defined!")
        Quket.n_orbital = Quket.hubbard_nx * Quket.hubbard_ny

    if cf.opt_method == "L-BFGS-B":
        opt_options = {
            "disp": True,
            "maxiter": Quket.maxiter,
            "gtol": Quket.gtol,
            "ftol": Quket.ftol,
            "eps": cf.eps,
            "maxfun": cf.maxfun,
        }
    elif cf.opt_method == "BFGS":
        opt_options = {
            "disp": True,
            "maxiter": Quket.maxiter,
            "gtol": Quket.gtol,
            "eps": cf.eps,
        }

    if cf.pyscf_guess == "read":
        cf.pyscf_guess = "chkfile"

    from src.init import QuketData
    from src.vqe import VQE_driver
    from src.qite.qite import QITE_driver

    prints("+-------------+")
    prints("|  Job # %3d  |" % job_no)
    prints("+-------------+")
    prints(
        "{} processes  x  {} threads  =  Total {} cores".format(
            mpi.nprocs, cf.nthreads, mpi.nprocs * int(cf.nthreads)
        )
    )

    if Quket.basis == "hubbard":
        model = "hubbard"
    elif "heisenberg" in Quket.basis:
        model = "heisenberg"
    else:
        model = "chemical"

    #############################
    #                           #
    #    Construct QuketData    #
    #                           #
    #############################
    Quket.initialize(pyscf_guess=cf.pyscf_guess)
    # Transform Jordan-Wigner Operators to Qulacs Format
    Quket.jw_to_qulacs()
    # Set projection parameters
    Quket.set_projection()

    if Quket.ansatz is None or Quket.maxiter == 0:
        continue

    #############
    # VQE part  #
    #############
    if Quket.method == "vqe":
        VQE_driver(
            Quket,
            cf.kappa_guess,
            cf.theta_guess,
            cf.mix_level,
            cf.opt_method,
            opt_options,
            cf.print_level,
            Quket.maxiter,
            cf.Kappa_to_T1,
        )

        # post VQE for excited states
        for istate in range(Quket.nexcited):
            Quket.lower_states.append(Quket.state)
            prints(
                "Performing VQE for excited states: {}/{} states".format(
                    istate + 1, Quket.nexcited
                )
            )
            Quket.det = Quket.excited_states[istate]
            VQE_driver(
                Quket,
                "zero",
                cf.theta_guess,
                cf.mix_level,
                cf.rho,
                cf.DS,
                cf.opt_method,
                opt_options,
                cf.print_level,
                Quket.maxiter,
                False,
            )
    ##############
    # QITE part  #
    ##############
    elif Quket.method == "qite":
        QITE_driver(Quket)
    prints("Normal termination of quket at ", datetime.datetime.now())  # time stamp
    if mpi.main_rank and os.path.exists(cf.tmp):
        os.remove(cf.tmp)
    # VQE part done, go to the next job.
