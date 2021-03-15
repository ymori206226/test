import os
import inspect
import datetime

from src import config as cf
from src import mpilib as mpi
from src.vqe import VQE_driver
from src.qite.qite import QITE_driver
from src.fileio import error, prints
from src.read import read_input, set_config
from src.init import QuketData, get_func_kwds


prints("///////////////////////////////////////////////////////////////////////////////////", opentype="w")
prints("///                                                                             ///")
prints("///                                                                             ///")
prints("///              QQQ       UUU  UUU    KKK   KK    EEEEEEE    TTTTTTT           ///")
prints("///             Q   Q       u    U      K   K       E    E    T  T  T           ///")
prints("///            Q     Q      U    U      K  K        E  E         T              ///")
prints("///            Q     Q      U    U      KKK         EEEE         T              ///")
prints("///            Q QQQ Q      U    U      K  K        E  E         T              ///")
prints("///             Q   Q       U    U      K   K       E    E       T              ///")
prints("///              QQQ QQ      UUUU      KKK   KK    EEEEEEE      TTT             ///")
prints("///                                                                             ///")
prints("///                      Quantum Computing Simulator Ver 0.4 dev                ///")
prints("///                                                                             ///")
prints("///        Copyright 2019-2021                                                  ///")
prints("///        QC Project Team, Ten-no Research Group                               ///")
prints("///        All rights Reserved.                                                 ///")
prints("///                                                                             ///")
prints("///////////////////////////////////////////////////////////////////////////////////")
prints(f"Start at {datetime.datetime.now()}")  # time stamp

######################################
###    Start reading input file    ###
######################################
kwds_list = read_input()
for job_no, kwds in enumerate(kwds_list, 1):
    # Get kwds for initialize QuketData
    init_dict = get_func_kwds(QuketData.__init__, kwds)
    Quket = QuketData(**init_dict)

    ##############
    # Set config #
    ##############
    set_config(kwds, Quket)

    #######################
    # Construct QuketData #
    #######################
    Quket.initialize(**kwds)
    # Transform Jordan-Wigner Operators to Qulacs Format
    Quket.jw_to_qulacs()
    # Set projection parameters
    Quket.set_projection()

    prints("+-------------+")
    prints("|  Job # %3d  |" % job_no)
    prints("+-------------+")
    prints(f"{mpi.nprocs} processes x {cf.nthreads} = "
           f"Total {mpi.nprocs*int(cf.nthreads)} cores")

    if Quket.maxiter <= 0:
        prints(f"\n   Continue the next job due to the {Quket.maxiter=}.\n")
        continue

    ############
    # VQE part #
    ############
    if Quket.method == "vqe":
        VQE_driver(Quket,
                   cf.kappa_guess,
                   cf.theta_guess,
                   cf.mix_level,
                   cf.opt_method,
                   cf.opt_options,
                   cf.print_level,
                   Quket.maxiter,
                   cf.Kappa_to_T1)

        # post VQE for excited states
        for istate in range(Quket.nexcited):
            Quket.lower_states.append(Quket.state)
            prints(f"Performing VQE for excited states: "
                   f"{istate+1}/{Quket.nexcited} states")
            Quket.det = Quket.excited_states[istate]
            VQE_driver(Quket,
                       "zero",
                       cf.theta_guess,
                       cf.mix_level,
                       cf.rho,
                       cf.DS,
                       cf.opt_method,
                       cf.opt_options,
                       cf.print_level,
                       Quket.maxiter,
                       False)
    #############
    # QITE part #
    #############
    elif Quket.method == "qite":
        QITE_driver(Quket)

    if mpi.main_rank and os.path.exists(cf.tmp):
        os.remove(cf.tmp)
    # VQE part done, go to the next job.

prints(f"Normal termination of quket at {datetime.datetime.now()}")
