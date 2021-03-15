"""
#######################
#        quket        #
#######################

vqe.py

Main driver of VQE.

"""
import time

import numpy as np
from scipy.optimize import minimize
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import QubitOperator

from . import sampling
from . import mpilib as mpi
from . import config as cf
from .hflib import cost_uhf, mix_orbitals
from .utils import T1mult, cost_mpi, jac_mpi
from .fileio import LoadTheta, SaveTheta, error, prints, printmat
from .phflib import cost_proj
from .ucclib import cost_uccd, cost_uccsdX
from .upcclib import cost_upccgsd
from .agpbcs import cost_bcs
from .jmucc import cost_jmucc
from .icmrucc import cost_ic_mrucc
from .prop import dipole, get_1RDM


def VQE_driver(Quket, kappa_guess, theta_guess, mix_level, opt_method,
               opt_options, print_level, maxiter, Kappa_to_T1):
    """Function:
    Main driver for VQE

    Author(s): Takashi Tsuchimochi
    """
    jw_hamiltonian = Quket.operators.jw_Hamiltonian
    jw_s2 = Quket.operators.jw_S2
    ansatz = Quket.ansatz
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    nca = Quket.nca # Number of Core orbitals of Alpha
    ncb = Quket.ncb # Number of Core orbitals of Beta

    t1 = time.time()
    cf.t_old = t1
    print_control = 1

    optk = 0
    Gen = 0
    # cf.constraint_lambda = 100

    ### set up the number of orbitals and such ###
    n_electrons = Quket.n_electrons
    n_qubit_system = Quket.n_qubits
    n_qubits = n_qubit_system + 1
    anc = n_qubit_system

    ### set ndim ###
    spin_gen = ansatz in "sghf"
    norbs = noa + nva
    ndim1 = noa*nva + nob*nvb
    ndim2aa = noa*(noa-1)*nva*(nva-1)//4
    ndim2ab = noa*nob*nva*nvb
    ndim2bb = nob*(nob-1)*nvb*(nvb-1)//4
    ndim2 = ndim2aa + ndim2ab + ndim2bb
    if ansatz in ("uhf", "phf", "suhf"):
        ndim = ndim1
    elif ansatz in ("uccd", "puccd"):
        ndim = ndim2
    elif ansatz in ("uccsd", "opt_puccd", "puccsd"):
        ndim = ndim1 + ndim2
    elif ansatz == "sghf":
        ndim1 = (noa+nob)*(nva+nvb)
        ndim = ndim1
    elif ansatz == "opt_psauccd":
        ndim2 = noa*nva*(noa*nva + 1)//2
        ndim = ndim1 + ndim2
    elif ansatz == "sauccsd":
        ndim1 = noa*nva
        ndim2 = ndim1*(ndim1+1)//2
        ndim = ndim1 + ndim2
    elif "bcs" in ansatz:
        if "ebcs" in ansatz:
            k_param = ansatz[0:nsatz.find("-ebcs")]
        else:
            k_param = ansatz[0:ansatz.find("-bcs")]
        if not k_param.isdecimal():
            prints(f"Unrecognized k: {k_param}")
            error("k-BCS without specifying k.")
        k_param = int(k_param)
        if k_param < 1:
            error("0-bcs is just HF!")
        ndim1 = norbs*(norbs-1)//2
        ndim2 = norbs
        ndim = k_param*(ndim1+ndim2)
    elif "pccgsd" in ansatz:
        if "upccgsd" in ansatz:
            k_param = ansatz[0:ansatz.find("-upccgsd")]
        elif "epccgsd" in ansatz:
            k_param = ansatz[0:ansatz.find("-epccgsd")]
        if not k_param.isdecimal():
            prints(f"Unrecognized k: {k_param}")
            error("k-UpCCGSD without specifying k.")
        k_param = int(k_param)
        if k_param < 1:
            error("0-upccgsd is just HF!")
        ndim1 = norbs*(norbs-1)//2
        ndim2 = norbs*(norbs-1)//2
        ndim = k_param*(ndim1+ndim2)
        if "epccgsd" in ansatz:
            ndim += ndim1
    elif ansatz == "ic_mrucc":
        # assume that noa = nob and nva = nvb
# そういやフローズンコアに対応してないよなQuketData
        ndim1 = (nca*noa + nca*nva + noa*(noa-1)//2 + noa*nva
               + ncb*nob + ncb*nvb + nob*(nob-1)//2 + nob*nvb)
        if cf.act2act_opt:
            ndim2aa = ((nca*(nca-1)//2 + nca*noa + noa*(noa-1)//2)
                      *(noa*(noa-1)//2 + noa*nva + nva*(nva-1)//2)
                      - noa*(noa-1)//2)
            ndim2ab = ((noa*noa + noa*nva*2 + nva*nva)
                      *(nob*nob + ncb*nob*2 + ncb*ncb)
                      - noa*nob)
            ndim2bb = ((ncb*(ncb-1)//2 + ncb*nob + nob*(nob-1)//2)
                      *(nob*(nob-1)//2 + nob*nvb + nvb*(nvb-1)//2)
                      - nob*(nob-1)//2)
        else:
            ndim2aa = ((nca*(nca-1)//2 + nca*noa + noa*(noa-1)//2)
                      *(noa*(noa-1)//2 + noa*nva + nva*(nva-1)//2)
                      - noa*(noa-1)//2)
            ndim2ab = ((noa*noa + noa*nva*2 + nva*nva)
                      *(nob*nob + ncb*nob*2 + ncb*ncb)
                      - noa*noa*nob*nob)
            ndim2bb = ((ncb*(ncb-1)//2 + ncb*nob + nob*(nob-1)//2)
                      *(nob*(nob-1)//2 + nob*nvb + nvb*(nvb-1)//2)
                      - noa*(noa-1)*nob*(nob-1)//4)
        ndim2 = ndim2aa + ndim2ab + ndim2bb
        ndim = ndim1 + ndim2
    elif ansatz == "jmucc":
        if Quket.multi.nstates == 0:
            error("JM-UCC specified without state specification!")
        ndim = Quket.multi.nstates*(ndim1+ndim2)

    # set number of dimensions QuketData
    Quket.ndim1 = ndim1
    Quket.ndim2 = ndim2
    Quket.ndim = ndim

    ### HxZ and S**2xZ and IxZ  ###
    ### Trick! Remove the zeroth-order term, which is the largest
    term0_H = Quket.qulacs.Hamiltonian.get_term(0)
    coef0_H = term0_H.get_coef()
    coef0_H = coef0_H.real
    term0_S2 = Quket.qulacs.S2.get_term(0)
    coef0_S2 = term0_S2.get_coef()
    coef0_S2 = coef0_S2.real

    coef0_H = 0
    coef0_S2 = 0

    jw_ancZ = QubitOperator(f"Z{anc}")
    jw_hamiltonianZ = (jw_hamiltonian - coef0_H*QubitOperator(""))*jw_ancZ
    jw_s2Z = (jw_s2 - coef0_S2*QubitOperator(""))*jw_ancZ
    qulacs_hamiltonianZ \
            = create_observable_from_openfermion_text(str(jw_hamiltonianZ))
    qulacs_s2Z \
            = create_observable_from_openfermion_text(str(jw_s2Z))
    qulacs_ancZ \
            = create_observable_from_openfermion_text(str(jw_ancZ))

    #############################
    ### set up cost functions ###
    #############################
    if ansatz in ("phf", "suhf", "sghf"):
        ### PHF ###
        cost_wrap = lambda kappa_list: cost_proj(
                Quket,
                0,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                )[0]
        cost_callback = lambda kappa_list, print_control: cost_proj(
                Quket,
                print_control,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                )
    elif ansatz == "uhf":
        ### UHF ###
        cost_wrap = lambda kappa_list: cost_uhf(
                Quket,
                0,
                kappa_list,
                )[0]
        cost_callback = lambda kappa_list, print_control: cost_uhf(
                Quket,
                print_control,
                kappa_list,
                )
    elif ansatz == "uccsd" or ansatz == "sauccsd":
        ### UCCSD ###
        cost_wrap = lambda theta_list: cost_uccsdX(
                Quket,
                0,
                kappa_list,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_uccsdX(
                Quket,
                print_control,
                kappa_list,
                theta_list,
                )
    elif "bcs" in ansatz:
        ###BCS###
        cost_wrap = lambda theta_list: cost_bcs(
                Quket,
                0,
                theta_list,
                k_param,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_bcs(
                Quket,
                print_control,
                theta_list,
                k_param,
                )
    elif "pccgsd" in ansatz:
        ###UpCCGSD###
        cost_wrap = lambda theta_list: cost_upccgsd(
                Quket,
                0,
                kappa_list,
                theta_list,
                k_param,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_upccgsd(
                Quket,
                print_control,
                kappa_list,
                theta_list,
                k_param,
                )
    elif ansatz == "uccd":
        ### UCCD ###
        cost_wrap = lambda theta_list: cost_uccd(
                Quket,
                0,
                kappa_list,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_uccd(
                Quket,
                print_control,
                kappa_list,
                theta_list,
                )
    elif ansatz == "puccsd":
        ### UCCSD ###
        cost_wrap = lambda theta_list: cost_proj(
                Quket,
                0,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_proj(
                Quket,
                print_control,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )
    elif ansatz == "puccd":
        ### UCCSD ###
        cost_wrap = lambda theta_list: cost_proj(
                Quket,
                0,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_proj(
                Quket,
                print_control,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )
    elif ansatz in ("opt_puccd", "opt_psauccd"):
        ### UCCSD ###
        cost_wrap = lambda theta_list: cost_proj(
                Quket,
                0,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_proj(
                Quket,
                print_control,
                qulacs_hamiltonianZ,
                qulacs_s2Z,
                coef0_H,
                coef0_S2,
                kappa_list,
                theta_list,
                )
    elif ansatz == "jmucc":
        cost_wrap = lambda theta_list: cost_jmucc(
                Quket,
                0,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_jmucc(
                Quket,
                print_control,
                theta_list,
                )
    elif ansatz == "ic_mrucc":
        cost_wrap = lambda theta_list : cost_ic_mrucc(
                Quket,
                0,
                qulacs_hamiltonian,
                qulacs_s2,
                theta_list,
                )[0]
        cost_callback = lambda theta_list, print_control: cost_ic_mrucc(
                Quket,
                print_control,
                qulacs_hamiltonian,
                qulacs_s2,
                theta_list,
                )

    fstr = f"0{n_qubit_system}b"
    prints(f"Performing VQE for {ansatz}")
    prints(f"Number of VQE parameters: {ndim}")
    prints(f"Initial configuration: | {format(Quket.det, fstr)} >")
    #prints("Convergence criteria:  ftol = {:1.0E}   gtol = {:1.0E}".format(Quket.ftol, Quket.gtol))
    prints(f"Convergence criteria: ftol = {Quket.ftol:1.0E}, "
                                 f"gtol = {Quket.gtol:1.0E}")

    #############################
    ### set up initial kappa  ###
    #############################
    kappa_list = np.empty(ndim1)
    if kappa_guess == "mix":
        if mix_level > 0:
            mix = mix_orbitals(noa, nob, nva, nvb, mix_level, False, np.pi/4)
            kappa_list = mix[:ndim1]
            printmat(kappa_list)
        elif mix_level == 0:
            error("kappa_guess = mix  but  mix_level = 0 !")
    elif kappa_guess == "random":
        if spin_gen:
            mix = mix_orbitals(noa+nob, 0, nva+nvb, 0, mix_level, True, np.pi/4)
        else:
            mix = mix_orbitals(noa, nob, nva, nvb, mix_level, True, np.pi/4)
        kappa_list = mix[:ndim1]
    elif kappa_guess == "read":
        kappa_list = LoadTheta(ndim1, cf.kappa_list_file)
        if mix_level > 0:
            temp = kappa_list[:ndim1]
            mix = mix_orbitals(noa, nob, nva, nvb, mix_level, False, np.pi/4)
            temp = T1mult(noa, nob, nva, nvb, mix, temp)
            kappa_list = temp[:ndim1]
        printmat(kappa_list)
    elif kappa_guess == "zero":
        kappa_list *= 0

    #############################
    ### set up initial theta  ###
    #############################
    theta_list = np.empty(ndim)
    prints(f"Theta list = {theta_guess}")
    if theta_guess == "zero":
        theta_list *= 0
    elif theta_guess == "read":
        theta_list = LoadTheta(ndim, cf.theta_list_file)
    elif theta_guess == "random":
        theta_list = (0.5-np.random.rand(ndim))*0.1
    if Kappa_to_T1 and theta_guess != "read":
        ### Use Kappa for T1  ###
        theta_list[:ndim1] = kappa_list[:ndim1]
        kappa_list *= 0
        prints("Initial T1 amplitudes will be read from kappa.")

    if optk:
        theta_list_fix = theta_list[ndim1:]
        theta_list = theta_list[:ndim1]
        if Gen:
            # Generalized Singles.
            temp = theta_list.copy()
            theta_list = np.zeros(ndim)
            indices = [i*(i-1)//2 + j
                        for i in range(norbs)
                            for j in range(i)
                                if i >= noa and j < noa]
            indices.extend([i*(i-1)//2 + j + ndim1
                                for i in range(norbs)
                                    for j in range(i)
                                        if i >= nob and j < nob])
            theta_list[indices] = temp[:len(indices)]
            # Same as following code;
            #theta_list = []
            #ij = 0
            #for i in range(norbs):
            #    for j in range(i):
            #        if i < noa and j < noa:
            #            ## occ-occ
            #            theta_list.append(0)
            #        elif i >= noa and j < noa:
            #            ## vir-occ
            #            theta_list.append(temp[ij])
            #            ij += 1
            #        elif i >= noa and j >= noa:
            #            ## vir-vir
            #            theta_list.append(0)
            #for i in range(norbs):
            #    for j in range(i):
            #        if i < nob and j < nob:
            #            ## occ-occ
            #            theta_list.append(0)
            #        elif i >= nob and j < nob:
            #            ## vir-occ
            #            theta_list.append(temp[ij])
            #            ij += 1
            #        elif i >= nob and j >= nob:
            #            ## vir-vir
            #            theta_list.append(0)
    else:
        theta_list_fix = 0

    ### Broadcast lists
    mpi.comm.Bcast(kappa_list, root=0)
    mpi.comm.Bcast(theta_list, root=0)

    ### print out initial results ###
    print_control = -1
    if ansatz in ("uhf", "phf", "suhf", "sghf"):
        cost_callback(kappa_list, print_control)
    else:
        if Quket.DS:
            prints("Circuit order: Exp[T2] Exp[T1] |0>")
        else:
            prints("Circuit order: Exp[T1] Exp[T2] |0>")
        # prints('Initial T1 amplitudes:')
        # prints('Intial T2 amplitudes:')
        cost_callback(theta_list, print_control)
    print_control = 1

    if maxiter > 0:
        ###################
        ### perform VQE ###
        ###################
        cf.icyc = 0
        # Use MPI for evaluating Gradients
        cost_wrap_mpi = lambda theta_list: cost_mpi(cost_wrap, theta_list)
        jac_wrap_mpi = lambda theta_list: jac_mpi(cost_wrap, theta_list)

        if ansatz in ("uhf", "phf", "suhf", "sghf"):
            opt = minimize(
                    cost_wrap_mpi,
                    kappa_list,
                    jac=jac_wrap_mpi,
                    method=opt_method,
                    options=opt_options,
                    callback=lambda x: cost_callback(x, print_control),
                    )
        else:  ### correlated ansatzs
            opt = minimize(
                    cost_wrap_mpi,
                    theta_list,
                    jac=jac_wrap_mpi,
                    method=opt_method,
                    options=opt_options,
                    callback=lambda x: cost_callback(x, print_control),
                    )

        ### print out final results ###
        final_param_list = opt.x
    elif maxiter == -1:
        # Skip VQE, and perform one-shot calculation
        prints("One-shot evaluation without parameter optimization")
        if ansatz in ("uhf", "phf", "suhf", "sghf"):
            final_param_list = kappa_list
        else:
            final_param_list = theta_list

    # Calculate final parameters.
    Evqe, S2 = cost_callback(final_param_list, print_control+1)
    if ansatz in ("uhf", "phf", "suhf", "sghf"):
        SaveTheta(ndim, final_param_list, cf.kappa_list_file)
    else:
        SaveTheta(ndim, final_param_list, cf.theta_list_file)
# Same as?
#    if ansatz in ("phf", "suhf", "sghf"):
#        Evqe, S2 = cost_proj(
#                Quket,
#                print_control + 1,
#                qulacs_hamiltonianZ,
#                qulacs_s2Z,
#                coef0_H,
#                coef0_S2,
#                final_param_list,
#                )
#        SaveTheta(ndim, final_param_list, cf.kappa_list_file)
#    elif ansatz == "uhf":
#        Evqe, S2 = cost_uhf(
#                Quket,
#                print_control + 1,
#                final_param_list,
#                )
#        SaveTheta(ndim, final_param_list, cf.kappa_list_file)
#    elif ansatz == "uccsd" or ansatz == "sauccsd":
#        Evqe, S2 = cost_uccsdX(
#                Quket,
#                print_control + 1,
#                kappa_list,
#                final_param_list,
#                )
#        SaveTheta(ndim, final_param_list, cf.theta_list_file)
#    elif "bcs" in ansatz:
#        Evqe, S2 = cost_bcs(
#                Quket,
#                print_control + 1,
#                final_param_list,
#                k_param,
#                )
#        SaveTheta(ndim, final_param_list, cf.theta_list_file)
#    elif "pccgsd" in ansatz:
#        Evqe, S2 = cost_upccgsd(
#                Quket,
#                print_control + 1,
#                kappa_list,
#                final_param_list,
#                k_param,
#                )
#        SaveTheta(ndim, final_param_list, cf.theta_list_file)
#    elif ansatz == "uccd":
#        Evqe, S2 = cost_uccd(
#                Quket,
#                print_control + 1,
#                kappa_list,
#                final_param_list,
#                )
#        SaveTheta(ndim, final_param_list, cf.theta_list_file)
#    elif ansatz in ("puccsd", "puccd", "opt_puccd", "opt_psauccd"):
#        Evqe, S2 = cost_proj(
#                Quket,
#                print_control + 1,
#                qulacs_hamiltonianZ,
#                qulacs_s2Z,
#                coef0_H,
#                coef0_S2,
#                kappa_list,
#                final_param_list,
#                )
#        SaveTheta(ndim, final_param_list, cf.theta_list_file)
#    elif ansatz == "jmucc":
#        Evqe, S2 = cost_jmucc(
#                Quket,
#                print_control + 1,
#                final_param_list,
#                )
#        SaveTheta(ndim, final_param_list, cf.theta_list_file)

    if Quket.state is not None:
        if Quket.model == 'chemical':
            dipole(Quket)
        if Quket.Do1RDM:
            Daa, Dbb = get_1RDM(Quket, print_level=1)
        ### Test
    #        from .opelib import single_operator_gradient
    #        g=np.zeros((n_qubit_system,n_qubit_system))
    #        for q in range(n_qubit_system):
    #            for p in range(q):
    #                g[p][q] = single_operator_gradient(p,q,jw_hamiltonian,cf.States,n_qubit_system)
    #        printmat(g,filepath=None,name="Grad")
    #
    t2 = time.time()
    cput = t2 - t1
    prints("\n Done: CPU Time =  ", "%15.4f" % cput)

    return Evqe, S2

    #####################
    ### Sampling test ###
    #####################
    #n_term = fermionic_hamiltonian.get_term_count()
    # There are n_term - 1 Pauli operators to measure (identity not counted).
    #<HUg> = \sum_I  h_I <P_I Ug>
    #theta_list = ([ 0.0604642 ,  0.785282  ,  1.28474901, -0.0248133 , -0.01770559,       -0.7854844 , -0.28473988,  0.02487922])
    #sampling.cost_phf_sample(Quket, 1, qulacs_hamiltonianZ, qulacs_s2Z, qulacs_ancZ, theta_list, 1000000)
    #sampling.cost_uhf_sample(Quket, 1, qulacs_hamiltonian, qulacs_s2, theta_list, 1000)
    #cost_uhf(1,n_qubit_system,n_electrons,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,theta_list)
    #sampling.cost_uhf_sample(Quket, 1, qulacs_hamiltonian, qulacs_s2, theta_list, 100000)
    #amplelist = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    #amplelist = [10, 100, 1000, 10000]
    #samplelist = [1]
    #samplelist = [1000000]
    #sampling.cost_uhf_sample(Quket, 1, qulacs_hamiltonian, qulacs_s2, uhf_theta_list, samplelist)
    #sampling.cost_phf_sample(Quket, 1, qulacs_hamiltonian, qulacs_hamiltonianZ, qulacs_s2Z, qulacs_ancZ, coef0_H, coef0_S2, method, opt.x, samplelist)

    #####################
    ### Rotation test ###
    #####################
    # The obtained results are invariant with respect to  occ-occ rotations?
#    if ansatz == 'uccsd':
#        Gen = 1
#        if Gen:
#            kappa_list = np.zeros(norbs*(norbs-1))
#        else:
#            kappa_list = np.zeros(ndim1)
#        theta_list = opt.x
#        cost_wrap = lambda kappa_list: cost_opttest_uccsd(0,n_qubit_system,n_electrons,noa,nob,nva,nvb,rho,DS,Gen,qulacs_hamiltonian,qulacs_s2,method,kappa_list,theta_list)[0]
#        opt = minimize(cost_wrap, kappa_list,
#              method=opt_method,options=opt_options)
