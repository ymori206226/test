"""
#######################
#        quket        #
#######################

driver.py

Main driver of VQE.

"""


import os
from . import mpilib as mpi
from . import config as cf
import numpy as np
from scipy.optimize import minimize
import sys
import time
import csv
from pprint import pprint
from openfermion.transforms import get_fermion_operator,jordan_wigner
from openfermion.hamiltonians import MolecularData
from openfermion.utils import number_operator, s_squared_operator
from pyscf import fci
from qulacs import QuantumState,QuantumCircuit
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import QubitOperator


from .hflib     import cost_uhf, mix_orbitals, bs_orbitals
from .utils     import LoadTheta, SaveTheta,  error, T1mult, cost_mpi, jac_mpi
from .mod       import run_pyscf_mod, generate_molecular_hamiltonian_mod
from .phflib    import cost_proj
from .ucclib    import cost_uccsd, cost_uccd, cost_opt_ucc, cost_opttest_uccsd, cost_upccgsd
from .jmucc     import cost_jmucc
from . import sampling

def get_hamiltonian(geometry, basis, multiplicity, charge, n_active_electrons, n_active_orbitals, guess):
    """ Function:
    Perform PySCF to generate molecular integrals and such,
    then perform Jordan-Wigner transformation of Hamiltonian and S**2 with OpenFermion
    """
    description = "tmp"
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_pyscf_mod(guess,n_active_orbitals,n_active_electrons,molecule,run_scf=1,run_fci=1)
    if mpi.main_rank:
        with open(cf.log,'a') as f:
            print("RHF Energy : ", molecule.hf_energy ,file=f)
            print("FCI Energy : ", molecule.fci_energy ,file=f)
    ### construct qulacs hamiltonian, S^2 for VQE ###
    ### H ###
    fermionic_hamiltonian = generate_molecular_hamiltonian_mod(guess,geometry, basis, multiplicity, charge, n_active_electrons, n_active_orbitals)
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
     
    ### S^2 ###
    fermionic_s2 = s_squared_operator(n_active_orbitals)
    jw_s2 = jordan_wigner(fermionic_s2)
    if(cf.constraint_lambda > 0):
        fermionic_s4 = fermionic_s2*fermionic_s2
        jw_s4 = jordan_wigner(fermionic_s4)
        cf.qulacs_s4 = create_observable_from_openfermion_text(str(jw_s4))
    return jw_hamiltonian, jw_s2



def get_hubbard(hubbard_u,hubbard_nx,hubbard_ny,n_electrons,run_fci=1):
    from openfermion.utils import QubitDavidson
    """ Function:
    Generate Hamiltonian for Hubbard.
    """
    from openfermion.hamiltonians import fermi_hubbard
    from openfermion.transforms import get_sparse_operator, jordan_wigner
    from openfermion.utils import get_ground_state
    fermionic_hamiltonian = fermi_hubbard(hubbard_nx, hubbard_ny, 1, hubbard_u)
    jw_hamiltonian = jordan_wigner(fermionic_hamiltonian)
    fermionic_s2 = s_squared_operator(hubbard_nx*hubbard_ny)
    jw_s2 = jordan_wigner(fermionic_s2)
    if run_fci == 1: 
        n_qubit = hubbard_nx*hubbard_ny*2
        jw_hamiltonian.compress()
        qubit_eigen = QubitDavidson(jw_hamiltonian,n_qubit)
        # Initial guess :  | 0000...00111111> 
        #                             ~~~~~~ = n_electrons
        guess = np.zeros((2**n_qubit,1))
        #
        guess[2**n_electrons - 1][0] = 1.0 
        n_state = 1
        results = qubit_eigen.get_lowest_n(n_state,guess)
        print("Convergence?           : ",  results[0])
        print("Ground State Energy    : ",  results[1][0])
        print("Wave function          : ")
        openfermion_print_state(results[2],n_qubit,0)
    return jw_hamiltonian, jw_s2
def openfermion_print_state(state,n_qubit,j_state):
    """
    print out jth wave function in state
    """
    opt='0'+str(n_qubit)+'b'
    for i in range(2**n_qubit):
        v = state[i][j_state]
        if abs(v)**2>0.01:
            print('|',format(i,opt),'> : ', '{a.real:+.4f} {a.imag:+.4f}i'.format(a=v))


def VQE_driver(jw_hamiltonian,jw_s2,n_active_electrons, n_active_orbitals, multiplicity, method,  
    kappa_guess,theta_guess,mix_level, rho, DS, opt_method, opt_options, print_level, maxiter,
    Kappa_to_T1, print_amp_thres):
    """ Function:
    Main driver for VQE
    """
    t1 = time.time()
    cf.t_old = t1
    print_control = 1

    optk = 0
    Gen = 0
    #cf.constraint_lambda = 100

    ### set up the number of orbitals and such ###
    n_electron = n_active_electrons
    n_qubit_system = n_active_orbitals*2 
    n_qubit = n_qubit_system + 1
    anc =  n_qubit_system
    
    # Number of occupied orbitals of alpha
    # NOA
    noa = int((n_electron + multiplicity - 1)/2)
    # Number of occupied orbitals of beta
    # NOB
    nob = n_electron - noa
    # Number of virtual orbitals of alpha
    # NVA
    nva = n_active_orbitals - noa
    # Number of virtual orbitals of beta
    # NVB
    nvb = n_active_orbitals - nob
    
    norbs = noa+nva
    if Gen:
        ndim1G = norbs*(norbs-1)
    else: 
        ndim1 = noa*nva + nob*nvb 
    ndim1 = noa*nva + nob*nvb
    ndim2aa = int(noa*(noa-1)*nva*(nva-1)/4)  
    ndim2ab = int(noa*nob*nva*nvb)  
    ndim2bb = int(nob*(nob-1)*nvb*(nvb-1)/4)  
    ndim2 = ndim2aa + ndim2ab + ndim2bb
    if(method == "sauccsd"):
        ndim1 = noa*nva 
        ndim2 = int(ndim1*(ndim1+1)/2)
    elif(method == "opt_psauccd"):
        ndim2 = int(noa*nva*(noa*nva+1)/2)
    
    qulacs_hamiltonian = create_observable_from_openfermion_text(str(jw_hamiltonian))
    qulacs_s2 = create_observable_from_openfermion_text(str(jw_s2))
    
    ### HxZ and S**2xZ and IxZ  ###
    ### Trick! Remove the zeroth-order term, which is the largest
    term0_H = qulacs_hamiltonian.get_term(0)
    coef0_H = term0_H.get_coef()
    coef0_H = coef0_H.real
    term0_S2 = qulacs_s2.get_term(0)
    coef0_S2 = term0_S2.get_coef()
    coef0_S2 = coef0_S2.real
    
    coef0_H = 0
    coef0_S2 = 0
    
    jw_ancZ = QubitOperator('Z%d' % anc)
    jw_hamiltonianZ = (jw_hamiltonian -coef0_H * QubitOperator('') ) * jw_ancZ
    jw_s2Z = (jw_s2 - coef0_S2 * QubitOperator('')) * jw_ancZ
    qulacs_hamiltonianZ = create_observable_from_openfermion_text(str(jw_hamiltonianZ))
    qulacs_s2Z = create_observable_from_openfermion_text(str(jw_s2Z))
    qulacs_ancZ = create_observable_from_openfermion_text(str(jw_ancZ))
    
    
    ############################# 
    ### set up cost functions ###
    ############################# 
    if (method == "phf"):
        ### PHF ###
        ndim = ndim1
        cost_wrap = lambda kappa_list : cost_proj(0,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list)[0]
        cost_callback = lambda kappa_list : cost_proj(print_control,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list)
    
    elif (method == "uhf"):
        ### UHF ###
        ndim = ndim1
        cost_wrap = lambda kappa_list : cost_uhf(0,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,kappa_list)[0]
        cost_callback = lambda kappa_list :  cost_uhf(print_control,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,kappa_list)
    
    elif (method == "uccsd" or method == "sauccsd"):
        ### UCCSD ###
        ndim = ndim1 + ndim2
        cost_wrap = lambda theta_list : cost_uccsd(0,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,qulacs_hamiltonian,qulacs_s2,method,kappa_list,theta_list)[0]
        cost_callback = lambda theta_list : cost_uccsd(print_control,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,qulacs_hamiltonian,qulacs_s2,method,kappa_list,theta_list)

    elif("upccgsd" in method):
        ###UpCCGSD###
        k_param = method[0:method.find('-upccgsd')]
        if(not k_param.isdecimal()):
            if mpi.main_rank:
                with open(cf.log,'a') as f:
                    print('Unrecognized k: ',kparam,file=f)
            error("k-UpCCGSD without specifying k.")
        k_param = int(k_param)
        ###ndim= noa * nvb
        ndim = k_param*(ndim1 + ndim2)
        cost_wrap = lambda theta_list : cost_upccgsd(0,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,qulacs_hamiltonian,qulacs_s2,kappa_list,theta_list,k_param)[0]
        cost_callback = lambda theta_list : cost_upccgsd(print_control,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,qulacs_hamiltonian,qulacs_s2,kappa_list,theta_list,k_param)        
        
    elif (method == "uccd"):
        ### UCCD ###
        ndim = ndim2
        cost_wrap = lambda theta_list : cost_uccd(0,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,qulacs_hamiltonian,qulacs_s2,kappa_list,theta_list)[0]
        cost_callback = lambda theta_list : cost_uccd(print_control,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,qulacs_hamiltonian,qulacs_s2,kappa_list,theta_list)

    elif method == "opt_uccd" or method == "opt_uccsd":
        ndim = ndim1 + ndim2
        cost_wrap = lambda theta_list : cost_opt_ucc(0,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,Gen,optk,qulacs_hamiltonian,qulacs_s2,method,theta_list_fix,theta_list)[0]
        cost_callback = lambda theta_list : cost_opt_ucc(print_control,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,Gen,optk,qulacs_hamiltonian,qulacs_s2,method,theta_list_fix,theta_list)

    elif (method == "puccsd"):
        ### UCCSD ###
        ndim = ndim1 + ndim2
        cost_wrap = lambda theta_list : cost_proj(0,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list,theta_list)[0]
        cost_callback = lambda theta_list : cost_proj(print_control,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list,theta_list)
    elif (method == "puccd"):
        ### UCCSD ###
        ndim =  ndim2
        cost_wrap = lambda theta_list : cost_proj(0,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list,theta_list)[0]
        cost_callback = lambda theta_list : cost_proj(print_control,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list,theta_list)
    elif method =="opt_puccd" or method == "opt_psauccd":
        ### UCCSD ###
        ndim =  ndim1+ndim2
        cost_wrap = lambda theta_list : cost_proj(0,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list,theta_list)[0]
        cost_callback = lambda theta_list : cost_proj(print_control,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list,theta_list)
    elif method =="jmucc":
        nstates = len(cf.multi_weights)
        if nstates == 0:
            error("JM-UCC specified without state specification!")
        ndim = nstates * (ndim1 + ndim2)
        cost_wrap = lambda theta_lists : \
        cost_jmucc(0,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,\
        qulacs_hamiltonian,qulacs_s2,theta_lists,print_amp_thres)
        cost_callback = lambda theta_lists : \
        cost_jmucc(print_control,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,\
        qulacs_hamiltonian,qulacs_s2,theta_lists,print_amp_thres)
    '''
    elif method == "sauccsd_eigen":
        ndim = 2 * (ndim1 + ndim2)
        from .ucc_eigen import cost_sa_XX
        cost_wrap = lambda theta_lists : cost_sa_XX(0,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,qulacs_hamiltonian,qulacs_s2,theta_lists,print_amp_thres)
        cost_callback = lambda theta_lists : cost_sa_XX(print_control,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,qulacs_hamiltonian,qulacs_s2,theta_lists,print_amp_thres)
    '''

    if mpi.main_rank:
        with open(cf.log,'a') as f:
            print('Number of VQE parameters:',ndim,file=f)
    ############################# 
    ### set up initial kappa  ###
    ############################# 
    if mpi.main_rank:
        with open(cf.log,'a') as f:
            print('Kappa values for orbital rotation:',file=f)
    kappa_list = np.zeros(ndim1)
    if  kappa_guess=="mix":
        if(mix_level >0):
            mix = mix_orbitals(noa,nob,nva,nvb,mix_level,False,np.pi/4)
            kappa_list[0:ndim1] = mix[0:ndim1]
            if mpi.main_rank:
                with open(cf.log,'a') as f:
                    pprint(kappa_list,stream=f)
        elif(mix_level ==0 ):
            error('kappa_guess = mix  but  mix_level = 0 !')
    elif  kappa_guess=="random":
        mix = mix_orbitals(noa,nob,nva,nvb,mix_level,True,np.pi/4)
        kappa_list[0:ndim1] = mix[0:ndim1]
    elif  kappa_guess=="read":
        kappa_list = LoadTheta(ndim1,cf.kappa_list_file)
        if(mix_level>0):
            temp = kappa_list[0:ndim1]
            mix = mix_orbitals(noa,nob,nva,nvb,mix_level,False,np.pi/4)
            temp = T1mult(noa,nob,nva,nvb,mix,temp)
            kappa_list = temp[:ndim1]
        if mpi.main_rank:
            with open(cf.log,'a') as f:
                pprint(kappa_list,stream=f)

    ############################# 
    ### set up initial theta  ###
    ############################# 
    theta_list = np.zeros(ndim)
    if  theta_guess=="zero":
        theta_list = np.zeros(ndim)
    elif  theta_guess=="read":
        theta_list = LoadTheta(ndim,cf.theta_list_file)
    elif  theta_guess=="random":
        for i in range (ndim):
            theta_list[i] = (0.5 - np.random.rand())*0.1

    if Kappa_to_T1 and theta_guess != "read":
        ### Use Kappa for T1  ###
        theta_list[:ndim1] = kappa_list[:ndim1]
        for i in range(ndim1):
            kappa_list[i] = 0 
        if mpi.main_rank:
            with open(cf.log,'a') as f:
                print('Initial T1 amplitudes will be read from kappa.',file=f)
        
    kappa_list = np.array(kappa_list)
    theta_list = np.array(theta_list)
    if optk:
        theta_list_fix = theta_list[ndim1:ndim1+ndim2]
        for i in range(ndim1,ndim1+ndim2):
            theta_list=np.delete(theta_list,ndim1)
        if Gen:
            ## Generalized singles
            temp = theta_list
            theta_list=[]
            ij=0
            for i in range(norbs):
                for j in range(i):
                    if i < noa and j < noa: 
                        ## occ-occ
                        theta_list.append(0)
                    elif i >= noa and j < noa: 
                        ## vir-occ
                        theta_list.append(temp[ij])
                        ij+=1
                    elif i >= noa and j >= noa: 
                        ## vir-vir
                        theta_list.append(0)
            for i in range(norbs):
                for j in range(i):
                    if i < nob and j < nob: 
                        ## occ-occ
                        theta_list.append(0)
                    elif i >= nob and j < nob: 
                        ## vir-occ
                        theta_list.append(temp[ij])
                        ij+=1
                    elif i >= nob and j >= nob: 
                        ## vir-vir
                        theta_list.append(0)
            theta_list = np.array(theta_list)
    else:
        theta_list_fix = 0

    ### print out initial results ###
    print_control = -1
    if (method == "phf" or method == "uhf"):
        cost_callback(kappa_list)
    else:
        if mpi.main_rank:
            with open(cf.log,'a') as f:
                if DS:
                    print(' Following order: Exp[T2] Exp[T1] |0>',file=f)
                else:
                    print(' Following order: Exp[T1] Exp[T2] |0>',file=f)
            with open(cf.log,'a') as f:
                print('Initial T1 amplitudes:',file=f)
                pprint(theta_list[:ndim1],stream=f)
                print('Intial T2 amplitudes:',file=f)
                pprint(theta_list[ndim1:ndim1+ndim2],stream=f)
        cost_callback(theta_list)
    print_control = 1

    if(maxiter == 0):
        return
    ###################    
    ### perform VQE ###
    ###################    
    if mpi.main_rank:
        with open(cf.log,'a') as f:
            print("Performing VQE for ",method, file=f)

    # Use MPI for evaluating Gradients        
    cost_wrap_mpi = lambda theta_list : cost_mpi(cost_wrap,theta_list)
    jac_wrap_mpi  = lambda theta_list : jac_mpi(cost_wrap,theta_list)

    if(method in ['uhf','phf']):
        opt = minimize(cost_wrap_mpi, kappa_list, jac=jac_wrap_mpi,
                   method=opt_method,options=opt_options,
                   callback=lambda x: cost_callback(x))
    else:  ### correlated methods
        #opt = minimize(cost_wrap, theta_list,    
        #           method=opt_method,options=opt_options,
        #           callback=lambda x: cost_callback(x))
        opt = minimize(cost_wrap_mpi, theta_list,jac=jac_wrap_mpi, 
                   method=opt_method,options=opt_options,
                   callback=lambda x: cost_callback(x))
    
    ### print out final results ###
    theta_or_kappa_list = opt.x
    if (method == "phf"):
        cost_proj(print_control+1,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,opt.x)
        SaveTheta(ndim1,theta_or_kappa_list,cf.kappa_list_file)

    elif (method == "uhf"):
        cost_uhf(print_control+1,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,opt.x)
        SaveTheta(ndim1,theta_or_kappa_list,cf.kappa_list_file)

    elif (method == "uccsd" or method =="sauccsd"):
        cost_uccsd(print_control+1,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,qulacs_hamiltonian,qulacs_s2,method,kappa_list,opt.x,print_amp_thres)
        SaveTheta(ndim,theta_or_kappa_list,cf.theta_list_file)

    elif (method == "uccd"):
        cost_uccd(print_control+1,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,qulacs_hamiltonian,qulacs_s2,kappa_list,opt.x,print_amp_thres)
        SaveTheta(ndim,theta_or_kappa_list,cf.theta_list_file)

    elif method == "opt_uccd" or method == "opt_uccsd":
        cost_opt_ucc(print_level,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,Gen,optk,qulacs_hamiltonian,qulacs_s2,method,theta_list_fix,opt.x)
        if optk:
            SaveTheta(ndim1,theta_or_kappa_list,cf.theta_list_file)
        else:
            SaveTheta(ndim,theta_or_kappa_list,cf.theta_list_file)

    elif (method == "puccsd" or method=="puccd" or method =="opt_puccd" or method == "opt_psauccd"):
        cost_proj(print_control+1,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,method,kappa_list,opt.x,print_amp_thres)
        SaveTheta(ndim,theta_or_kappa_list,cf.theta_list_file)
     
    t2 = time.time()
    cput = t2 - t1
    if mpi.main_rank:
        with open(cf.log,'a') as f:
            print("\n Done: CPU Time =  ",'%15.4f' % cput, file=f)
    
    
    ##################### 
    ### Sampling test ### 
    ##################### 
    #n_term = fermionic_hamiltonian.get_term_count()
    # There are n_term - 1 Pauli operators to measure (identity not counted).
    # <HUg> = \sum_I  h_I <P_I Ug>
    #theta_list = ([ 0.0604642 ,  0.785282  ,  1.28474901, -0.0248133 , -0.01770559,       -0.7854844 , -0.28473988,  0.02487922])
    #sampling.cost_phf_sample(1,n_qubit,n_electron,noa,nob,nva,nvb,anc,qulacs_hamiltonianZ,qulacs_s2Z,qulacs_ancZ,theta_list,1000000)
    #sampling.cost_uhf_sample(1,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,theta_list,1000)
    #cost_uhf(1,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,theta_list)
    #sampling.cost_uhf_sample(1,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,theta_list,100000)
    samplelist = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    samplelist = [10, 100, 1000, 10000]
    #samplelist = [1]
    #samplelist = [1000000]
    #sampling.cost_uhf_sample(1,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,uhf_theta_list,samplelist)
    #sampling.cost_phf_sample(1,n_qubit,n_electron,noa,nob,nva,nvb,rho,anc,qulacs_hamiltonian,qulacs_hamiltonianZ,qulacs_s2Z,qulacs_ancZ,coef0_H,coef0_S2,method,opt.x,samplelist)


    ##################### 
    ### Rotation test ### 
    ##################### 
    # The obtained results are invariant with respect to  occ-occ rotations?
    '''
    if method == 'uccsd':
        Gen = 1
        if Gen:
            kappa_list = np.zeros(norbs*(norbs-1))
        else:
            kappa_list = np.zeros(ndim1)
        theta_list = opt.x 
        cost_wrap = lambda kappa_list : cost_opttest_uccsd(0,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,Gen,qulacs_hamiltonian,qulacs_s2,method,kappa_list,theta_list)[0]
        opt = minimize(cost_wrap, kappa_list,    
               method=opt_method,options=opt_options)
    '''
