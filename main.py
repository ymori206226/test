"""
 Quantum Computing Simulator Ver Beta
     Copyright 2019-2020 Takashi Tsuchimochi, Yuto Mori, Takahiro Yoshikura. All rights Reserved.

 This suite of programs simulates quantum computing for electronic Hamiltonian.
 It currently supports the following methods:
   
   - Ground state VQE

"""

import os
import sys
import datetime
import re
from src import  utils, config
with open(config.log,'w') as f:
    print(datetime.datetime.now(),file=f)  # time stamp

#############################
###    Default options    ###   
#############################

# PySCF
basis            = "sto-3G"         #Gaussian Basis Set 
multiplicity     = 1                #Spin multiplicity (defined as Nalpha - Nbeta + 1) 
charge           = 0                #Electron charge (0 for neutral) 
pyscf_guess = 'minao'               #Guess for pyscf: 'minao', 'chkfile'
run_fci          = 1 
# (Hubbard mode is entered when basis = hubbard, 2d-hubbard)
hubbard_u        = 1                #Hubbard model (U interaction) 
hubbard_nx       = 0                #Number of lattices in x direction: not needed if n_orbitals is defined (see below) 
hubbard_ny       = 1                #Number of lattices in y direction (needed only for 2d-hubbard) 

# qulacs (VQE part)
print_level = 1                     #Printing level
mix_level = 0                       #Number of pairs of orbitals to be mixed (to break symmetry)
rho              = 1                #Trotter number 
kappa_guess = 'zero'                #Guess for kappa: 'zero', 'read', 'mix', 'random'
theta_guess = 'zero'                #Guess for T1 and T2 amplitudes: 'zero', 'read', 'mix', 'random'
Kappa_to_T1 = 0                     #Flag to use ***.kappa file (T1-like) for initial guess of T1
spin = 1                            #Spin quantum number for spin-projection
ng   = 2                            #Number of grid points for spin-projection
DS = 0                              #Ordering of T1 and T2: 0 for Exp[T1]Exp[T2], 1 for Exp[T2]Exp[T1]
print_amp_thres = 1e-2              #Threshold for T amplitudes to be printed
config.constraint_lambda = 0        #Constraint for spin 

# scipy.optimize
opt_method = "L-BFGS-B"             #Method for optimization
gtol = 1e-12                        #Convergence criterion based on gradient
ftol = 1e-12                        #Convergence criterion based on energy (cost)
eps  = 1e-6                         #Numerical step     
maxiter = 10000                     #Maximum iterations: if 0, skip VQE and only JW-transformation is carried out. 


# other parameters that do not have default values
    # method        : VQE ansatz 
    # n_electrons   : number of electrons 
    # n_orbitals    : number of spatial orbitals, Nqubit is twice this value
    # geometry      : a sequence of 'atom x y z' (this needs to be written more cleanly in future).

# Putting '@@@' in lines separates multiple jobs in one input file.
# (the options from previous jobs will be used unless redefined)

######################################
###    Start reading input file    ###   
######################################

num_lines = sum(1 for line in open(config.input_file))  # number of lines in the input file
Finish = False
job_no = 0
geom_update = False
while Finish == False:
    job_no += 1
    job_k = 0
    Finish = True
    f = open(config.input_file)
    lines = f.readlines()
    iline = 0
    while iline < num_lines:  
        line=lines[iline].replace('=',' ')
        line=line.replace(':',' ')
        line=line.replace(',',' ')
        line=line.replace("'",' ')
        words = [x.strip() for x in line.split() if not line.strip() == '']
        len_words = len(words)
        if(len_words>0 and words[0][0] not in {'!','#'}):
            if words[0].lower() == "basis":
                if(len_words>2):
                    atoms=[]
                    atom_basis=[]
                    for i in  range(0, int(len_words/2)):
                        atoms.append(words[2*i+1])
                        atom_basis.append(words[2*i+2])
                    basis = dict(zip(atoms,atom_basis))
                else:
                    basis = words[1].lower()
            elif words[0].lower() == "method":
                method = words[1]
            elif words[0].lower() == "multiplicity":
                multiplicity = int(words[1])
            elif words[0].lower() == "charge":
                charge = int(words[1])
            elif words[0].lower() == "rho":
                rho = int(words[1])
            elif words[0].lower() == "run_fci":
                run_fci = int(words[1])
            elif words[0].lower() == "print_level":
                print_level = int(words[1])
            elif words[0].lower() == "opt_method":
                opt_method = words[1]
            elif words[0].lower() == "mix_level":
                mix_level = int(words[1])
            elif words[0].lower() == "eps":
                eps = float(words[1])
            elif words[0].lower() == "gtol":
                gtol = float(words[1])
            elif words[0].lower() == "ftol":
                ftol = float(words[1])
            elif words[0].lower() == "print_amp_thres":
                print_amp_thres = float(words[1])
            elif words[0].lower() == "maxiter":
                maxiter = int(words[1])
            elif words[0].lower() == "pyscf_guess":
                pyscf_guess = words[1]
            elif words[0].lower() == "kappa_guess":
                kappa_guess = words[1]
            elif words[0].lower() == "theta_guess":
                theta_guess = words[1]
            elif words[0].lower() == "1PDM":
                config.Do1PDM = words[1]
                # 1PDM = 1  (TRUE)
                # 1PDM = 0  (FALSE)
            elif words[0].lower() == "kappa_to_t1":
                Kappa_to_T1 = int(words[1])
            elif words[0].lower() == "n_electrons":
                n_electrons = int(words[1])
            elif words[0].lower() == "n_orbitals":
                n_orbitals = int(words[1])
            elif words[0].lower() == "spin":
                spin = int(words[1])
            elif words[0].lower() == "ng":
                ng = int(words[1])
            elif words[0].lower() == "ds":
                DS = int(words[1])
            elif words[0].lower() == "mix_level":
                mix_level = int(words[1])
            elif words[0].lower() == "lambda":
                config.constraint_lambda = float(words[1])
            elif words[0].lower() == "geometry":
                # Reading Geometry ... 
                if(job_k == job_no-1):
                    geom_update = True

                iatom  = 0
                geometry = []
                while iatom >= 0:
                    iline += 1
                    geom = lines[iline].replace('=',' ')
                    geom = [x.strip() for x in geom.split() if not geom.strip() == '']
                    if geom[0] in config.PeriodicTable:
                        xyz = []
                        xyz.append(float(geom[1]))
                        xyz.append(float(geom[2]))
                        xyz.append(float(geom[3]))
                        atom = []
                        atom.append(geom[0])
                        atom.append(xyz)
                        geometry.append(atom)
                    else:
                        iatom = -1
                        iline += -1

            elif words[0].lower() == "npar":
                config.nthreads = words[1]
            elif words[0].lower() == "@@@":
                job_k += 1
                if job_k == job_no:
                    Finish = False
                    break
            elif words[0].lower() == "hubbard_u":
                hubbard_u = float(words[1])
            elif words[0].lower() == "hubbard_nx":
                hubbard_nx = int(words[1])
            elif words[0].lower() == "hubbard_ny":
                hubbard_ny = int(words[1])
            else:
                with open(config.log,'a') as f:
                    print('No option "%s"' % words[0],file=f)
                exit()

        iline += 1       # read next line... 

    f.close()
    from src import driver

    if n_electrons == 0 :
        utils.error('# electrons = 0 !')
    if basis != 'hubbard':
        if n_orbitals == 0 :
            utils.error('# orbitals = 0 !')
    else:
        if hubbard_nx == 0 :
            utils.error('Hubbard model but hubbard_nx is not defined!')
        n_orbitals = hubbard_nx * hubbard_ny

    if opt_method == "L-BFGS-B":
        opt_options = {"disp": True, "maxiter": maxiter, "gtol": gtol, "ftol": ftol, "eps": eps}
    elif opt_method == "BFGS":
        opt_options = {"disp": True, "maxiter": maxiter, "gtol": gtol, "eps": eps}

    if pyscf_guess == 'read':
       pyscf_guess = 'chkfile' 
    
    with open(config.log,'a') as f:
        print('+-------------+',file=f)
        print('|  Job # %3d  |' % job_no,file=f)
        print('+-------------+',file=f)

    if(basis != 'hubbard' and geom_update):
    # Set Jordan-Wigner Hamiltonian and S2 operators using PySCF and Open-Fermion
        jw_hamiltonian, jw_s2 = driver.get_hamiltonian(geometry, basis, multiplicity, charge, n_electrons, n_orbitals, pyscf_guess) 
        geom_update = False
        with open(config.log,'a') as f:
            print('JW_Hamiltonian done',file=f)
    if(basis == 'hubbard'):
        jw_hamiltonian, jw_s2 = driver.get_hubbard(hubbard_u,hubbard_nx,hubbard_ny,n_electrons,run_fci)
        with open(config.log,'a') as f:
            print('Hubbard model: nx = %d  ' % hubbard_nx,  'ny = %d  ' % hubbard_ny,  'U = %2.2f' % hubbard_u, file=f)

    # If maxiter = 0, skip the VQE part. This option is useful to do PySCF for different geometries 
    # (to read and utilize initial guess HF orbitals, which sometimes can change by occupying the wrong orbitals).
    if(maxiter == 0):
        continue

    # VQE part
    driver.VQE_driver(jw_hamiltonian,jw_s2,n_electrons, n_orbitals, method,  
    kappa_guess,theta_guess,mix_level, rho, DS, opt_method, opt_options, print_level, maxiter,
    Kappa_to_T1, spin, ng, print_amp_thres)
    os.remove(config.tmp)
    # VQE part done, go to the next job.

