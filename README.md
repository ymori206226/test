# quket
"""
 Quantum Computing Simulator Ver Beta
     Copyright 2019-2020 Takashi Tsuchimochi, Yuto Mori, Takahiro Yoshikura. All rights Reserved.

 This suite of programs simulates quantum computing for electronic Hamiltonian.
 It currently supports the following methods:
   
   - Ground state VQE

"""

Requisites:
-----------

The following external modules and libraries are required.
 - openfermion        0.10.0 
 - openfermionpyscf   0.4    
 - Qulacs             0.1.9   
 - numpy
 - scipy 

All the necessary libtaries are installed under Python3.8 in Ravel. 
Type "pip3.8 list" to check this.
You may run this program in your local machine too.
The environment for Titan is under construction.



How to use:
-----------

(1) Create an input file as ***.inp (quick instruction below)
(2) Run main.py with python3.8

     python3.8 main.py *** 

The result is logged out in ***.log.
If run on Ravel, it is recommended to add "nohup" option to prevent it from stopping when you log out the workstation (This is not necessary for Titan).



File descriptions:
------------------

***.inp is an input file.
***.chk contains integrals (and energy) from PySCF.
Depending on the method you choose in ***.inp, there will be also ***.theta and/or ***.kappa. 
***.theta stores t-amplitudes from UCC. 
***.kappa stores kappa-amplitudes for orbital rotation.

You may read these files for initial guesses of subsequent calculations.




How to write ***.inp: 
---------------------

Simply put options listed below.
The order does not matter.
Sample inputs are found in samples directory.

*******************
* MINIMUM OPTIONS *
*******************
method        : method for VQE, either of  uhf, uccsd, sauccsd, phf, opt_puccd, etc.
geometry      : a sequence of 'atom x y z' (this needs to be re-formatted more cleanly in future).
n_electrons   : number of electrons 
n_orbitals    : number of spatial orbitals, Nqubit is twice this value


*Putting '@@@' in lines separates jobs. This enables multiple jobs with a single input file.
 CAUTION!! The options from previous jobs remain the same unless redefined.

*Options that have a default value (see main.py for details)
# For PySCF
basis               :Gaussian Basis Set 
multiplicity        :Spin multiplicity (defined as Nalpha - Nbeta + 1) 
charge              :Electron charge (0 for neutral) 
pyscf_guess         :Guess for pyscf: 'minao', 'chkfile'

# For qulacs (VQE part)
print_level         :Printing level
mix_level           :Number of pairs of orbitals to be mixed (to break symmetry)
rho                 :Trotter number 
kappa_guess         :Guess for kappa: 'zero', 'read', 'mix', 'random'
theta_guess         :Guess for T1 and T2 amplitudes: 'zero', 'read', 'mix', 'random'
Kappa_to_T1         :Flag to use ***.kappa file (T1-like) for initial guess of T1
spin                :Spin quantum number for spin-projection
ng                  :Number of grid points for spin-projection
DS                  :Ordering of T1 and T2: 0 for Exp[T1]Exp[T2], 1 for Exp[T2]Exp[T1]
print_amp_thres     :Threshold for T amplitudes to be printed
constraint_lambda   :Constraint for spin 

# For scipy.optimize
opt_method          :Method for optimization

gtol                :Convergence criterion based on gradient

ftol                :Convergence criterion based on energy (cost)

eps                 :Numerical step     

maxiter             :Maximum iterations: if 0, skip VQE and only PySCF --> JW-transformation is carried out. 


