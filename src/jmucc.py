"""
#######################
#        quket        #
#######################

jmucc.py

Multi-reference UCC.
Jeziorski-Monkhorst UCC.

"""
import numpy as np
import itertools
import time
from qulacs import QuantumState, QuantumCircuit
from qulacs.state import inner_product
from . import config as cf
from . import mpilib as mpi
from .fileio import SaveTheta, print_amplitudes, print_state, prints, printmat
from .phflib import S2Proj
from .utils  import root_inv
from .ucclib import create_uccsd_state
from .init   import get_occvir_lists, int2occ

def create_kappalist(ndim1,occ_list,noa,nob,nva,nvb):
    """ Function
    Create kappalist from occ_list, which stores the occupied qubits.

    Author(s):  Yuto Mori
    """
    kappa = np.zeros(ndim1)
    occ_hf = set(range(len(occ_list)))
    dup = occ_hf & set(occ_list)
    cre_set = list(set(occ_list)-dup)
    ann_set = list(occ_hf-dup)
    kappalist = []
    for c in cre_set:
        if c%2 == 0:
            for a in ann_set:
                if a%2 == 0:
                    ann_set.remove(a)
                    kappalist.append(int(a/2 + noa*(c/2-noa)))
        else:
            for a in ann_set:
                if a%2 == 1:
                    ann_set.remove(a)
                    kappalist.append(int((a-1)/2 + nob*((c-1)/2-nob) + noa*nva))
    for i in kappalist:
        kappa[i] = np.pi/2
    return kappa

def create_HS2S(qulacs_hamiltonian,qulacs_s2,states):
    X_num = len(states)
    H = np.zeros((X_num,X_num),dtype=np.complex)
    S2 = np.zeros((X_num,X_num),dtype=np.complex)
    S = np.zeros((X_num,X_num),dtype=np.complex)
    for i in range(X_num):
#        prints('\nState {} in create_HS2S'.format(i))
#        print_state(states[i])
#        prints('S**2 = ',qulacs_s2.get_expectation_value(states[i]))
        for j in range(i+1):
            H[i,j] = qulacs_hamiltonian.get_transition_amplitude(states[i],states[j])
            S2[i,j] = qulacs_s2.get_transition_amplitude(states[i],states[j])
            S[i,j] = inner_product(states[i],states[j])
            if i != j:
                H[j,i] = H[i,j]
                S2[j,i] = S2[i,j]
                S[j,i] = S[i,j]
    return H,S2,S


## Spin Adapted ##
def create_sa_state(n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,kappa_list,theta_list,occ_list,vir_list):
    """ Function
    Create a spin-free Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    state = QuantumState(n_qubit_system)
    from .hflib import set_circuit_rhf,set_circuit_rohf,set_circuit_uhf
    if(noa == nob):
        circuit_rhf = set_circuit_rhf(n_qubit_system,n_electron)
    else:
        circuit_rhf = set_circuit_rohf(n_qubit_system,noa,nob)
    circuit_rhf.update_quantum_state(state)
    theta_list_rho = theta_list/rho
    circuit = set_circuit_sauccsdX(n_qubit_system,noa,nob,nva,nvb,DS,theta_list_rho,occ_list,vir_list)
    if np.linalg.norm(kappa_list) > 0.0001:
        circuit_uhf = set_circuit_uhf(n_qubit_system,noa,nob,nva,nvb,kappa_list)
        circuit_uhf.update_quantum_state(state)
    for i in range(rho):
        circuit.update_quantum_state(state)
    return state

def set_circuit_sauccsdX(n_qubit_system,noa,nob,nva,nvb,DS,theta_list,occ_list,vir_list):
    """ Function
    Prepare a Quantum Circuit for a spin-free Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ndim1 = noa*nva
    circuit = QuantumCircuit(n_qubit_system)
    if DS:
        ucc_sa_singlesX(circuit,theta_list,occ_list,vir_list,0)
        ucc_sa_doublesX(circuit,theta_list,occ_list,vir_list,ndim1)
    else:
        ucc_sa_doublesX(circuit,theta_list,occ_list,vir_list,ndim1)
        ucc_sa_singlesX(circuit,theta_list,occ_list,vir_list,0)
    return circuit

def ucc_sa_singlesX(circuit,theta_list,occ_list,vir_list,ndim2=0):
    """ Function
    Prepare a Quantum Circuit for the single exictation part of a spin-free Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    from .ucclib import single_ope_Pauli
    ia = ndim2
    global ncnot
    occ_list_a = [i for i in occ_list if i%2 == 0]
    # occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    # vir_list_b = [i for i in vir_list if i%2 == 1]
### alpha (& beta) ###
    ncnot = 0
    for a in vir_list_a:
        for i in occ_list_a:
            single_ope_Pauli(a,i,circuit,theta_list[ia])
            single_ope_Pauli(a+1,i+1,circuit,theta_list[ia])
            ia = ia + 1

def ucc_sa_doublesX(circuit,theta_list,occ_list,vir_list,ndim1=0):
    """ Function
    Prepare a Quantum Circuit for the double exictation part of a spin-free Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    from .ucclib import double_ope_Pauli
    ijab = ndim1
    global ncnot
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]
### aa or bb ##
    ncnot = 0
    for [a,b] in itertools.combinations(vir_list_a,2):
        for [i,j] in itertools.combinations(occ_list_a,2):
            double_ope_Pauli(b,a,j,i,circuit,theta_list[ijab])
            double_ope_Pauli(b+1,a+1,j+1,i+1,circuit,theta_list[ijab])
            ijab = ijab + 1
### ab ###
    no = len(occ_list_a)
    nv = len(vir_list_a)
    for b in vir_list_b:
        for a in vir_list_a:
            for j in occ_list_b:
                for i in occ_list_a:
                    b_i = vir_list_b.index(b)
                    a_i = vir_list_a.index(a)
                    j_i = occ_list_b.index(j)
                    i_i = occ_list_a.index(i)
                    baji = get_baji(b_i,a_i,j_i,i_i,no,nv)
                    double_ope_Pauli(max(b,a),min(b,a),max(j,i),min(j,i),circuit,theta_list[ijab+baji])
    
def get_baji(b,a,j,i,no,nv):
    """ Function
    Get index for b,a,j,i 

    Author(s):  Yuto Mori
    """
    nov = no*nv
    aa = i*nv + a
    bb = j*nv + b
    baji = int(nov*(nov-1)/2 - (nov-1-aa)*(nov-aa)/2 + bb)
    return baji


def cost_jmucc(print_level,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,qulacs_hamiltonian,qulacs_s2,theta_lists,threshold,SpinProj=None):
    """ Function
    Cost function of Jeziorski-Monkhorst UCCSD.

    Author(s):  Yuto Mori, Takashi Tsuchimochi
    """
    import copy
    t1 = time.time()
    nstates = len(cf.multi_weights)
    ndim1 = noa*nva + nob*nvb
    ndim2aa = int(noa*(noa-1)*nva*(nva-1)/4)  
    ndim2ab = int(noa*nob*nva*nvb)  
    ndim2bb = int(nob*(nob-1)*nvb*(nvb-1)/4)  
    ndim2 = ndim2aa + ndim2ab + ndim2bb
    ndim = ndim1 + ndim2
    nocc = noa + nob
    nvir = nva + nvb
    occ_lists = []
    vir_lists = []
    for istate in range(nstates):
        ### Read state integer and extract occupied/virtual info
        occ_list_tmp = int2occ(cf.multi_states[istate])
        vir_list_tmp = [i for i in range(n_qubit_system) if i not in occ_list_tmp]
        occ_lists.extend( occ_list_tmp )
        vir_lists.extend( vir_list_tmp )

    ### Prepare kappa_lists
    kappa_lists = []    
    for istate in range(nstates):
        kappa_list = create_kappalist(ndim1,occ_lists[nocc*istate:nocc*(istate+1)],noa,nob,nva,nvb)
        kappa_lists.extend(kappa_list)

    ### Prepare JM basis
    states = []
    for istate  in range(nstates):
        det = cf.multi_states[istate]
        state = create_uccsd_state(n_qubit_system,noa,nob,nva,nvb,rho,DS,\
        theta_lists[ndim*istate:ndim*(istate+1)],det,SpinProj=SpinProj)
#        prints('\n State {}?'.format(istate))    
#        print_state(state)
        states.append(state)
    H,S2,S = create_HS2S(qulacs_hamiltonian,qulacs_s2,states)

#    invS   = np.linalg.inv(S)
#    H_invS = np.dot(invS,H)
#    np.set_printoptions(precision=17)
#    en,cvec = np.linalg.eig(H_invS)
#    printmat(en,name="energy")
#    printmat(cvec,name="cvec")

    root_invS = root_inv(S.real)
    H_ortho = root_invS.T@H@root_invS
    nstates0 = root_invS.shape[1]
    #print(nstates0)

    en,dvec = np.linalg.eig(H_ortho)

    ind   = np.argsort(en.real,-1)
    en    = en.real[ind] 
    dvec  = dvec[:,ind] 
    #printmat(en,name="energy")
    #printmat(dvec,name="dvec")
    cvec  = root_invS@dvec

    # Renormalize
#    Sdig     =cvec.T@S@cvec 
    #printmat(Sdig,name="Sdig")
#    for istate in range(nstates):
#        cvec[:,istate] = cvec[:,istate] / np.sqrt(Sdig[istate,istate].real)
    # Compute <S**2> of each state     
    S2dig    =cvec.T@S2@cvec 
#    printmat(S2dig)

    s2 = []
    for istate in range(nstates0):
        s2.append(S2dig[istate,istate].real)

    # compute <S**2> for each state
    t2 = time.time()
    cpu1 = t2-t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints("{cyc:5}:".format(cyc=cf.icyc),end="") 
        for istate in range(nstates0):
            prints("  E[{i:01}] = {en:.8f} (<S**2> = {s2: 7.5f})  ".format(i=istate,en=en[istate],s2=s2[istate]), end="")
        prints("  CPU Time = ", '%5.2f' % cput, " (%2.2f / step)" % cpu1)
        SaveTheta((ndim1+ndim2)*nstates,theta_lists,cf.tmp)
        # cf.iter_threshold = 0
    if print_level > 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        prints("Final:",end="")
        for istate in range(nstates0):
            prints("  E[{i:01}] = {en:.8f} (<S**2> = {s2: 7.5f})  ".format(i=istate,en=en[istate],s2=s2[istate]), end="")
        prints("  CPU Time = ", '%5.2f' % cput, " (%2.2f / step)" % cpu1)
        prints("\n------------------------------------")
        for istate in range(nstates):
            prints("JM Basis   {:01}".format(istate))
            print_state(states[istate])
            prints("")
        printmat(cvec.real,name="Coefficients: ")
        prints("------------------------------------\n\n")
        prints("###############################################")
        prints("#                  JM states                  #")
        prints("###############################################",end="")

        for istate in range(nstates0):
            prints("\n State        :  {:01} ".format(istate))
            prints(" E            : {:.8f} ".format(en[istate]))
            prints(" <S**2>       : {:.5f} ".format(s2[istate]))
            prints(" Superposition: ")
            spstate = QuantumState(n_qubit_system)
            spstate.multiply_coef(0)
            for jstate in range(nstates0):
                state = states[jstate].copy()
                coef  = cvec[jstate,istate]
                state.multiply_coef(coef)
                spstate.add_state(state)

            print_state(spstate)

        prints("###############################################")
    cost = 0        
    norm = 0
    for istate in range(nstates0):
        norm += cf.multi_weights[istate]
        cost += cf.multi_weights[istate] * en[istate]
    cost /= norm
    return cost
