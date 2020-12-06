import numpy as np
import itertools
import time
from qulacs import QuantumState, QuantumCircuit
from qulacs.state import inner_product
from . import config as cf
from . import mpilib as mpi
from .utils import SaveTheta, print_amplitudes, print_state

def create_kappalist(ndim1,occ_list,noa,nob,nva,nvb):
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

def create_state(n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,kappa_list,theta_list,occ_list,vir_list):
    state = QuantumState(n_qubit_system)
    from .hflib import set_circuit_rhf,set_circuit_rohf,set_circuit_uhf
    if(noa == nob):
        circuit_rhf = set_circuit_rhf(n_qubit_system,n_electron)
    else:
        circuit_rhf = set_circuit_rohf(n_qubit_system,noa,nob)
    circuit_rhf.update_quantum_state(state)
    theta_list_rho = theta_list/rho
    circuit = set_circuit_uccsdX(n_qubit_system,noa,nob,nva,nvb,DS,theta_list_rho,occ_list,vir_list)
    '''
    if DS:
        if np.linalg.norm(kappa_list) > 0.0001:
            circuit_uhf = set_circuit_uhf(n_qubit_system,noa,nob,nva,nvb,kappa_list)
            circuit_uhf.update_quantum_state(state)
        for i in range(rho):
            circuit.update_quantum_state(state)
    else:
        for i in range(rho):
            circuit.update_quantum_state(state)
        if np.linalg.norm(kappa_list) > 0.0001:
            circuit_uhf = set_circuit_uhf(n_qubit_system,noa,nob,nva,nvb,kappa_list)
            circuit_uhf.update_quantum_state(state)
    '''        
    if np.linalg.norm(kappa_list) > 0.0001:
        circuit_uhf = set_circuit_uhf(n_qubit_system,noa,nob,nva,nvb,kappa_list)
        circuit_uhf.update_quantum_state(state)
    for i in range(rho):
        circuit.update_quantum_state(state)
    return state

def set_circuit_uccsdX(n_qubit_system,noa,nob,nva,nvb,DS,theta_list,occ_list,vir_list):
    ndim1 = noa*nva + nob*nvb
    circuit = QuantumCircuit(n_qubit_system)
    if DS:
        ucc_singlesX(circuit,theta_list,occ_list,vir_list,0)
        ucc_doublesX(circuit,theta_list,occ_list,vir_list,ndim1)
    else:
        ucc_doublesX(circuit,theta_list,occ_list,vir_list,ndim1)
        ucc_singlesX(circuit,theta_list,occ_list,vir_list,0)
    return circuit

def ucc_singlesX(circuit,theta_list,occ_list,vir_list,ndim2=0):
    from .ucclib import single_ope_Pauli
    ia = ndim2
    global ncnot
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]
### alpha ###
    ncnot = 0
    for a in vir_list_a:
        for i in occ_list_a:
            single_ope_Pauli(a,i,circuit,theta_list[ia])
            ia = ia + 1
### beta ###
    for a in vir_list_b:
        for i in occ_list_b:
            single_ope_Pauli(a,i,circuit,theta_list[ia])
            ia = ia + 1

def ucc_doublesX(circuit,theta_list,occ_list,vir_list,ndim1=0):
    from .ucclib import double_ope_Pauli
    ijab = ndim1
    global ncnot
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]
### aa -> aa ###
    ncnot = 0
    for [a,b] in itertools.combinations(vir_list_a,2):
        for [i,j] in itertools.combinations(occ_list_a,2):
            double_ope_Pauli(b,a,j,i,circuit,theta_list[ijab])
            ijab = ijab + 1
### ab -> ab ###
    for b in vir_list_b:
        for a in vir_list_a:
            for j in occ_list_b:
                for i in occ_list_a:
                    double_ope_Pauli(max(b,a),min(b,a),max(j,i),min(j,i),circuit,theta_list[ijab])
                    ijab = ijab + 1
### bb -> bb ###
    for [a,b] in itertools.combinations(vir_list_b,2):
        for [i,j] in itertools.combinations(occ_list_b,2):
            double_ope_Pauli(b,a,j,i,circuit,theta_list[ijab])
            ijab = ijab + 1

def create_HS2S(qulacs_hamiltonian,qulacs_s2,states):
    X_num = len(states)
    H = np.zeros((X_num,X_num),dtype=np.complex)
    S2 = np.zeros((X_num,X_num),dtype=np.complex)
    S = np.zeros((X_num,X_num),dtype=np.complex)
    for i in range(X_num):
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
    nov = no*nv
    aa = i*nv + a
    bb = j*nv + b
    baji = int(nov*(nov-1)/2 - (nov-1-aa)*(nov-aa)/2 + bb)
    # print(aa,bb,nov,baji)
    return baji


def cost_jmucc(print_level,n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,qulacs_hamiltonian,qulacs_s2,theta_lists,threshold):
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
        state = create_state(n_qubit_system,n_electron,noa,nob,nva,nvb,rho,DS,kappa_lists[ndim1*istate:ndim1*(istate+1)], \
        theta_lists[ndim*istate:ndim*(istate+1)],occ_lists[nocc*istate:nocc*(istate+1)],vir_lists[nvir*istate:nvir*(istate+1)])
        states.append(state)
    H,S2,S = create_HS2S(qulacs_hamiltonian,qulacs_s2,states)
    invS = np.linalg.inv(S)
    H_invS = np.dot(invS,H)
    S2_invS = np.dot(invS,S2)
    np.set_printoptions(precision=17)
    en,cvec = np.linalg.eig(H_invS)
    ind   = np.argsort(en.real,-1)
    en    = en.real[ind] 
    cvec  = cvec.real[:,ind] 
    # Renormalize
    Sdig     =cvec.T@S@cvec 
    for istate in range(nstates):
        cvec[:,istate] = cvec[:,istate] / np.sqrt(Sdig[istate,istate].real)
    # Compute <S**2> of each state     
    S2dig    =cvec.T@S2@cvec 
    s2 = []
    for istate in range(nstates):
        s2.append(S2dig[istate,istate].real)

    # compute <S**2> for each state
    t2 = time.time()
    cpu1 = t2-t1
    if print_level == 1 and mpi.main_rank:
        cput = t2 - cf.t_old
        cf.t_old = t2
        with open(cf.log,'a') as f:
            for istate in range(nstates):
                print("E[{i:01}] = {en:.8f} (<S**2> = {s2:.5f})  ".format(i=istate,en=en[istate],s2=s2[istate]), end="", file=f)
            print("  CPU Time = ", '%2.5f' % cput, " (%2.5f / step)" % cpu1, file=f)
        SaveTheta((ndim1+ndim2)*nstates,theta_lists,cf.tmp)
        # cf.iter_threshold = 0
    if print_level > 1 and mpi.main_rank:
        with open(cf.log,'a') as f:
            print("\n------------------------------------",file=f)
        for istate in range(nstates):
            with open(cf.log,'a') as f:
                print("JM Basis   {:01}".format(istate),file=f)
            print_state(states[istate])
            with open(cf.log,'a') as f:
                print("",file=f)
        with open(cf.log,'a') as f:
            print("Coefficients: ", cvec,file=f)
            print("------------------------------------\n\n",file=f)
            print("###############################################",file=f)
            print("#                  JM states                  #",file=f)
            print("###############################################",file=f)

        for istate in range(nstates):
            with open(cf.log,'a') as f:
                print("State        :  {:01} ".format(istate),file=f)
                print("E            : {:.8f} ".format(en[istate]),file=f)
                print("<S**2>       : {:.5f} ".format(s2[istate]),file=f)
                print("Superposition: ",file=f)
            spstate = QuantumState(n_qubit_system)
            spstate.multiply_coef(0)
            for jstate in range(nstates):
                state = states[jstate].copy()
                coef  = cvec[jstate,istate]
                state.multiply_coef(coef)
                spstate.add_state(state)

            print_state(spstate)

        with open(cf.log,'a') as f:
            print("###############################################",file=f)
    cost = 0        
    for istate in range(nstates):
        cost += cf.multi_weights[istate] * en[istate]
    return cost

def int2occ(state_int):
    '''
        Given an integer, find the index for 1 in base-2 (occ_list)
    '''
    occ_list=[]
    k = 0
    while k < state_int:
        kk = 1 << k
        if kk & state_int >0:
            occ_list.append(k)
        k += 1
    return occ_list
