import numpy as np
import time
import itertools
from .init import int2occ
from . import config as cf
from qulacs import QuantumState, QuantumCircuit
from scipy.special import comb
from .fileio import prints, print_state, SaveTheta, printmat
from .utils import root_inv

def ucc_Gsingles_listversion(circuit,a_list,i_list,theta_list,theta_index):
    """ Function
    generalized singles. i_list -> a_list
    Author(s): Yuto Mori
    """
    from .ucclib import single_ope_Pauli
    if a_list == i_list:
        ai_list = itertools.combinations(a_list,2)      
    else:
        ai_list = itertools.product(a_list,i_list)
    for [a,i] in ai_list:
        single_ope_Pauli(a,i,circuit,theta_list[theta_index])
        theta_index = theta_index + 1
        single_ope_Pauli(a+1,i+1,circuit,theta_list[theta_index])
        theta_index = theta_index + 1
    return circuit, theta_index

def ucc_Gdoubles_listversion(circuit,b_list,a_list,j_list,i_list,theta_list,theta_index):
    """ Function
    Author(s): Yuto Mori
    """
    from .expope import Gdouble_ope
    
    ab = []
    ij = []
    if b_list == a_list:
        ab = itertools.combinations(b_list,2)
    else:
        ab = itertools.product(a_list,b_list)
    if j_list == i_list:
        ij = itertools.combinations(j_list,2)
    else:
        ij = itertools.product(i_list,j_list)
    
    abij = itertools.product(ab,ij)
    for [[a,b],[i,j]] in abij:
        max_id = max(a,b,i,j)
        if a != i or b != j:
            if b == max_id:
                Gdouble_ope(b,a,j,i,circuit,theta_list[theta_index])
                theta_index = theta_index + 1
            elif a == max_id:
                Gdouble_ope(a,b,i,j,circuit,theta_list[theta_index])
                theta_index = theta_index + 1
            elif i == max_id:
                Gdouble_ope(i,j,a,b,circuit,theta_list[theta_index])
                theta_index = theta_index + 1
            elif j == max_id:
                Gdouble_ope(j,i,b,a,circuit,theta_list[theta_index])
                theta_index = theta_index + 1
    return circuit, theta_index

def icmr_ucc_singles(circuit,n_qubits_system,v_n,a_n,c_n,theta_list,ndim2=0):
    """ Function
    Author(s): Yuto Mori
    """
    theta_index = ndim2
    vir_list_a  = [i for i in range(c_n+a_n,n_qubits_system) if i%2 == 0]
    # vir_list_b  = [i for i in range(c_n+a_n,n_qubits_system) if i%2 == 1]
    act_list_a  = [i for i in range(c_n,c_n+a_n) if i%2 == 0]
    # act_list_b  = [i for i in range(c_n,c_n+a_n) if i%2 == 1]
    core_list_a = [i for i in range(c_n) if i%2 == 0]
    # core_list_b = [i for i in range(c_n) if i%2 == 1]

    circuit, theta_index = ucc_Gsingles_listversion(circuit,act_list_a,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gsingles_listversion(circuit,vir_list_a,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gsingles_listversion(circuit,act_list_a,act_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gsingles_listversion(circuit,vir_list_a,act_list_a,theta_list,theta_index)

def icmr_ucc_doubles(circuit,n_qubits_system,v_n,a_n,c_n,theta_list,ndim1=0):
    """ Function
    Author(s): Yuto Mori
    """
    theta_index = ndim1
    vir_list_a  = [i for i in range(c_n+a_n,n_qubits_system) if i%2 == 0]
    vir_list_b  = [i for i in range(c_n+a_n,n_qubits_system) if i%2 == 1]
    act_list_a  = [i for i in range(c_n,c_n+a_n) if i%2 == 0]
    act_list_b  = [i for i in range(c_n,c_n+a_n) if i%2 == 1]
    core_list_a = [i for i in range(c_n) if i%2 == 0]
    core_list_b = [i for i in range(c_n) if i%2 == 1]

    ### aaaa ###
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_a,act_list_a,core_list_a,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_a,act_list_a,core_list_a,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_a,vir_list_a,core_list_a,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_a,act_list_a,act_list_a,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_a,act_list_a,act_list_a,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_a,vir_list_a,act_list_a,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_a,act_list_a,act_list_a,act_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_a,vir_list_a,act_list_a,act_list_a,theta_list,theta_index)
    ### bbbb ###
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,act_list_b,core_list_b,core_list_b,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,act_list_b,core_list_b,core_list_b,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,vir_list_b,core_list_b,core_list_b,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,act_list_b,act_list_b,core_list_b,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,act_list_b,act_list_b,core_list_b,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,vir_list_b,act_list_b,core_list_b,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,act_list_b,act_list_b,act_list_b,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,vir_list_b,act_list_b,act_list_b,theta_list,theta_index)
    ### aabb ###
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,vir_list_a,core_list_b,act_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,vir_list_a,core_list_b,act_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,vir_list_a,act_list_b,act_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,vir_list_a,act_list_b,act_list_a,theta_list,theta_index)

    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,act_list_a,core_list_b,act_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,act_list_a,core_list_b,act_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,act_list_a,act_list_b,act_list_a,theta_list,theta_index)

    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,vir_list_a,core_list_b,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,vir_list_a,core_list_b,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,vir_list_a,act_list_b,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,vir_list_a,act_list_b,core_list_a,theta_list,theta_index)

    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,act_list_a,core_list_b,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,act_list_a,core_list_b,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,act_list_a,act_list_b,core_list_a,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,vir_list_b,act_list_a,act_list_b,core_list_a,theta_list,theta_index)

    if cf.act2act_ops:
        circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_a,act_list_a,act_list_a,act_list_a,theta_list,theta_index)
        circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,act_list_a,act_list_b,act_list_a,theta_list,theta_index)
        circuit, theta_index = ucc_Gdoubles_listversion(circuit,act_list_b,act_list_b,act_list_b,act_list_b,theta_list,theta_index)



    
def set_circuit_ic_mrucc(n_qubits_system,v_n,a_n,c_n,DS,theta_list,ndim1):
    """ Function
    Author(s): Yuto Mori
    """
    circuit = QuantumCircuit(n_qubits_system)

    if DS:
        icmr_ucc_singles(circuit,n_qubits_system,v_n,a_n,c_n,theta_list,0)
        icmr_ucc_doubles(circuit,n_qubits_system,v_n,a_n,c_n,theta_list,ndim1)
    else:
        icmr_ucc_doubles(circuit,n_qubits_system,v_n,a_n,c_n,theta_list,ndim1)
        icmr_ucc_singles(circuit,n_qubits_system,v_n,a_n,c_n,theta_list,0)
    return circuit

def create_icmr_uccsd_state(n_qubits_system,v_n,a_n,c_n,rho,DS,theta_list,det,ndim1):
    """ Function
    Author(s): Yuto Mori
    """
    state = QuantumState(n_qubits_system)
    state.set_computational_basis(det)

    theta_list_rho = theta_list/rho
    circuit = set_circuit_ic_mrucc(n_qubits_system,v_n,a_n,c_n,DS,theta_list_rho,ndim1)
    for i in range(rho):
        circuit.update_quantum_state(state)

    if cf.SpinProj:
        from .phflib import S2Proj
        state_P = S2Proj(state)
        return state_P
    else:
        return state

def calc_num_ic_theta(n_qubits_system,vir_num,act_num,core_num):
    """ Function
    Calc number of ndim1,ndim2 for icmr_uccsd
    Author(s): Yuto Mori
    """
    v = vir_num/2
    a = act_num/2
    c = core_num/2
    ### ndim1 ###
    ndim1 = c*a + c*v + comb(a,2) + a*v
    ndim1 = ndim1 * 2

    ### ndim2 ###
    if cf.act2act_ops:
        ndim2ab = (a*a + a*v*2 + v*v)*(a*a + c*a*2 + c*c) - a*a
        ndim2aa = (comb(c,2) + c*a + comb(a,2))*(comb(a,2) + comb(v,2) + a*v) - comb(a,2)
    else:
        ndim2ab = (a*a + a*v*2 + v*v)*(a*a + c*a*2 + c*c) - a**4
        ndim2aa = (comb(c,2) + c*a + comb(a,2))*(comb(a,2) + comb(v,2) + a*v) - comb(a,2)**2
    ndim2 = ndim2ab + ndim2aa*2
    return int(ndim1),int(ndim2)

def cost_ic_mrucc(print_level,n_qubits_system,n_electrons,vir_num,act_num,core_num,rho,DS,qulacs_hamiltonian,qulacs_s2,theta_list,threshold):
    """ Function
    Author(s): Yuto Mori
    """
    import copy
    t1 = time.time()
    nstates = len(cf.multi_weights)
    # core_num = n_qubits_system
    # vir_index = 0
    # for istate in range(nstates):
    #     ### Read state integer and extract occupied/virtual info
    #     occ_list_tmp = int2occ(cf.multi_states[istate])
    #     vir_tmp = occ_list_tmp[-1] + 1
    #     for ii in range(len(occ_list_tmp)):
    #         if ii == occ_list_tmp[ii]: core_tmp = ii + 1
    #     vir_index = max(vir_index,vir_tmp)
    #     core_num = min(core_num,core_tmp)
    # vir_num = n_qubits_system - vir_index
    # act_num = n_qubits_system - core_num - vir_num

    ndim1, ndim2 = calc_num_ic_theta(n_qubits_system,vir_num,act_num,core_num)

    states = []
    for istate in range(nstates):
        det = cf.multi_states[istate]
        state = create_icmr_uccsd_state(n_qubits_system,vir_num,act_num,core_num,rho,DS,theta_list,det,ndim1)
        states.append(state)
    from .jmucc import create_HS2S
    H,S2,S = create_HS2S(qulacs_hamiltonian,qulacs_s2,states)

    root_invS = root_inv(S.real)
    H_ortho = root_invS.T@H@root_invS
    nstates0 = root_invS.shape[1]

    en,dvec = np.linalg.eig(H_ortho)

    ind   = np.argsort(en.real,-1)
    en    = en.real[ind] 
    dvec  = dvec[:,ind] 

    cvec  = root_invS@dvec

    S2dig = cvec.T@S2@cvec 

    s2 = []
    for istate in range(nstates0):
        s2.append(S2dig[istate,istate].real)

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
        SaveTheta((ndim1+ndim2),theta_list,cf.tmp)
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
            prints("ic Basis   {:01}".format(istate))
            print_state(states[istate])
            prints("")
        printmat(cvec.real,name="Coefficients: ")
        prints("------------------------------------\n\n")
        prints("###############################################")
        prints("#                  ic states                  #")
        prints("###############################################",end="")

        for istate in range(nstates0):
            prints("\n State        :  {:01} ".format(istate))
            prints(" E            : {:.8f} ".format(en[istate]))
            prints(" <S**2>       : {:.5f} ".format(s2[istate]))
            prints(" Superposition: ")
            spstate = QuantumState(n_qubits_system)
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