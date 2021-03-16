import time
import copy
import itertools

import numpy as np
from qulacs import QuantumState, QuantumCircuit

from . import config as cf
from .utils import root_inv
from .jmucc import create_HS2S
from .expope import Gdouble_ope
from .ucclib import single_ope_Pauli
from .fileio import prints, print_state, SaveTheta, printmat


def calc_num_v_a_c(Quket):
    """
    Calculation the number of virtual, active, core oribitals for ic-MRUCC.

    [Ex. if multi is (00001111 & 00110011), v_n = 1, a_n = 2, c_n = 1 ]

    Args:
        Quket(QuketData)

    Return:
        v_n(int): Number of virtual orbitals
        a_n(int): Number of active orbitals
        c_n(int): Number of core orbitals
    Author(s): Yuhto Mori
    """
    from .quket_data import int2occ
    nstates = Quket.multi.nstates
    n_orbitals = Quket.n_orbitals

    c_n = n_orbitals
    v_n = n_orbitals
    for istate in range(nstates):
        occ_list_tmp = int2occ(Quket.multi.states[istate])
        vir_list_tmp = [i for i in range(Quket.n_qubits) if i not in occ_list_tmp]
        core_tmp = vir_list_tmp[0] // 2
        vir_tmp = n_orbitals - (occ_list_tmp[-1] + 1) // 2
        c_n = min(c_n, core_tmp)
        v_n = min(v_n, vir_tmp)
    a_n = n_orbitals - v_n - c_n
    return v_n, a_n, c_n
            
def calc_num_ic_theta(Quket):
    """
    Calculation the number of theta_list for ic-MRUCC.

    Return:
        ndim1(int)
        ndim2(int)
    Author(s): Yuhto Mori
    """
    v, a, c = calc_num_v_a_c(Quket)
    a2a = Quket.multi.act2act_opt
    
    ### ndim1 ###
    ndim1 = c*v + c*a + a*v + a*(a-1)//2
    ndim1 = ndim1 * 2

    ### ndim2 ###
    if a2a:
        ndim2aa = ((c*(c-1)//2 + c*a + a*(a-1)//2) *
                   (v*(v-1)//2 + v*a + a*(a-1)//2) - 
                   a*(a-1)//2)
        ndim2ab = ((v*v + a*a + 2*v*a) *
                   (c*c + a*a + 2*a*c) -
                   a*a)
    else:
        ndim2aa = ((c*(c-1)//2 + c*a + a*(a-1)//2) *
                   (v*(v-1)//2 + v*a + a*(a-1)//2) - 
                   (a*(a-1)//2)**2)
        ndim2ab = ((v*v + a*a + 2*v*a) *
                   (c*c + a*a + 2*a*c) -
                   a**4)
    ndim2 = ndim2aa*2 + ndim2ab

    prints(f"vir: {v}, act: {a}, core: {c}")
    prints(f"act to act: {a2a}  [ndim1: {ndim1}, ndim2: {ndim2}]")
    return int(ndim1), int(ndim2)


def ucc_Gsingles_listversion(circuit, a_list, i_list, theta_list, theta_index):
    """ Function:
    generalized singles. i_list -> a_list

    Author(s): Yuhto Mori
    """
    if np.all(a_list == i_list):
        ai_list = itertools.combinations(a_list, 2)
    else:
        ai_list = itertools.product(a_list, i_list)

    for a, i in ai_list:
        single_ope_Pauli(a, i, circuit, theta_list[theta_index])
        theta_index += 1
        single_ope_Pauli(a+1, i+1, circuit, theta_list[theta_index])
        theta_index += 1
    return circuit, theta_index

def icmr_ucc_singles(circuit, v_n, a_n, c_n, theta_list, ndim2=0):
    """
    """
    theta_index = ndim2
    n_qubits = (v_n + a_n + c_n)*2

    c_list_a = np.arange(c_n*2)[::2]
    # c_list_b = np.arange(c_n*2)[1::2]
    a_list_a  = np.arange(c_n*2, (c_n+a_n)*2)[::2]
    # a_list_b  = np.arange(c_n*2, (c_n+a_n)*2)[1::2]
    v_list_a  = np.arange((a_n+c_n)*2, n_qubits)[::2]
    # v_list_b  = np.arange((a_n+c_n)*2, n_qubits)[1::2]

    circuit, theta_index = ucc_Gsingles_listversion(
                            circuit, v_list_a, c_list_a, theta_list, theta_index)
    circuit, theta_index = ucc_Gsingles_listversion(
                            circuit, v_list_a, a_list_a, theta_list, theta_index)
    circuit, theta_index = ucc_Gsingles_listversion(
                            circuit, a_list_a, c_list_a, theta_list, theta_index)
    circuit, theta_index = ucc_Gsingles_listversion(
                            circuit, a_list_a, a_list_a, theta_list, theta_index)
    return circuit

def ucc_Gdoubles_listversion(circuit, i1_list,i2_list,a1_list,a2_list,theta_list,theta_index):
    from .expope import Gdouble_ope
    if i1_list == i2_list:
        i1i2_aa = list(itertools.combinations(i1_list,2))
        i1i2_ab = list(itertools.product(i1_list,i2_list))
    else:
        i1i2_aa = list(itertools.product(i1_list,i2_list))
        i1i2_ab = list(itertools.product(i1_list,i2_list))
        i1i2_ab.extend(list(itertools.product(i2_list,i1_list)))
    if a1_list == a2_list:
        a1a2_aa = list(itertools.combinations(a1_list,2))
        a1a2_ab = list(itertools.product(a1_list,a2_list))
    else:
        a1a2_aa = list(itertools.product(a1_list,a2_list))
        a1a2_ab = list(itertools.product(a1_list,a2_list))
        a1a2_ab.extend(list(itertools.product(a2_list,a1_list)))
    
    ### aa-aa ###
    for [i1,i2] in i1i2_aa:
        for [a1,a2] in a1a2_aa:
            b = a2 * 2
            a = a1 * 2
            j = i2 * 2
            i = i1 * 2
            max_id = max(a,b,i,j)
            if a != i or b != j:
                theta = theta_list[theta_index]
                if b == max_id:
                    Gdouble_ope(b,a,j,i,circuit,theta)
                elif a == max_id:
                    Gdouble_ope(a,b,i,j,circuit,theta)
                elif i == max_id:
                    Gdouble_ope(i,j,a,b,circuit,-theta)
                elif j == max_id:
                    Gdouble_ope(j,i,b,a,circuit,-theta)
                theta_index += 1

    ### bb-bb ###
    for [i1,i2] in i1i2_aa:
        for [a1,a2] in a1a2_aa:
            b = a2 * 2 + 1
            a = a1 * 2 + 1
            j = i2 * 2 + 1
            i = i1 * 2 + 1
            max_id = max(a,b,i,j)
            if a != i or b != j:
                theta = theta_list[theta_index]
                if b == max_id:
                    Gdouble_ope(b,a,j,i,circuit,theta)
                elif a == max_id:
                    Gdouble_ope(a,b,i,j,circuit,theta)
                elif i == max_id:
                    Gdouble_ope(i,j,a,b,circuit,-theta)
                elif j == max_id:
                    Gdouble_ope(j,i,b,a,circuit,-theta)
                theta_index += 1

    ### ab-ab ###
    for [i1,i2] in i1i2_ab:
        for [a1,a2] in a1a2_ab:
            b = a2 * 2 + 1
            a = a1 * 2
            j = i2 * 2 + 1
            i = i1 * 2
            max_id = max(a,b,i,j)
            if a != i or b != j:
                theta = theta_list[theta_index]
                # if (b > a and i > j) or (a > b and j > i):
                #     theta *= -1
                if b == max_id:
                    Gdouble_ope(b,a,j,i,circuit,theta)
                elif a == max_id:
                    Gdouble_ope(a,b,i,j,circuit,theta)
                elif i == max_id:
                    Gdouble_ope(i,j,a,b,circuit,-theta)
                elif j == max_id:
                    Gdouble_ope(j,i,b,a,circuit,-theta)
                theta_index += 1
    return circuit, theta_index

def icmr_ucc_doubles(circuit, v_n, a_n, c_n, theta_list, ndim1=0, a2a=False):
    """
    """
    theta_index = ndim1
    n_orbitals = v_n + a_n + c_n

    v_list = list(range((c_n + a_n), n_orbitals))
    a_list = list(range(c_n, (c_n + a_n)))
    c_list = list(range(c_n))

    circuit, theta_index = ucc_Gdoubles_listversion(circuit,c_list,c_list,v_list,v_list,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,c_list,c_list,a_list,v_list,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,c_list,c_list,a_list,a_list,theta_list,theta_index)

    circuit, theta_index = ucc_Gdoubles_listversion(circuit,c_list,a_list,v_list,v_list,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,c_list,a_list,a_list,v_list,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,c_list,a_list,a_list,a_list,theta_list,theta_index)

    circuit, theta_index = ucc_Gdoubles_listversion(circuit,a_list,a_list,v_list,v_list,theta_list,theta_index)
    circuit, theta_index = ucc_Gdoubles_listversion(circuit,a_list,a_list,a_list,v_list,theta_list,theta_index)

    if a2a:
        circuit, theta_index = ucc_Gdoubles_listversion(circuit,a_list,a_list,a_list,a_list,theta_list,theta_index)

    return circuit

def set_circuit_ic_mrucc(n_qubits, v_n, a_n, c_n, DS, theta_list, ndim1, a2a):
    """
    """
    circuit = QuantumCircuit(n_qubits)

    if DS:
        circuit = icmr_ucc_singles(circuit,v_n, a_n, c_n, theta_list, 0)
        circuit = icmr_ucc_doubles(circuit,v_n, a_n, c_n, theta_list, ndim1, a2a)
    else:
        circuit = icmr_ucc_doubles(circuit,v_n, a_n, c_n, theta_list, ndim1, a2a)
        circuit = icmr_ucc_singles(circuit,v_n, a_n, c_n, theta_list, 0)
    return circuit

def create_icmr_uccsd_state(n_qubits, v_n, a_n, c_n, rho, DS, theta_list,
                            det, ndim1, act2act=False, SpinProj=False):
    """
    """
    state = QuantumState(n_qubits)
    state.set_computational_basis(det)

    theta_list_rho = theta_list/rho
    circuit = set_circuit_ic_mrucc(n_qubits, v_n, a_n, c_n, DS,
                                    theta_list_rho,ndim1, act2act)
    for i in range(rho):
        circuit.update_quantum_state(state)
    
    if SpinProj:
        from .phflib import S2Proj
        state_P = S2Proj(state)
        return state_P
    else:
        return state

def cost_ic_mrucc(Quket, print_level, theta_list):
    """ Function
    Author(s): Yuto Mori
    """
    t1 = time.time()

    nstates = Quket.multi.nstates
    n_qubits = Quket.n_qubits
    ndim = Quket.ndim
    ndim1 = Quket.ndim1
    rho = Quket.rho
    DS = Quket.DS
    v_n, a_n, c_n = calc_num_v_a_c(Quket)

    states = []
    for istate in range(nstates):
        det = Quket.multi.states[istate]
        state = create_icmr_uccsd_state(
                    n_qubits, v_n, a_n, c_n, rho, DS, theta_list, det, 
                    ndim1, act2act=Quket.multi.act2act_opt,
                    SpinProj=Quket.projection.SpinProj)
        states.append(state)
    H, S2, S = create_HS2S(Quket, states)

    root_invS = root_inv(S.real)
    H_ortho = root_invS.T@H@root_invS
    nstates0 = root_invS.shape[1]

    en, dvec = np.linalg.eig(H_ortho)
    idx   = np.argsort(en.real,-1)
    en    = en.real[idx]
    dvec  = dvec[:, idx]
    cvec  = root_invS@dvec
    S2dig = cvec.T@S2@cvec
    s2 = [S2dig[i, i].real for i in range(nstates0)]

    t2 = time.time()
    cpu1 = t2-t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:5d}: ", end="")
        for istate in range(nstates0):
            prints(f"E[{istate}] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:7.5f})  ", end="")
        prints(f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(ndim, theta_list, cf.tmp)
        # cf.iter_threshold = 0
    if print_level > 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        prints("Final: ", end="")
        for istate in range(nstates0):
            prints(f"E[{istate}] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:7.5f})  ", end="")
        prints(f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)\n")
        prints("------------------------------------")
        for istate in range(nstates):
            prints(f"ic Basis   {istate}")
            print_state(states[istate])
            prints("")
        printmat(cvec.real, name="Coefficients: ")
        prints("------------------------------------\n\n")
        prints("###############################################")
        prints("#                  ic states                  #")
        prints("###############################################", end="")

        for istate in range(nstates0):
            prints("")
            prints(f"State         : {istate}")
            prints(f"E             : {en[istate]:.8f}")
            prints(f"<S**2>        : {s2[istate]:.5f}")
            prints(f"Superposition : ")
            spstate = QuantumState(n_qubits)
            spstate.multiply_coef(0)
            for jstate in range(nstates0):
                state = states[jstate].copy()
                coef  = cvec[jstate, istate]
                state.multiply_coef(coef)
                spstate.add_state(state)
            print_state(spstate)
        prints("###############################################")
    cost = np.sum(Quket.multi.weights*en)
    norm = np.sum(Quket.multi.weights)
    cost /= norm
    return cost, s2



#### Spin-Free ####

def calc_num_ic_theta_spinfree(Quket):
    """
    Calculation the number of theta_list for ic-MRUCC(Spin-Free).

    Return:
        ndim1(int)
        ndim2(int)
    Author(s): Yuhto Mori
    """
    v, a, c = calc_num_v_a_c(Quket)
    a2a = Quket.multi.act2act_opt

    ### spin-free ###
    ndim1 = c*v + c*a + a*v + a*(a-1)//2
    ndim2 = ((a+v)*(a+c)+1) * ((a+v)*(a+c)) // 2
    if a2a:
        red = a*(a+1)//2
    else:
        red = a*a * (a*a + 1) // 2
    ndim2 = ndim2 - red

    prints(f"vir: {v}, act: {a}, core: {c}")
    prints(f"act to act: {a2a}  [ndim1: {ndim1}, ndim2: {ndim2}]")
    return int(ndim1), int(ndim2)

def get_baji_for_icmrucc_spinfree(b, a, j, i, v_n, a_n, c_n, a2a):
    """
    Author(s): Yuhto Mori
    """
    if a2a:
        bj = (b - c_n) * (a_n + c_n) + j
        ai = (a - c_n) * (a_n + c_n) + i
        if bj > ai:
            if b > j:
                if b >= a_n + c_n:
                    redu = (a_n+1)*a_n//2
                else:
                    redu = (b-c_n+1)*(b-c_n)//2
            elif b < j:
                redu = (b-c_n+2)*(b-c_n+1)//2
            else:
                redu = (b-c_n+1)*(b-c_n)//2 + a - c_n 
                if a < i:
                    redu = redu + 1
            baji = bj * (bj + 1) // 2 + ai - redu
        else:
            if a > i:
                if a >= a_n + c_n:
                    redu = (a_n+1)*a_n//2
                else:
                    redu = (a-c_n+1)*(a-c_n)//2
            elif a < i:
                redu = (a-c_n+2)*(a-c_n+1)//2
            else:
                redu = (a-c_n+1)*(a-c_n)//2 + b - c_n
                if b < j:
                    redu = redu + 1
            baji = ai * (ai + 1) // 2 + bj - redu
        index = int(baji)
    else:
        bj = (b - c_n) * (a_n + c_n) + j
        ai = (a - c_n) * (a_n + c_n) + i
        if bj > ai:
            if b >= a_n + c_n:
                redu = a_n*a_n * (a_n*a_n + 1) // 2
            else:
                tmp = a_n * (b - c_n)
                if j >= c_n:
                    tmp = tmp + j - c_n
                redu = tmp * (tmp + 1) // 2 + a_n * (a - c_n)
            baji = bj * (bj + 1) // 2 + ai - redu
        else:
            if a >= a_n + c_n:
                redu = a_n*a_n * (a_n*a_n + 1) // 2
            else:
                tmp = a_n * (a - c_n)
                if i >= c_n:
                    tmp = tmp + i - c_n
                redu = tmp * (tmp + 1) // 2 + a_n * (b - c_n)
            baji = ai * (ai + 1) // 2 + bj - redu
        index = int(baji)
    return index

def ucc_Gsingles_listversion_spinfree(circuit, a_list, i_list, theta_list, theta_index):
    """ Function:
    generalized singles. i_list -> a_list

    Author(s): Yuhto Mori
    """
    if np.all(a_list == i_list):
        ai_list = itertools.combinations(a_list, 2)
    else:
        ai_list = itertools.product(a_list, i_list)

    for a, i in ai_list:
        single_ope_Pauli(a, i, circuit, theta_list[theta_index])
        single_ope_Pauli(a+1, i+1, circuit, theta_list[theta_index])
        theta_index += 1
    return circuit, theta_index

def icmr_ucc_singles_spinfree(circuit, v_n, a_n, c_n, theta_list, ndim2=0):
    """
    Author(s): Yuhto Mori
    """
    theta_index = ndim2
    n_qubits = (v_n + a_n + c_n)*2

    c_list_a = np.arange(c_n*2)[::2]
    # c_list_b = np.arange(c_n*2)[1::2]
    a_list_a  = np.arange(c_n*2, (c_n+a_n)*2)[::2]
    # a_list_b  = np.arange(c_n*2, (c_n+a_n)*2)[1::2]
    v_list_a  = np.arange((a_n+c_n)*2, n_qubits)[::2]
    # v_list_b  = np.arange((a_n+c_n)*2, n_qubits)[1::2]

    circuit, theta_index = ucc_Gsingles_listversion_spinfree(
                            circuit, v_list_a, c_list_a, theta_list, theta_index)
    circuit, theta_index = ucc_Gsingles_listversion_spinfree(
                            circuit, v_list_a, a_list_a, theta_list, theta_index)
    circuit, theta_index = ucc_Gsingles_listversion_spinfree(
                            circuit, a_list_a, c_list_a, theta_list, theta_index)
    circuit, theta_index = ucc_Gsingles_listversion_spinfree(
                            circuit, a_list_a, a_list_a, theta_list, theta_index)
    return circuit

def ucc_Gdoubles_listversion_spinfree(
    circuit, i1_list, i2_list, a1_list, a2_list, theta_list, ndim1, v_n, a_n, c_n, a2a):
    from .expope import Gdouble_ope
    if i1_list == i2_list:
        i1i2_aa = list(itertools.combinations(i1_list,2))
        i1i2_ab = list(itertools.product(i1_list,i2_list))
    else:
        i1i2_aa = list(itertools.product(i1_list,i2_list))
        i1i2_ab = list(itertools.product(i1_list,i2_list))
        i1i2_ab.extend(list(itertools.product(i2_list,i1_list)))
    if a1_list == a2_list:
        a1a2_aa = list(itertools.combinations(a1_list,2))
        a1a2_ab = list(itertools.product(a1_list,a2_list))
    else:
        a1a2_aa = list(itertools.product(a1_list,a2_list))
        a1a2_ab = list(itertools.product(a1_list,a2_list))
        a1a2_ab.extend(list(itertools.product(a2_list,a1_list)))
    
    ### aa-aa ###
    for [i1,i2] in i1i2_aa:
        for [a1,a2] in a1a2_aa:
            b = a2 * 2
            a = a1 * 2
            j = i2 * 2
            i = i1 * 2
            max_id = max(a,b,i,j)
            if a != i or b != j:
                baji = get_baji_for_icmrucc_spinfree(a2,a1,i2,i1,v_n,a_n,c_n,a2a) + ndim1
                baij = get_baji_for_icmrucc_spinfree(a2,a1,i1,i2,v_n,a_n,c_n,a2a) + ndim1
                theta = theta_list[baji] - theta_list[baij]
                if b == max_id:
                    Gdouble_ope(b,a,j,i,circuit,theta)
                elif a == max_id:
                    Gdouble_ope(a,b,i,j,circuit,theta)
                elif i == max_id:
                    Gdouble_ope(i,j,a,b,circuit,-theta)
                elif j == max_id:
                    Gdouble_ope(j,i,b,a,circuit,-theta)

    ### bb-bb ###
    for [i1,i2] in i1i2_aa:
        for [a1,a2] in a1a2_aa:
            b = a2 * 2 + 1
            a = a1 * 2 + 1
            j = i2 * 2 + 1
            i = i1 * 2 + 1
            max_id = max(a,b,i,j)
            if a != i or b != j:
                baji = get_baji_for_icmrucc_spinfree(a2,a1,i2,i1,v_n,a_n,c_n,a2a) + ndim1
                baij = get_baji_for_icmrucc_spinfree(a2,a1,i1,i2,v_n,a_n,c_n,a2a) + ndim1
                theta = theta_list[baji] - theta_list[baij]
                if b == max_id:
                    Gdouble_ope(b,a,j,i,circuit,theta)
                elif a == max_id:
                    Gdouble_ope(a,b,i,j,circuit,theta)
                elif i == max_id:
                    Gdouble_ope(i,j,a,b,circuit,-theta)
                elif j == max_id:
                    Gdouble_ope(j,i,b,a,circuit,-theta)

    ### ab-ab ###
    for [i1,i2] in i1i2_ab:
        for [a1,a2] in a1a2_ab:
            b = a2 * 2 + 1
            a = a1 * 2
            j = i2 * 2 + 1
            i = i1 * 2
            max_id = max(a,b,i,j)
            if a != i or b != j:
                baji = get_baji_for_icmrucc_spinfree(a2,a1,i2,i1,v_n,a_n,c_n,a2a) + ndim1
                theta = theta_list[baji]
                if b == max_id:
                    Gdouble_ope(b,a,j,i,circuit,theta)
                elif a == max_id:
                    Gdouble_ope(a,b,i,j,circuit,theta)
                elif i == max_id:
                    Gdouble_ope(i,j,a,b,circuit,-theta)
                elif j == max_id:
                    Gdouble_ope(j,i,b,a,circuit,-theta)
    return circuit

def icmr_ucc_doubles_spinfree(circuit, v_n, a_n, c_n, theta_list, ndim1=0, a2a=False):
    """
    """
    n_orbitals = v_n + a_n + c_n

    v_list = list(range((c_n + a_n), n_orbitals))
    a_list = list(range(c_n, (c_n + a_n)))
    c_list = list(range(c_n))

    circuit = ucc_Gdoubles_listversion_spinfree(
                circuit, c_list, c_list, v_list, v_list, theta_list, ndim1, v_n, a_n, c_n, a2a)
    circuit = ucc_Gdoubles_listversion_spinfree(
                circuit, c_list, c_list, a_list, v_list, theta_list, ndim1, v_n, a_n, c_n, a2a)
    circuit = ucc_Gdoubles_listversion_spinfree(
                circuit, c_list, c_list, a_list, a_list, theta_list, ndim1, v_n, a_n, c_n, a2a)

    circuit = ucc_Gdoubles_listversion_spinfree(
                circuit, c_list, a_list, v_list, v_list, theta_list, ndim1, v_n, a_n, c_n, a2a)
    circuit = ucc_Gdoubles_listversion_spinfree(
                circuit, c_list, a_list, a_list, v_list, theta_list, ndim1, v_n, a_n, c_n, a2a)
    circuit = ucc_Gdoubles_listversion_spinfree(
                circuit, c_list, a_list, a_list, a_list, theta_list, ndim1, v_n, a_n, c_n, a2a)

    circuit = ucc_Gdoubles_listversion_spinfree(
                circuit, a_list, a_list, v_list, v_list, theta_list, ndim1, v_n, a_n, c_n, a2a)
    circuit = ucc_Gdoubles_listversion_spinfree(
                circuit, a_list, a_list, a_list, v_list, theta_list, ndim1, v_n, a_n, c_n, a2a)
 
    if a2a:
        circuit = ucc_Gdoubles_listversion_spinfree(
                    circuit, a_list, a_list, a_list, a_list, theta_list, ndim1, v_n, a_n, c_n, a2a)

    return circuit

def set_circuit_ic_mrucc_spinfree(n_qubits, v_n, a_n, c_n, DS, theta_list, ndim1, a2a):
    """
    """
    circuit = QuantumCircuit(n_qubits)

    if DS:
        circuit = icmr_ucc_singles_spinfree(circuit,v_n, a_n, c_n, theta_list, 0)
        circuit = icmr_ucc_doubles_spinfree(circuit,v_n, a_n, c_n, theta_list, ndim1, a2a)
    else:
        circuit = icmr_ucc_doubles_spinfree(circuit,v_n, a_n, c_n, theta_list, ndim1, a2a)
        circuit = icmr_ucc_singles_spinfree(circuit,v_n, a_n, c_n, theta_list, 0)
    return circuit

def create_icmr_uccsd_state_spinfree(n_qubits, v_n, a_n, c_n, rho, DS, theta_list,
                            det, ndim1, act2act=False, SpinProj=False):
    """
    """
    state = QuantumState(n_qubits)
    state.set_computational_basis(det)

    theta_list_rho = theta_list/rho
    circuit = set_circuit_ic_mrucc_spinfree(n_qubits, v_n, a_n, c_n, DS,
                                    theta_list_rho,ndim1, act2act)
    for i in range(rho):
        circuit.update_quantum_state(state)
    
    if SpinProj:
        from .phflib import S2Proj
        state_P = S2Proj(state)
        return state_P
    else:
        return state

def cost_ic_mrucc_spinfree(Quket, print_level, theta_list):
    """ Function

    Author(s): Yuto Mori
    """
    t1 = time.time()

    nstates = Quket.multi.nstates
    n_qubits = Quket.n_qubits
    ndim = Quket.ndim
    ndim1 = Quket.ndim1
    rho = Quket.rho
    DS = Quket.DS
    v_n, a_n, c_n = calc_num_v_a_c(Quket)

    states = []
    for istate in range(nstates):
        det = Quket.multi.states[istate]
        state = create_icmr_uccsd_state_spinfree(
                    n_qubits, v_n, a_n, c_n, rho, DS, theta_list, det, 
                    ndim1, act2act=Quket.multi.act2act_opt,
                    SpinProj=Quket.projection.SpinProj)
        states.append(state)
    H, S2, S = create_HS2S(Quket, states)

    root_invS = root_inv(S.real)
    H_ortho = root_invS.T@H@root_invS
    nstates0 = root_invS.shape[1]

    en, dvec = np.linalg.eig(H_ortho)
    idx   = np.argsort(en.real,-1)
    en    = en.real[idx]
    dvec  = dvec[:, idx]
    cvec  = root_invS@dvec
    S2dig = cvec.T@S2@cvec
    s2 = [S2dig[i, i].real for i in range(nstates0)]

    t2 = time.time()
    cpu1 = t2-t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:5d}: ", end="")
        for istate in range(nstates0):
            prints(f"E[{istate}] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:7.5f})  ", end="")
        prints(f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(ndim, theta_list, cf.tmp)
        # cf.iter_threshold = 0
    if print_level > 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        prints("Final: ", end="")
        for istate in range(nstates0):
            prints(f"E[{istate}] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:7.5f})  ", end="")
        prints(f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)\n")
        prints("------------------------------------")
        for istate in range(nstates):
            prints(f"ic Basis   {istate}")
            print_state(states[istate])
            prints("")
        printmat(cvec.real, name="Coefficients: ")
        prints("------------------------------------\n\n")
        prints("###############################################")
        prints("#                  ic states                  #")
        prints("###############################################", end="")

        for istate in range(nstates0):
            prints("")
            prints(f"State         : {istate}")
            prints(f"E             : {en[istate]:.8f}")
            prints(f"<S**2>        : {s2[istate]:.5f}")
            prints(f"Superposition : ")
            spstate = QuantumState(n_qubits)
            spstate.multiply_coef(0)
            for jstate in range(nstates0):
                state = states[jstate].copy()
                coef  = cvec[jstate, istate]
                state.multiply_coef(coef)
                spstate.add_state(state)
            print_state(spstate)
        prints("###############################################")
    cost = np.sum(Quket.multi.weights*en)
    norm = np.sum(Quket.multi.weights)
    cost /= norm
    return cost, s2