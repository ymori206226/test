import time
import copy
import itertools

import numpy as np
from scipy.special import comb
from qulacs import QuantumState, QuantumCircuit

from . import config as cf
from .utils import root_inv
from .expope import Gdouble_ope
from .ucclib import single_ope_Pauli
from .fileio import prints, print_state, SaveTheta, printmat


def ucc_Gsingles_listversion(circuit, a_list, i_list, theta_list, theta_index):
    """ Function
    generalized singles. i_list -> a_list
    Author(s): Yuto Mori
    """
    if a_list == i_list:
        ai_list = itertools.combinations(a_list, 2)
    else:
        ai_list = itertools.product(a_list, i_list)

    for a, i in ai_list:
        single_ope_Pauli(a, i, circuit, theta_list[theta_index])
        theta_index += 1
        single_ope_Pauli(a+1, i+1, circuit, theta_list[theta_index])
        theta_index += 1
    return circuit, theta_index


def ucc_Gdoubles_listversion(circuit, b_list, a_list, j_list, i_list,
                             theta_list, theta_index):
    """ Function
    Author(s): Yuto Mori
    """
    if b_list == a_list:
        ab = itertools.combinations(b_list, 2)
    else:
        ab = itertools.product(a_list, b_list)
    if j_list == i_list:
        ij = itertools.combinations(j_list, 2)
    else:
        ij = itertools.product(i_list, j_list)

    abij = itertools.product(ab, ij)
    for [a, b], [i, j] in abij:
        max_id = max(a, b, i, j)
        if a != i or b != j:
            if b == max_id:
                Gdouble_ope(b, a, j, i, circuit, theta_list[theta_index])
                theta_index += 1
            elif a == max_id:
                Gdouble_ope(a, b, i, j, circuit, theta_list[theta_index])
                theta_index += 1
            elif i == max_id:
                Gdouble_ope(i, j, a, b, circuit, theta_list[theta_index])
                theta_index += 1
            elif j == max_id:
                Gdouble_ope(j, i, b, a, circuit, theta_list[theta_index])
                theta_index += 1
    return circuit, theta_index


def icmr_ucc_singles(circuit, n_qubit_system, nv, na, nc, theta_list,
                     ndim2=0):
    """ Function
    Author(s): Yuto Mori
    """
    theta_index = ndim2
    core_list_a = np.arange(nc)[::2]
    core_list_b = np.arange(nc)[1::2]
    act_list_a  = np.arange(nc, nc+na)[::2]
    act_list_b  = np.arange(nc, nc+na)[1::2]
    vir_list_a  = np.arange(nc+na, n_qubit_system)[::2]
    vir_list_b  = np.arange(nc_na, n_qubit_system)[1::2]

# alphaしか使ってない...?
    circuit, theta_index \
            = ucc_Gsingles_listversion(circuit, act_list[::2], core_list[::2],
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gsingles_listversion(circuit, vir_list[::2], core_list[::2],
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gsingles_listversion(circuit, act_list[::2], act_list[::2],
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gsingles_listversion(circuit, vir_list[::2], act_list[::2],
                                       theta_list, theta_index)


def icmr_ucc_doubles(circuit, n_qubit_system, nv, na, nc, theta_list,
                     ndim1=0):
    """ Function
    Author(s): Yuto Mori
    """
    theta_index = ndim1
    core_list_a = np.arange(nc)[::2]
    core_list_b = np.arange(nc)[1::2]
    act_list_a  = np.arange(nc, nc+na)[::2]
    act_list_b  = np.arange(nc, nc+na)[1::2]
    vir_list_a  = np.arange(nc+na, n_qubit_system)[::2]
    vir_list_b  = np.arange(nc_na, n_qubit_system)[1::2]

    ### aaaa ###
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_a, act_list_a,
                                       core_list_a, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_a, act_list_a,
                                       core_list_a, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_a, vir_list_a,
                                       core_list_a, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_a, act_list_a,
                                       act_list_a, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_a, act_list_a,
                                       act_list_a, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_a, vir_list_a,
                                       act_list_a, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_a, act_list_a,
                                       act_list_a, act_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_a, vir_list_a,
                                       act_list_a, act_list_a,
                                       theta_list, theta_index)

    ### bbbb ###
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, act_list_b,
                                       core_list_b, core_list_b,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, act_list_b,
                                       core_list_b, core_list_b,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, vir_list_b,
                                       core_list_b, core_list_b,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, act_list_b,
                                       act_list_b, core_list_b,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, act_list_b,
                                       act_list_b, core_list_b,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, vir_list_b,
                                       act_list_b, core_list_b,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, act_list_b,
                                       act_list_b, act_list_b,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, vir_list_b,
                                       act_list_b, act_list_b,
                                       theta_list, theta_index)
    ### aabb ###
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, vir_list_a,
                                       core_list_b, act_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, vir_list_a,
                                       core_list_b, act_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, vir_list_a,
                                       act_list_b, act_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, vir_list_a,
                                       act_list_b, act_list_a,
                                       theta_list, theta_index)

    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, act_list_a,
                                       core_list_b, act_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, act_list_a,
                                       core_list_b, act_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, act_list_a,
                                       act_list_b, act_list_a,
                                       theta_list, theta_index)

    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, vir_list_a,
                                       core_list_b, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, vir_list_a,
                                       core_list_b, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, vir_list_a,
                                       act_list_b, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, vir_list_a,
                                       act_list_b, core_list_a,
                                       theta_list, theta_index)

    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, act_list_a,
                                       core_list_b, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, act_list_a,
                                       core_list_b, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       act_list_b, act_list_a,
                                       act_list_b, core_list_a,
                                       theta_list, theta_index)
    circuit, theta_index \
            = ucc_Gdoubles_listversion(circuit,
                                       vir_list_b, act_list_a,
                                       act_list_b, core_list_a,
                                       theta_list, theta_index)

    if cf.act2act_ops:
        circuit, theta_index \
                = ucc_Gdoubles_listversion(circuit,
                                           act_list_a, act_list_a,
                                           act_list_a, act_list_a,
                                           theta_list, theta_index)
        circuit, theta_index \
                = ucc_Gdoubles_listversion(circuit,
                                           act_list_b, act_list_a,
                                           act_list_b, act_list_a,
                                           theta_list, theta_index)
        circuit, theta_index \
                = ucc_Gdoubles_listversion(circuit,
                                           act_list_b, act_list_b,
                                           act_list_b, act_list_b,
                                           theta_list, theta_index)


def set_circuit_ic_mrucc(n_qubit_system, nv, na, nc, DS, theta_list, ndim1):
    """ Function
    Author(s): Yuto Mori
    """
    circuit = QuantumCircuit(n_qubit_system)

    if DS:
        icmr_ucc_singles(circuit, n_qubit_system, nv, na, nc, theta_list, 0)
        icmr_ucc_doubles(circuit, n_qubit_system, nv, na, nc, theta_list, ndim1)
    else:
        icmr_ucc_doubles(circuit, n_qubit_system, nv, na, nc, theta_list, ndim1)
        icmr_ucc_singles(circuit, n_qubit_system, nv, na, nc, theta_list, 0)
    return circuit


def create_icmr_uccsd_state(n_qubit_system, nv, na, nc, rho, DS, theta_list,
                            det, ndim1):
    """ Function
    Author(s): Yuto Mori
    """
    state = QuantumState(n_qubit_system)
    state.set_computational_basis(det)

    theta_list_rho = theta_list/rho
    circuit = set_circuit_ic_mrucc(n_qubit_system, nv, na, nc, DS,
                                   theta_list_rho, ndim1)
    for i in range(rho):
        circuit.update_quantum_state(state)

    if cf.SpinProj:
        from .phflib import S2Proj
        state_P = S2Proj(state)
        return state_P
    else:
        return state


#def calncum_ic_theta(n_qubit_system, nv, na, nc):
#    """ Function
#    Calc number of ndim1, ndim2 for icmr_uccsd
#    Author(s): Yuto Mori
#    """
#    v = nv/2
#    a = na/2
#    c = nc/2
#    ### ndim1 ###
#    ndim1 = c*a + c*v + comb(a, 2) + a*v
#    ndim1 = ndim1*2
#
#    ### ndim2 ###
#    if cf.act2act_ops:
#        ndim2ab = (a*a + a*v*2 + v*v)*(a*a + c*a*2 + c*c) - a*a
#        ndim2aa = ((comb(c, 2) + c*a + comb(a, 2))
#                  *(comb(a, 2) + comb(v, 2) + a*v)
#                  - comb(a, 2))
#    else:
#        ndim2ab = (a*a + a*v*2 + v*v)*(a*a + c*a*2 + c*c) - a**4
#        ndim2aa = ((comb(c, 2) + c*a + comb(a, 2))
#                  *(comb(a, 2) + comb(v, 2) + a*v)
#                  - comb(a, 2)**2)
#    ndim2 = ndim2ab + ndim2aa*2
#    return int(ndim1), int(ndim2)


#def cost_ic_mrucc(print_level, n_qubit_system, n_electrons, nv, na, nc, rho, DS,
#                  qulacs_hamiltonian, qulacs_s2, theta_list, threshold):
def cost_ic_mrucc(Quket, print_level, qulacs_hamiltonian, qulacs_s2,
                  theta_list):
    """ Function
    Author(s): Yuto Mori
    """
    from .init import int2occ
    from .jmucc import create_HS2S

    t1 = time.time()
    #nstates = len(cf.multi_weights)

    # nc = n_qubit_system
    # vir_index = 0
    # for istate in range(nstates):
    #     ### Read state integer and extract occupied/virtual info
    #     occ_list_tmp = int2occ(cf.multi_states[istate])
    #     vir_tmp = occ_list_tmp[-1] + 1
    #     for ii in range(len(occ_list_tmp)):
    #         if ii == occ_list_tmp[ii]: core_tmp = ii + 1
    #     vir_index = max(vir_index,vir_tmp)
    #     nc = min(nc,core_tmp)
    # nv = n_qubit_system - vir_index
    # na = n_qubit_system - nc - nv

    #ndim1, ndim2 = calncum_ic_theta(n_qubit_system,nv,na,nc)

    nca = ncb = 0
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    ndim1 = Quket.ndim1
    ndim2 = Quket.ndim2
    ndim = Quket.ndim
    nstates = Quket.Multi.nstates

    # assume that nca = ncb, noa = nob and nva = nvb
    nc = nca
    no = noa
    nv = nva

    states = []
    for istate in range(nstates):
        det = cf.multi_states[istate]
        state = create_icmr_uccsd_state(n_qubit_system, nv, na, nc, rho, DS,
                                        theta_list, det, ndim1)
        states.append(state)
    H, S2, S = create_HS2S(qulacs_hamiltonian, qulacs_s2, states)

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
            spstate = QuantumState(n_qubit_system)
            spstate.multiply_coef(0)
            for jstate in range(nstates0):
                state = states[jstate].copy()
                coef  = cvec[jstate, istate]
                state.multiply_coef(coef)
                spstate.add_state(state)
            print_state(spstate)
        prints("###############################################")

    cost = norm = 0
    norm = np.sum(Quket.multi.weights)
    cost = np.sum(Quket.multi.weights*en)
    cost /= norm
    return cost, s2
