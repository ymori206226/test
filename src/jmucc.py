"""
#######################
#        quket        #
#######################

jmucc.py

Multi-reference UCC.
Jeziorski-Monkhorst UCC.

"""
import time
import itertools

import numpy as np
from qulacs import QuantumState, QuantumCircuit
from qulacs.state import inner_product

from . import config as cf
from .fileio import SaveTheta, print_state, prints, printmat
from .utils import root_inv
from .ucclib import create_uccsd_state
from .quket_data import int2occ


def create_kappalist(ndim1, occ_list, noa, nob, nva, nvb):
    """Function
    Create kappalist from occ_list, which stores the occupied qubits.

    Author(s):  Yuto Mori
    """
    kappa = np.zeros(ndim1)
    occ_hf = set(range(len(occ_list)))
    dup = occ_hf & set(occ_list)
    cre_set = list(set(occ_list) - dup)
    ann_set = list(occ_hf - dup)
    kappalist = []
    for c in cre_set:
        if c%2 == 0:
            for a in ann_set:
                if a%2 == 0:
                    ann_set.remove(a)
                    kappalist.append(int(a/2 + noa*(c/2 - noa)))
        else:
            for a in ann_set:
                if a% 2 == 1:
                    ann_set.remove(a)
                    kappalist.append(
                            int((a-1)/2 + nob*((c-1)/2 - nob) + noa*nva))
    for i in kappalist:
        kappa[i] = np.pi/2
    return kappa


def create_HS2S(QuketData, states):
    X_num = len(states)
    H = np.zeros((X_num, X_num), dtype=np.complex)
    S2 = np.zeros((X_num, X_num), dtype=np.complex)
    S = np.zeros((X_num, X_num), dtype=np.complex)
    for i in range(X_num):
        for j in range(i+1):
            H[i, j] = QuketData.qulacs.Hamiltonian.get_transition_amplitude(
                    states[i], states[j])
            S2[i, j] = QuketData.qulacs.S2.get_transition_amplitude(
                    states[i], states[j])
            S[i, j] = inner_product(states[i], states[j])
        H[:i, i] = H[i, :i]
        S2[:i, i] = S2[i, :i]
        S[:i, i] = S[i, :i]
    return H, S2, S


## Spin Adapted ##
def create_sa_state(n_qubits, n_electrons, noa, nob, nva, nvb, rho, DS,
                    kappa_list, theta_list, occ_list, vir_list,
                    threshold=1e-4):
    """Function
    Create a spin-free Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    from .hflib import set_circuit_rhf, set_circuit_rohf, set_circuit_uhf

    state = QuantumState(n_qubits)
    if noa == nob:
        circuit_rhf = set_circuit_rhf(n_qubits, n_electrons)
    else:
        circuit_rhf = set_circuit_rohf(n_qubits, noa, nob)
    circuit_rhf.update_quantum_state(state)
    theta_list_rho = theta_list/rho
    circuit = set_circuit_sauccsdX(n_qubits, noa, nob, nva, nvb, DS,
                                   theta_list_rho, occ_list, vir_list)
    if np.linalg.norm(kappa_list) > threshold:
        circuit_uhf = set_circuit_uhf(n_qubits, noa, nob, nva, nvb, kappa_list)
        circuit_uhf.update_quantum_state(state)
    for i in range(rho):
        circuit.update_quantum_state(state)
    return state


def set_circuit_sauccsdX(n_qubits, noa, nob, nva, nvb, DS, theta_list,
                         occ_list, vir_list):
    """Function
    Prepare a Quantum Circuit for a spin-free Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    ndim1 = noa*nva
    circuit = QuantumCircuit(n_qubits)
    if DS:
        ucc_sa_singlesX(circuit, theta_list, occ_list, vir_list, 0)
        ucc_sa_doublesX(circuit, theta_list, occ_list, vir_list, ndim1)
    else:
        ucc_sa_doublesX(circuit, theta_list, occ_list, vir_list, ndim1)
        ucc_sa_singlesX(circuit, theta_list, occ_list, vir_list, 0)
    return circuit


def ucc_sa_singlesX(circuit, theta_list, occ_list, vir_list, ndim2=0):
    """Function
    Prepare a Quantum Circuit for the single exictation part of a spin-free
    Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    from .ucclib import single_ope_Pauli
    global ncnot

# 3重項の時だめじゃね？
    ia = ndim2
    occ_list_a = [i for i in occ_list if i%2 == 0]
    #occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    #vir_list_b = [i for i in vir_list if i%2 == 1]

    ### alpha (& beta) ###
    ncnot = 0
    for a in vir_list_a:
        for i in occ_list_a:
            single_ope_Pauli(a, i, circuit, theta_list[ia])
            single_ope_Pauli(a+1, i+1, circuit, theta_list[ia])
            ia += 1


def ucc_sa_doublesX(circuit, theta_list, occ_list, vir_list, ndim1=0):
    """Function
    Prepare a Quantum Circuit for the double exictation part of a spin-free
    Jeziorski-Monkhorst UCC state based on theta_list.

    Author(s):  Yuto Mori
    """
    from .ucclib import double_ope_Pauli
    global ncnot

    ijab = ndim1
    occ_list_a = [i for i in occ_list if i%2 == 0]
    occ_list_b = [i for i in occ_list if i%2 == 1]
    vir_list_a = [i for i in vir_list if i%2 == 0]
    vir_list_b = [i for i in vir_list if i%2 == 1]

    ### aa or bb ##
    ncnot = 0
    for a, b in itertools.combinations(vir_list_a, 2):
        for i, j in itertools.combinations(occ_list_a, 2):
            double_ope_Pauli(b, a, j, i, circuit, theta_list[ijab])
            double_ope_Pauli(b+1, a+1, j+1, i+1, circuit, theta_list[ijab])
            ijab += 1
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
                    baji = get_baji(b_i, a_i, j_i, i_i, no, nv)
                    double_ope_Pauli(max(b, a), min(b, a),
                                     max(j, i), min(j, i),
                                     circuit, theta_list[ijab+baji])


def get_baji(b, a, j, i, no, nv):
    """Function
    Get index for b,a,j,i

    Author(s):  Yuto Mori
    """
    nov = no*nv
    aa = i*nv + a
    bb = j*nv + b
    baji = int(nov*(nov-1)/2 - (nov-1-aa)*(nov-aa)/2 + bb)
    return baji


def cost_jmucc(Quket, print_level, theta_lists):
    """Function
    Cost function of Jeziorski-Monkhorst UCCSD.

    Author(s):  Yuto Mori, Takashi Tsuchimochi
    """
    t1 = time.time()

    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    nocc = noa + nob
    n_electrons = Quket.n_active_electrons
    n_qubits = Quket.n_qubits
    nstates = len(Quket.multi.weights)
    ndim1 = Quket.ndim1
    ndim2 = Quket.ndim2
    ndim = Quket.ndim
    ndim_i = ndim1 + ndim2
    rho = Quket.rho
    DS = Quket.DS
    det = Quket.det

    occ_lists = []
    vir_lists = []
    for istate in range(nstates):
        ### Read state integer and extract occupied/virtual info
        occ_list_tmp = int2occ(Quket.multi.states[istate])
        vir_list_tmp = [i for i in range(n_qubits) if i not in occ_list_tmp]
        occ_lists.extend(occ_list_tmp)
        vir_lists.extend(vir_list_tmp)

    ### Prepare kappa_lists
    kappa_lists = []
    for istate in range(nstates):
        kappa_list = create_kappalist(
                ndim1, occ_lists[nocc*istate : nocc*(istate+1)],
                noa, nob, nva, nvb)
        kappa_lists.extend(kappa_list)

    ### Prepare JM basis
    states = []
    for istate in range(nstates):
        det = Quket.multi.states[istate]
        state = create_uccsd_state(
                n_qubits, rho, DS,
                theta_lists[ndim_i*istate : ndim_i*(istate+1)],
                det, ndim1)
        if Quket.projection.SpinProj:
            from .phflib import S2Proj
            state = S2Proj(Quket, state)
        #prints('\n State {}?'.format(istate))
        #print_state(state)
        states.append(state)
    H, S2, S = create_HS2S(Quket, states)

    #invS   = np.linalg.inv(S)
    #H_invS = np.dot(invS, H)
    #np.set_printoptions(precision=17)
    #en,cvec = np.linalg.eig(H_invS)
    #printmat(en, name="energy")
    #printmat(cvec, name="cvec")

    root_invS = root_inv(S.real)
    H_ortho = root_invS.T@H@root_invS
    nstates0 = root_invS.shape[1]
    #print(nstates0)

    en, dvec = np.linalg.eig(H_ortho)

    ind = np.argsort(en.real, -1)
    en = en.real[ind]
    dvec = dvec[:, ind]
    #printmat(en, name="energy")
    #printmat(dvec, name="dvec")
    cvec = root_invS@dvec

    # Renormalize
    #    Sdig     =cvec.T@S@cvec
    #printmat(Sdig,name="Sdig")
    #    for istate in range(nstates):
    #        cvec[:,istate] = cvec[:,istate] / np.sqrt(Sdig[istate,istate].real)
    # Compute <S**2> of each state
    S2dig = cvec.T@S2@cvec
    #printmat(S2dig)

    s2 = []
    for istate in range(nstates0):
        s2.append(S2dig[istate, istate].real)

    # compute <S**2> for each state
    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(f"{cf.icyc:5d}:", end="")
        for istate in range(nstates0):
            prints(f"E[{istate}] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:7.5f})  ",
                   end="")
        prints(f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        SaveTheta(ndim, theta_lists, cf.tmp)
        # cf.iter_threshold = 0
    if print_level > 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        prints("Final:", end="")
        for istate in range(nstates0):
            prints(f"E[{istate}] = {en[istate]:.8f}  "
                   f"(<S**2> = {s2[istate]:7.5f})  ",
                   end="")
        prints(f"CPU Time = {cput:5.2f}  ({cpu1:2.2f} / step)")
        prints("\n------------------------------------")
        for istate in range(nstates):
            prints(f"JM Basis   {istate}")
            print_state(states[istate])
            prints("")
        printmat(cvec.real, name="Coefficients: ")
        prints("------------------------------------\n\n")
        prints("###############################################")
        prints("#                  JM states                  #")
        prints("###############################################", end="")

        for istate in range(nstates0):
            prints("")
            prints(f"State        : {istate}")
            prints(f"E            : {en[istate]:.8f}")
            prints(f"<S**2>       : {s2[istate]:.5f}")
            prints(f"Superposition:")
            spstate = QuantumState(n_qubits)
            spstate.multiply_coef(0)
            for jstate in range(nstates0):
                state = states[jstate].copy()
                coef = cvec[jstate, istate]
                state.multiply_coef(coef)
                spstate.add_state(state)
            print_state(spstate)
        prints("###############################################")

    cost = np.sum(Quket.multi.weights*en)
    norm = np.sum(Quket.multi.weights)
    cost /= norm
    return cost, s2
