"""
#######################
#        quket        #
#######################

phflib.py

Functions related to spin-projection.

"""

import numpy as np
import time
from qulacs import QuantumCircuit, QuantumState
from . import config as cf
from .ucclib import (
    ucc_singles,
    ucc_singles_g,
    set_circuit_uccsd,
    set_circuit_uccd,
    set_circuit_sauccd,
    single_ope_Pauli,
)
from .fileio import (
    SaveTheta,
    LoadTheta,
    print_state,
    print_amplitudes,
    prints,
    error,
)

def trapezoidal(x0, x1, n):
    """Function
    Return points and weights based on trapezoidal rule
    """
    if n == 1:
        h = 0
    else:
        h = (x1 - x0) / (n - 1)
    w = []
    x = []
    w.append(h / 2)
    x.append(x0)
    for i in range(1, n - 1):
        w.append(h)
        x.append(x0 + i * h)
    w.append(h / 2)
    x.append(x1)
    return x, w


def simpson(x0, x1, n):
    """Function
    Return points and weights based on simpson's rule
    """
    if n % 2 == 0:
        error("Simpson's rule cannot be applied with even grids.")

    if n == 1:
        h = 0
    else:
        h = (x1 - x0) / (n - 1)
    w = []
    x = []
    w.append(h / 3)
    x.append(x0)
    for i in range(1, n - 1):
        if i % 2 == 0:
            w.append(4 / 3 * h)
        else:
            w.append(2 / 3 * h)
        x.append(x0 + i * h)
    w.append(h / 3)
    x.append(x1)
    return x, w


def weightspin(nbeta, spin, m, n, beta):
    """Function
    Calculae Wigner small d-matrix d^j_{mn}(beta)
    """

    j = spin - 1
    dmm = []
    for irot in range(nbeta):
        fnorm = (j + 1) / 2
        dmm.append(wigner_d_matrix(j, m, n, beta[irot]) * fnorm)
    return dmm


def wigner_d_matrix(j, m, n, angle):
    import math

    i1 = int((j + n) / 2)
    i2 = int((j - n) / 2)
    i3 = int((j + m) / 2)
    i4 = int((j - m) / 2)
    i5 = int((n - m) / 2)
    if i1 == -1:
        f1 = 1
    else:
        f1 = math.factorial(i1)
    if i2 == -1:
        f2 = 1
    else:
        f2 = math.factorial(i2)
    if i3 == -1:
        f3 = 1
    else:
        f3 = math.factorial(i3)
    if i4 == -1:
        f4 = 1
    else:
        f4 = math.factorial(i4)
    mink = max(0, i5)
    maxk = min(i1, i4)
    root = np.sqrt(f1 * f2 * f3 * f4)
    cosB = np.cos(angle / 2)
    sinB = np.sin(angle / 2)
    d_matrix = 0
    for k in range(mink, maxk + 1):
        if i1 - k == -1:
            x1 = 1
        else:
            x1 = math.factorial(i1 - k)
        if k == -1:
            x = 1
        else:
            x = math.factorial(k)
        if i4 - k == -1:
            x4 = 1
        else:
            x4 = math.factorial(i4 - k)
        if k - i5 == -1:
            x5 = 1
        else:
            x5 = math.factorial(k - i5)
        # dominator = ((-1)**(k-i5))*math.factorial(i1-k)*math.factorial(k)*math.factorial(i4-k)*math.factorial(k-i5)
        denominator = ((-1) ** (k - i5)) * x1 * x * x4 * x5
        numerator = cosB ** (i1 + i4 - 2 * k) * sinB ** (2 * k - i5) * root
        d_matrix = d_matrix + numerator / denominator
    return d_matrix


def set_circuit_rhfZ(n_qubits, n_electrons):
    """Function:
    Construct circuit for RHF |0000...1111> with one ancilla

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    for i in range(n_electrons):
        circuit.add_X_gate(i)
    return circuit


def set_circuit_rohfZ(n_qubits, noa, nob):
    """Function:
    Construct circuit for ROHF |0000...10101111> with one ancilla

    Author(s): Takashi Tsuchimochi
    """
    # generate circuit for rhf
    circuit = QuantumCircuit(n_qubits)
    for i in range(noa):
        circuit.add_X_gate(2 * i)
    for i in range(nob):
        circuit.add_X_gate(2 * i + 1)
    return circuit


def set_circuit_uhfZ(n_qubits, noa, nob, nva, nvb, theta_list):
    """Function:
    Construct circuit for UHF with one ancilla

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    ucc_singles(circuit, noa, nob, nva, nvb, theta_list)
    return circuit


def set_circuit_ghfZ(n_qubits, no, nv, theta_list):
    """Function:
    Construct circuit for GHF with one ancilla

    Author(s): Takashi Tsuchimochi
    """
    circuit = QuantumCircuit(n_qubits)
    ucc_singles_g(circuit, no, nv, theta_list)
    return circuit


def set_circuit_Ug(circuit, n_qubits_system, beta):
    """Function:
    Construct circuit for Ug in spin-projection (only exp[-i beta Sy])

    Author(s): Takashi Tsuchimochi
    """
    ### Ug
    for i in range(n_qubits_system):
        if i % 2 == 0:
            #
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i + 1, np.pi / 2)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_RZ_gate(i, -beta / 2)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i + 1, -np.pi / 2)
            ###
            circuit.add_H_gate(i + 1)
            circuit.add_RX_gate(i, np.pi / 2)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_RZ_gate(i, beta / 2)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_H_gate(i + 1)
            circuit.add_RX_gate(i, -np.pi / 2)


def set_circuit_ExpSy(circuit, n_qubits_system, angle):
    """Function
    Construct circuit Exp[ -i angle Sy ]

    Author(s): Takashi Tsuchimochi
    """

    for i in range(n_qubits_system):
        if i % 2 == 0:
            single_ope_Pauli(i + 1, i, circuit, angle / 2, approx=False)


def set_circuit_ExpSz(circuit, n_qubits_system, angle):
    """Function
    Construct circuit Exp[ -i angle Sz ]
    (20210205) Bug fixed
    Sz = 1/4 (-Z0 +Z1 -Z2 +Z3 ...)

    Author(s): Takashi Tsuchimochi
    """

    for i in range(n_qubits_system):
        if i % 2 == 0:
            circuit.add_RZ_gate(i, angle / 2)
        else:
            circuit.add_RZ_gate(i, -angle / 2)

def set_circuit_ExpNa(circuit, n_qubits_system, angle):
    """Function
    Construct circuit Exp[ -i angle Na ] Exp[ i angle M/2]
    Na = Number operator for alpha spin
       = M/2  -  1/2 ( Z0 + Z2 + Z4 + ...) 
    The phase Exp[ -i angle M/2 ] is canceled out here,
    and treated elsewhere.

    Author(s): Takashi Tsuchimochi
    """

    for i in range(n_qubits_system):
        if i % 2 == 0:
            circuit.add_RZ_gate(i, angle)
            
def set_circuit_ExpNb(circuit, n_qubits_system, angle):
    """Function
    Construct circuit Exp[ -i angle Nb ] Exp[ i angle M/2]
      Nb = Number operator for beta spin
         = M/2  -  1/2 ( Z1 + Z3 + Z5 + ...) 
    The phase Exp[ -i angle M/2 ] is canceled out here,
    and treated elsewhere.

    Author(s): Takashi Tsuchimochi
    """

    for i in range(n_qubits_system):
        if i % 2 == 1:
            circuit.add_RZ_gate(i, angle)

def set_circuit_Rg(circuit, n_qubits_system, alpha, beta, gamma):
    """Function
    Construct circuit Rg for complete spin-projection

    Author(s): Takashi Tsuchimochi
    """
    set_circuit_ExpSz(circuit, n_qubits_system, gamma)
    set_circuit_ExpSy(circuit, n_qubits_system, beta)
    set_circuit_ExpSz(circuit, n_qubits_system, alpha)


def controlled_Ug_gen(circuit, n_qubits, anc, alph, beta, gamm):
    """Function:
    Construct circuit for controlled-Ug in general spin-projection

    Args:
        circuit (QuantumCircuit): Circuit to be updated in return
        n_qubits (int): Total number of qubits (including ancilla)
        anc (int): The index number of ancilla (= n_qubits - 1)
        alph (float): alpha angle for spin-rotation
        beta (float): beta angle for spin-rotation
        gamm (float): gamma angle for spin-rotation

    Author(s): Takashi Tsuchimochi
    """
    ### Controlled Ug(alph,beta,gamm)
    if gamm > 1e-6:
        for i in range(n_qubits - 1):
            if i % 2 == 0:
                circuit.add_RZ_gate(i, gamm / 4)
                circuit.add_CNOT_gate(anc, i)
                circuit.add_RZ_gate(i, -gamm / 4)
                circuit.add_CNOT_gate(anc, i)
            elif i % 2 == 1:
                circuit.add_RZ_gate(i, -gamm / 4)
                circuit.add_CNOT_gate(anc, i)
                circuit.add_RZ_gate(i, gamm / 4)
                circuit.add_CNOT_gate(anc, i)

    for i in range(n_qubits - 1):
        if i % 2 == 0:
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i + 1, np.pi / 2)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_RZ_gate(i, -beta / 4)
            circuit.add_CNOT_gate(anc, i)
            circuit.add_RZ_gate(i, beta / 4)
            circuit.add_CNOT_gate(anc, i)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i + 1, -np.pi / 2)
            ###
            circuit.add_H_gate(i + 1)
            circuit.add_RX_gate(i, np.pi / 2)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_RZ_gate(i, beta / 4)
            circuit.add_CNOT_gate(anc, i)
            circuit.add_RZ_gate(i, -beta / 4)
            circuit.add_CNOT_gate(anc, i)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_H_gate(i + 1)
            circuit.add_RX_gate(i, -np.pi / 2)

    if alph > 1e-6:
        for i in range(n_qubits - 1):
            if i % 2 == 0:
                circuit.add_RZ_gate(i, alph / 4)
                circuit.add_CNOT_gate(anc, i)
                circuit.add_RZ_gate(i, -alph / 4)
                circuit.add_CNOT_gate(anc, i)
            elif i % 2 == 1:
                circuit.add_RZ_gate(i, -alph / 4)
                circuit.add_CNOT_gate(anc, i)
                circuit.add_RZ_gate(i, alph / 4)
                circuit.add_CNOT_gate(anc, i)


def controlled_Ug(circuit, n_qubits, anc, beta):
    """Function:
    Construct circuit for controlled-Ug in spin-projection

    Author(s): Takashi Tsuchimochi
    """
    ### Controlled Ug
    for i in range(n_qubits - 1):
        if i % 2 == 0:
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i + 1, np.pi / 2)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_RZ_gate(i, -beta / 4)
            circuit.add_CNOT_gate(anc, i)
            circuit.add_RZ_gate(i, beta / 4)
            circuit.add_CNOT_gate(anc, i)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i + 1, -np.pi / 2)
            ###
            circuit.add_H_gate(i + 1)
            circuit.add_RX_gate(i, np.pi / 2)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_RZ_gate(i, beta / 4)
            circuit.add_CNOT_gate(anc, i)
            circuit.add_RZ_gate(i, -beta / 4)
            circuit.add_CNOT_gate(anc, i)
            circuit.add_CNOT_gate(i + 1, i)
            circuit.add_H_gate(i + 1)
            circuit.add_RX_gate(i, -np.pi / 2)


def cost_proj(
    Quket,
    print_level,
    qulacs_hamiltonianZ,
    qulacs_s2Z,
    coef0_H,
    coef0_S2,
    kappa_list,
    theta_list=0,
    threshold=0.01,
):
    """Function:
    Energy functional for projected methods (phf, puccsd, puccd, opt_puccd)

    Author(s): Takashi Tsuchimochi
    """
    t1 = time.time()
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    n_electrons = Quket.n_electrons
    rho = Quket.rho
    DS = Quket.DS
    anc = Quket.anc
    n_qubits_system = Quket.n_qubits
    n_qubits = n_qubits_system + 1
    ndim1 = noa * nva + nob * nvb
    ndim2aa = int(noa * (noa - 1) * nva * (nva - 1) / 4)
    ndim2ab = int(noa * nob * nva * nvb)
    ndim2bb = int(nob * (nob - 1) * nvb * (nvb - 1) / 4)
    ndim2 = ndim2aa + ndim2ab + ndim2bb
    ref = Quket.ansatz


    state = QuantumState(n_qubits)
    if noa == nob:
        circuit_rhf = set_circuit_rhfZ(n_qubits, n_electrons)
    else:
        circuit_rhf = set_circuit_rohfZ(n_qubits, noa, nob)
    circuit_rhf.update_quantum_state(state)
    if ref == "phf":
        circuit_uhf = set_circuit_uhfZ(n_qubits, noa, nob, nva, nvb, kappa_list)
        circuit_uhf.update_quantum_state(state)
        if print_level > 0:
            SaveTheta(ndim1, kappa_list, cf.tmp)
    elif ref == "sghf":
        circuit_ghf = set_circuit_ghfZ(n_qubits, noa + nob, nva + nvb, kappa_list)
        circuit_ghf.update_quantum_state(state)
        if print_level > 0:
            SaveTheta(ndim1, kappa_list, cf.tmp)
    elif ref == "puccsd":
        # First prepare UHF determinant
        circuit_uhf = set_circuit_uhfZ(n_qubits, noa, nob, nva, nvb, kappa_list)
        circuit_uhf.update_quantum_state(state)
        # Then prepare UCCSD
        theta_list_rho = theta_list / rho
        circuit = set_circuit_uccsd(n_qubits, noa, nob, nva, nvb, 0, theta_list_rho)
        for i in range(rho):
            circuit.update_quantum_state(state)
        if print_level > 0:
            SaveTheta(ndim1 + ndim2, theta_list, cf.tmp)
    elif ref == "puccd":
        # First prepare UHF determinant
        circuit_uhf = set_circuit_uhfZ(n_qubits, noa, nob, nva, nvb, kappa_list)
        circuit_uhf.update_quantum_state(state)
        # Then prepare UCCD
        theta_list_rho = theta_list / rho
        circuit = set_circuit_uccd(n_qubits, noa, nob, nva, nvb, theta_list_rho)
        for i in range(rho):
            circuit.update_quantum_state(state)
        if print_level > 0:
            SaveTheta(ndim2, theta_list, cf.tmp)
    elif ref == "opt_puccd":
        if DS:
            # First prepare UHF determinant
            circuit_uhf = set_circuit_uhfZ(n_qubits, noa, nob, nva, nvb, theta_list)
            circuit_uhf.update_quantum_state(state)
            # Then prepare UCCD
            theta_list_rho = theta_list[ndim1 : ndim1 + ndim2] / rho
            circuit = set_circuit_uccd(n_qubits, noa, nob, nva, nvb, theta_list_rho)
            for i in range(rho):
                circuit.update_quantum_state(state)
        else:
            # First prepare UCCD
            theta_list_rho = theta_list[ndim1 : ndim1 + ndim2] / rho
            circuit = set_circuit_uccd(n_qubits, noa, nob, nva, nvb, theta_list_rho)
            for i in range(rho):
                circuit.update_quantum_state(state)
            # then rotate
            circuit_uhf = set_circuit_uhfZ(n_qubits, noa, nob, nva, nvb, theta_list)
            circuit_uhf.update_quantum_state(state)

        if print_level > 0:
            SaveTheta(ndim1 + ndim2, theta_list, cf.tmp)
    elif ref == "opt_psauccd":
        theta_list_rho = theta_list[ndim1 : ndim1 + ndim2] / rho
        circuit = set_circuit_sauccd(n_qubits, noa, nva, theta_list_rho)
        for i in range(rho):
            circuit.update_quantum_state(state)
        circuit_uhf = set_circuit_uhfZ(n_qubits, noa, nob, nva, nvb, theta_list)
        circuit_uhf.update_quantum_state(state)

    if print_level > 1:
        prints("State before projection")
        print_state(state, n_qubits=n_qubits_system)
        if ref == "puccsd" or ref == "opt_puccd":
            print_amplitudes(theta_list, noa, nob, nva, nvb, threshold)
    #    '''
    ### grid loop ###
    ### a list to compute the probability to observe 0 in ancilla qubit
    ### Array for <HUg>, <S2Ug>, <Ug>
    HUg = []
    S2Ug = []
    Ug = []
    Ep = 0
    S2 = 0
    Norm = 0
    nalph = max(Quket.projection.euler_ngrids[0], 1)
    nbeta = max(Quket.projection.euler_ngrids[1], 1)
    ngamm = max(Quket.projection.euler_ngrids[2], 1)
    ig = 0
    for ialph in range(nalph):
        alph = Quket.projection.sp_angle[0][ialph]
        alph_coef = Quket.projection.sp_weight[0][ialph]
        for ibeta in range(nbeta):
            beta = Quket.projection.sp_angle[1][ibeta]
            beta_coef = Quket.projection.sp_weight[1][ibeta] * Quket.projection.dmm[ibeta]
            for igamm in range(ngamm):
                gamm = Quket.projection.sp_angle[2][igamm]
                gamm_coef = Quket.projection.sp_weight[2][igamm]

                ### Copy quantum state of UHF (cannot be done in real device) ###
                state_g = QuantumState(n_qubits)
                state_g.load(state)
                ### Construct Ug test
                circuit_ug = QuantumCircuit(n_qubits)
                ### Hadamard on anc
                circuit_ug.add_H_gate(anc)
                # circuit_ug.add_X_gate(anc)
                # controlled_Ug(circuit_ug, n_qubits, anc, beta)
                controlled_Ug_gen(circuit_ug, n_qubits, anc, alph, beta, gamm)
                # circuit_ug.add_X_gate(anc)
                circuit_ug.add_H_gate(anc)
                circuit_ug.update_quantum_state(state_g)

                ### Compute expectation value <HUg> ###
                HUg.append(qulacs_hamiltonianZ.get_expectation_value(state_g))
                ### <S2Ug> ###
                # print_state(state_g)
                S2Ug.append(qulacs_s2Z.get_expectation_value(state_g))
                ### <Ug> ###
                p0 = state_g.get_zero_probability(anc)
                p1 = 1 - p0
                Ug.append(p0 - p1)
                ### Norm accumulation ###
                Norm += alph_coef * beta_coef * gamm_coef * Ug[ig]
                Ep += alph_coef * beta_coef * gamm_coef * HUg[ig]
                S2 += alph_coef * beta_coef * gamm_coef * S2Ug[ig]
                ig += 1
    #        print('p0 : ',p0,'  p1 : ',p1,  '  p0 - p1 : ',p0-p1)
    #    '''
    #    print("Time: ",t2-t1)
    ### Energy calculation <HP>/<P> and <S**2P>/<P> ###
    Ep /= Norm
    S2 /= Norm
    t2 = time.time()
    cpu1 = t2 - t1
    Ep += coef0_H
    S2 += coef0_S2
    if print_level == -1:
        prints(
            "Initial E[%s] = " % ref,
            "{:.12f}".format(Ep),
            "  <S**2> =",
            "% 17.15f" % S2,
            "  rho = %d" % rho,
        )
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(
            "{cyc:5}:".format(cyc=cf.icyc),
            "  E[%s] = " % ref,
            "{:.12f}".format(Ep),
            "  <S**2> =",
            "% 17.15f" % S2,
            "  CPU Time = ",
            "%5.2f" % cput,
            " (%2.2f / step)" % cpu1,
        )
    if print_level > 1:
        prints(
            "Final:  E[%s] = " % ref,
            "{:.12f}".format(Ep),
            "  <S**2> =",
            "% 17.15f" % S2,
            "  rho = %d" % rho,
        )
        print_state(state, n_qubits=n_qubits - 1)
        if ref == "puccsd" or ref == "opt_puccd":
            print_amplitudes(theta_list, noa, nob, nva, nvb)
        prints("HUg", HUg)
        prints("Ug", Ug)


    return Ep, S2


def S2Proj(Quket,Q):
    """Function
       Perform spin-projection to QuantumState |Q>

          |Q'>  =  Ps |Q>

       where Ps is a spin-projection operator (non-unitary).

          Ps = \sum_i^ng   wg[i] Ug[i]

       This function provides a shortcut to |Q'>, which is unreal.
       One actually needs to develop a quantum circuit for this (See PRR 2, 043142 (2020)).


    Author(s): Takashi Tsuchimochi
    """
    spin = Quket.projection.spin
    s = (spin - 1) / 2
    Ms = Quket.projection.Ms / 2   

    n_qubits = Q.get_qubit_count()
    state_P = QuantumState(n_qubits)
    state_P.multiply_coef(0)
    nalpha = max(Quket.projection.euler_ngrids[0], 1)
    nbeta = max(Quket.projection.euler_ngrids[1], 1)
    ngamma = max(Quket.projection.euler_ngrids[2], 1)
    for ialpha in range(nalpha):
        alpha = Quket.projection.sp_angle[0][ialpha]
        alpha_coef = Quket.projection.sp_weight[0][ialpha] * np.exp(1j * alpha * Ms)
        for ibeta in range(nbeta):
            beta = Quket.projection.sp_angle[1][ibeta]
            beta_coef = Quket.projection.sp_weight[1][ibeta] * Quket.projection.dmm[ibeta]
            for igamma in range(ngamma):
                gamma = Quket.projection.sp_angle[2][igamma]
                gamma_coef = Quket.projection.sp_weight[2][igamma] * np.exp(1j * gamma * Ms)
                # Total Weight
                coef = (2 * s + 1) / (8 * np.pi) * (alpha_coef * beta_coef * gamma_coef)

                state_g = QuantumState(n_qubits)
                state_g.load(Q)
                circuit_Rg = QuantumCircuit(n_qubits)
                set_circuit_Rg(circuit_Rg, n_qubits, alpha, beta, gamma)
                circuit_Rg.update_quantum_state(state_g)
                state_g.multiply_coef(coef)

                state_P.add_state(state_g)

    # Normalize
    norm2 = state_P.get_squared_norm()
    if norm2 < 1e-8:
        error("Norm of spin-projected state is too small!\n",
              "This usually means the broken-symmetry state has NO component ",
              "of the target spin.")
    state_P.normalize(norm2)
    # print_state(state_P,name="P|Q>",threshold=1e-6)
    return state_P


def NProj(Quket,Q):
    """Function
       Perform number-projection to QuantumState |Q>

          |Q'>  =  PN |Q>

       where PN is a number-projection operator (non-unitary).

          PN = \sum_i^ng   wg[i] Ug[i]

       This function provides a shortcut to |Q'>, which is unreal.
       One actually needs to develop a quantum circuit for this (See QST 6, 014004 (2021)).


    Author(s): Takashi Tsuchimochi
    """
    n_qubits = Q.get_qubit_count()
    state_P = QuantumState(n_qubits)
    state_P.multiply_coef(0)
    state_g = QuantumState(n_qubits)
    nphi = max(Quket.projection.number_ngrids, 1)
    #print_state(Q)
    for iphi in range(nphi):
        coef = Quket.projection.np_weight[iphi] * np.exp(1j * Quket.projection.np_angle[iphi] * (Quket.projection.n_active_electrons - Quket.projection.n_active_orbitals))
        state_g= Q.copy()
        circuit = QuantumCircuit(n_qubits)
        set_circuit_ExpNa(circuit,n_qubits,Quket.projection.np_angle[iphi])
        set_circuit_ExpNb(circuit,n_qubits,Quket.projection.np_angle[iphi])
        circuit.update_quantum_state(state_g)
        state_g.multiply_coef(coef)
        state_P.add_state(state_g)       
    norm2 = state_P.get_squared_norm()
    if norm2 < 1e-8:
        error("Norm of number-projected state is too small!\n",
              "This usually means the broken-symmetry state has NO component ",
              "of the target number.")
    state_P.normalize(norm2)
    return state_P

