"""
#######################
#        quket        #
#######################

expope.py

Functions to prepare several types of rotations, including singles and doubles rotations.

"""
import numpy as np
from qulacs.gate import PauliRotation

from . import config as cf


def Gdouble_ope(p, q, r, s, circuit, theta):
    """ Function:
    Construct exp[theta ( p!q!rs - s!r!qp ) ] as a whole unitary
    and add to circuit.
    Here, max(p,q,r,s) = p is assumed.
    There are 5 cases:
        (1) p > q > r > s (includes p > q > s > r)
        (2) p > r > q > s (includes p > r > s > q)
        (3) p = r > q > s (includes p = r > s > q)
        (4) p > q = r > s (includes p > q = s > r)
        (5) p > r > q = s (includes p > s > q = r)
    Note that in Qulacs, the function "PauliRotation" rotates as
        Exp [i theta/2 ...]
    so theta is divided by two.
    Accordingly, we need to multiply theta by two.

    Args:
        p (int): excitation index.
        q (int): excitation index.
        r (int): excitation index.
        s (int): excitation index.
        circuit (QuantumCircuit): circuit to be updated.
        theta (float): real rotation parameter.

    Returns:
        circuit (QuantumCircuit): circuit to be updated.

    Author(s): Takashi Tsuchimochi
    """
    if p != max(p, q, r, s):
        print(f"Error!   {p=}!= {max(p, q, r, s)=}")
        return
    if p == q or r == s:
        print(f"Caution:  {p=} == {q=}  or  {r=} == {s=}")
        return

    if p == r:
        if q > s:
            # p^ q^ p s  (p > q, p > s)
            Gdoubles_pqps(p, q, s, circuit, theta)
        elif q < s:
            # p^ q^ p s =  p^ s^ p q  (p > s, p > q)
            Gdoubles_pqps(p, s, q, circuit, theta)
        else:
            print("Error!  p^r^ pr - h.c. = zero")
    elif p == s:  #(necessarily  r < s)
        Gdoubles_pqps(p, q, r, circuit, -theta)
    elif q == r:
        if q > s:
            # p^ q^ q s  (p > q, q > s)
            Gdoubles_pqqs(p, q, s, circuit, theta)
        elif q < s:
            # p^ q^ q s = - p^ q^ s q  (p > q, s > q)
            Gdoubles_pqrq(p, q, s, circuit, -theta)
    elif q == s:
        if q < r:
            # p^ q^ r q  (p > q, r > q)
            Gdoubles_pqrq(p, q, r, circuit, theta)
        elif q > r:
            # p^ q^ r q  = - p^ q^ q r  (p > q, q > r)
            Gdoubles_pqqs(p, q, r, circuit, -theta)
    else:
        if r > s:
            Gdoubles_pqrs(p, q, r, s, circuit, theta)
        elif r < s:
            Gdoubles_pqrs(p, q, s, r, circuit, -theta)


def Gdoubles_pqrs(p, q, r, s, circuit, theta, approx=cf.approx_exp):
    """ Function
    Given
            Epqrs = p^ q^ r s
    with p > q, r > s and max(p, q, r, s) = p,
    compute Exp[i theta (Epqrs - Epqrs!)]

    Author(s): Takashi Tsuchimochi
    """
    i4 = p  # (Fixed)
    if q > r > s:
        i1 = s
        i2 = r
        i3 = q
    elif r > q > s:
        i1 = s
        i2 = q
        i3 = r
    elif r > s > q:
        i1 = q
        i2 = s
        i3 = r
    ndim1 = i2 - i1 - 1
    ndim2 = ndim1 + i4 - i3 - 1
    ndim = ndim2 + 4

    ### Type (a):
    ### (1)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p X_q X_r X_s)]
    ### (2)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p Y_q X_r X_s)]
    ### (3)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p Y_q X_r Y_s)]
    ### (4)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p Y_q Y_r X_s)]
    ### (5)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p X_q X_r Y_s)]
    ### (6)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p X_q Y_r X_s)]
    ### (7)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p X_q Y_r Y_s)]
    ### (8)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p Y_q Y_r Y_s)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(i1+1, i2))
    target_list[ndim1:ndim2] = list(range(i3+1, i4))
    target_list[ndim2:] = p, q, r, s
    pauli_index = [3]*ndim

    ### (1)              Yp,Xq,Xr,Xs
    pauli_index[ndim2:] = 2, 1, 1, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (2)              Xp,Yq,Xr,Xs
    pauli_index[ndim2:] = 1, 2, 1, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (3)              Yp,Yq,Xr,Ys
    pauli_index[ndim2:] = 2, 2, 1, 2
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (4)              Yp,Yq,Yr,Xs
    pauli_index[ndim2:] = 2, 2, 2, 1
    gate = PauliRotation(target_list, pauli_index, theta/4)
    circuit.add_gate(gate)

    ### (5)              Xp,Xq,Xr,Ys
    pauli_index[ndim2:] = 1, 1, 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (6)              Xp,Xq,Yr,Xs
    pauli_index[ndim2:] = 1, 1, 2, 1
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (7)              Yp,Xq,Yr,Ys
    pauli_index[ndim2:] = 2, 1, 2, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)

    ### (8)              Xp,Yq,Yr,Ys
    pauli_index[ndim2:] = 1, 2, 2, 2
    gate = PauliRotation(target_list, pauli_index, -theta/4)
    circuit.add_gate(gate)


def Gdoubles_pqps(p, q, s, circuit, theta):
    """ Function
    Given
            Epqps = p^ q^ p s
    with p > q > s, compute Exp[i theta (Epqps - Epqps!)]
    """
    ndim1 = q - s - 1
    ndim = ndim1 + 3

    ### Type (b)
    ### (1)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k (Z_p X_q Y_s)]
    ### (2)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k (    Y_q X_s)]
    ### (3)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k (Z_p Y_q X_s)]
    ### (4)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k (    X_q Y_s)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(s+1, q))
    target_lsit[ndim1:] = p, q, s
    pauli_index = [3]*ndim

    ### (1)              Zp,Xq,Ys
    pauli_index[ndim1:] = 3, 1, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Ip,Yq,Xs
    pauli_index[ndim1:] = 0, 2, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Zp,Yq,Xs
    pauli_index[ndim1:] = 3, 2, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Ip,Xq,Ys
    pauli_index[ndim1:] = 0, 1, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)


def Gdoubles_pqqs(p, q, s, circuit, theta):
    """ Function
    Given
            Epqps = p^ q^ q s
    with p > q > s, compute Exp[i theta (Epqqs - Epqqs!)]
    """
    ndim1 = q - s - 1
    ndim2 = ndim1 + p - q - 1
    ndim = ndim2 + 3

    ### Type (c)
    ### (1)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (X_p Z_q Y_s)]
    ### (2)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (Y_p     X_s)]
    ### (3)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (Y_p Z_q X_s)]
    ### (4)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (X_p     Y_s)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(s+1, q))
    target_list[ndim1:ndim2] = list(range(q+1, p))
    target_list[ndim2:] = p, q, s
    pauli_index = [3]*ndim

    ### (1)              Xp,Zq,Ys
    pauli_index[ndim2:] = 1, 3, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Yp,Iq,Xs
    pauli_index[ndim2:] = 2, 0, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Yp,Zq,Xs
    pauli_index[ndim2:] = 2, 3, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Xp,Iq,Ys
    pauli_index[ndim2:] = 1, 0, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)


def Gdoubles_pqrq(p, q, r, circuit, theta):
    """ Function
    Given
            Epqps = p^ q^ r q
    with p > r > q, compute Exp[i theta (Epqrq - Epqrq!)]
    """
    ndim1 = p - r - 1
    ndim = ndim1 + 3

    ### Type (d)
    ### (1)   Exp[ i theta/4  Prod_{k=r+1}^{p-1} Z_k  (X_p Z_q Y_r)]
    ### (2)   Exp[ i theta/4  Prod_{k=r+1}^{p-1} Z_k  (Y_p     X_r)]
    ### (3)   Exp[-i theta/4  Prod_{k=r+1}^{p-1} Z_k  (Y_p Z_q X_r)]
    ### (4)   Exp[-i theta/4  Prod_{k=r+1}^{p-1} Z_k  (X_p     Y_r)]
    target_list = [0]*ndim
    target_list[:ndim1] = list(range(r+1, p))
    target_list[ndim1:] = p, q, r
    pauli_index = [3]*ndim

    ### (1)              Xp,Zq,Yr
    pauli_index[ndim1:] = 1, 3, 2
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (2)              Yp,Iq,Xr
    pauli_index[ndim1:] = 2, 0, 1
    gate = PauliRotation(target_list, pauli_index, theta/2)
    circuit.add_gate(gate)

    ### (3)              Yp,Zq,Xr
    pauli_index[ndim1:] = 2, 3, 1
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)

    ### (4)              Xp,Iq,Yr
    pauli_index[ndim1:] = 1, 0, 2
    gate = PauliRotation(target_list, pauli_index, -theta/2)
    circuit.add_gate(gate)


def bcs_single_ope(p, circuit, lam, theta):
    """Function
    Construct circuit for
           exp[ -i lam Z/2 ] exp[ -i theta Y/2) ]
    acting on 2p and 2p+1 qubits,
    required for BCS wave function.

    Args:
        p (int): orbital index
        circuit (QuantumCircuit): circuit to be updated
        lam (float): phase parameter
        theta (float): rotation parameter

    Returns:
        circuit (QuantumCircuit): circuit to be updated
    """
    pass
