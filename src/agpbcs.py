import time
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import PauliRotation
from . import config as cf
from .fileio import (
    SaveTheta,
    printmat,
    print_state,
    prints,
)
from .upcclib import upcc_Gsingles
from .utils import orthogonal_constraint
def set_circuit_bcs(n_qubit, norbs, theta_list, k):
    circuit = QuantumCircuit(n_qubit)
    ndim1 = norbs*(norbs-1)//2 
    ndim = norbs + ndim1
    for i in range(k):
        ioff  = i * ndim
        for p in range(norbs):
            pa = 2*p
            pb = 2*p + 1
            target_list = [2*p, 2*p+1]
            pauli_index = [1,2]
            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff]) 
            circuit.add_gate(gate) 
            pauli_index = [2,1]
            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])     
            circuit.add_gate(gate)
            if "ebcs" in Quket.ansatz:
                if (p < norbs-1):
                    circuit.add_CNOT_gate(2*p, 2*p+2)
                    circuit.add_CNOT_gate(2*p+1, 2*p+3)
        upcc_Gsingles(circuit, norbs, theta_list, ndim1, norbs, i)
    return circuit

def cost_bcs(
    Quket,
    print_level,
    theta_list,
    k,
):
    """Function:
    Energy functional of kBCS

    Author(s): Takahiro Yoshikura
    """

    t1 = time.time()
    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    n_qubit = Quket.n_qubit
    det = Quket.det
    norbs = noa + nva
    ndim1 = (norbs-1)*norbs//2
    ndim2 = norbs
    state = QuantumState(n_qubit)
    state.set_computational_basis(det)

    circuit = set_circuit_bcs(n_qubit, norbs, theta_list, k)
    circuit.update_quantum_state(state)

    if Quket.projection.SpinProj:
        from .phflib import S2Proj
        state_P = S2Proj(Quket,state)
        state   = state_P.copy()
    if Quket.projection.NumberProj:
        from .phflib import NProj
        state_P = NProj(Quket,state)
        state   = state_P.copy()
    #print_state(state, threshold=cf.print_amp_thres)

    Ebcs = Quket.qulacs.Hamiltonian.get_expectation_value(state)
    cost = Ebcs
    ### Project out the states contained in 'lower_states'
    cost += orthogonal_constraint(Quket, state)

    S2 = Quket.qulacs.S2.get_expectation_value(state)
    t2 = time.time()
    cpu1 = t2 - t1
    if print_level == 1:
        cput = t2 - cf.t_old
        cf.t_old = t2
        cf.icyc += 1
        prints(
            "{cyc:5}:".format(cyc=cf.icyc),
            "  E[{}-kBCS] = {:.12f}".format(k, Ebcs),
            "  <S**2> =",
            "% 17.15f" % S2,
            "  CPU Time = ",
            "%5.2f" % cput,
            " (%2.2f / step)" % cpu1,
        )
        SaveTheta(k * (ndim1 + ndim2), theta_list, cf.tmp)
    if print_level > 1:
        prints(
            "Final:  E[{}-kBCS] = {:.12f}".format(k, Ebcs),
            "  <S**2> =",
            "% 17.15f" % S2,
        )
        prints("\n({}-kBCS state)".format(k))
        printmat(theta_list)
        print_state(state, threshold=Quket.print_amp_thres)
    # Store kBCS wave function
    Quket.state = state
    return cost, S2


