import time

from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import PauliRotation

from . import config as cf
from .fileio import SaveTheta, printmat, print_state, prints,
from .upcclib import upcc_Gsingles
from .utils import orthogonal_constraint


def set_circuit_bcs(Quket, theta_list, k):
    ndim1 = (Quket.n_orbitals-1)*Quket.n_orbitals//2
    ndim2 = Quket.n_orbitals
    ndim = ndim1 + ndim2
    #ndim = theta_list.size
    circuit = QuantumCircuit(Quket.n_qubits)

    for i in range(k):
        ioff  = i * ndim
        for p in range(Quket.n_orbitals):
            pa = 2*p
            pb = 2*p + 1
            target_list = [pa, pb]
            pauli_index = [1, 2]
            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])
            circuit.add_gate(gate)
            pauli_index = [2, 1]
            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])
            circuit.add_gate(gate)
            if "ebcs" in Quket.ansatz:
                if (p < Quket.n_orbitals-1):
                    circuit.add_CNOT_gate(pa, pa+2)
                    circuit.add_CNOT_gate(pb, pb+2)
        upcc_Gsingles(circuit, Quket.n_orbitals, theta_list, ndim1, ndim2, i)
    return circuit


#def set_circuit_bcs(n_qubits, n_orbitals, theta_list, k):
#    circuit = QuantumCircuit(n_qubits)
#    ndim1 = n_orbitals*(n_orbitals-1)//2
#    ndim2 = n_orbitals
#    ndim = n_orbitals + ndim1
#    for i in range(k):
#        ioff  = i * ndim
#        for p in range(n_orbitals):
#            pa = 2*p
#            pb = 2*p + 1
#            target_list = [2*p, 2*p+1]
#            pauli_index = [1,2]
#            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])
#            circuit.add_gate(gate)
#            pauli_index = [2,1]
#            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])
#            circuit.add_gate(gate)
#            if "ebcs" in Quket.ansatz:
#                if (p < n_orbitals-1):
#                    circuit.add_CNOT_gate(2*p, 2*p+2)
#                    circuit.add_CNOT_gate(2*p+1, 2*p+3)
#        upcc_Gsingles(circuit, n_orbitals, theta_list, ndim1, n_orbitals, i)
#    return circuit


def cost_bcs(Quket, print_level, theta_list, k):
    """Function:
    Energy functional of kBCS

    Author(s): Takahiro Yoshikura
    """
    t1 = time.time()

#    ndim1 = (Quket.n_orbitals-1)*Quket.n_orbitals//2
#    ndim2 = Quket.n_orbitals
#    ndim = ndim1 + ndim2
    #ndim = theta_list.size

    state = QuantumState(Quket.n_qubits)
    state.set_computational_basis(Quket.det)

    circuit = set_circuit_bcs(Quket, theta_list, k)
    circuit.update_quantum_state(state)

    if Quket.projection.SpinProj:
        from .phflib import S2Proj
        state_P = S2Proj(Quket, state)
        state   = state_P.copy()
    if Quket.projection.NumberProj:
        from .phflib import NProj
        state_P = NProj(Quket, state)
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
        prints(f"{cf.icyc:5d}: "
               f"E[{k}-kBCS] = {Ebcs:.12f}  "
               f"<S**2> = {S2:17.15f}  "
               f"CPU Time = {cput:5.2f}  "
               f"({cpu1:2.2f} / step)")
        #SaveTheta(k*(ndim1+ndim2), theta_list, cf.tmp)
        SaveTheta(theta_list, cf.tmp)
    if print_level > 1:
        prints(f"Final: "
               f"E[{k}-kBCS] = {Ebcs:.12f}  "
               f"<S**2> = {S2:17.15f}")
        prints(f"\n({k}-kBCS state)")
        printmat(theta_list)
        print_state(state, threshold=Quket.print_amp_thres)
    # Store kBCS wave function
    Quket.state = state
    return cost, S2


