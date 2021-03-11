import time

from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import PauliRotation

from . import config as cf
from .fileio import SaveTheta, printmat, print_state, prints
from .upcclib import upcc_Gsingles
from .utils import orthogonal_constraint


def set_circuit_bcs(ansatz, n_qubits, n_orbitals, ndim1, ndim, theta_list, k):
    circuit = QuantumCircuit(n_qubits)
    target_list = np.empty(2)
    pauli_index = np.empty(2)
    for i in range(k):
        ioff  = i*ndim
        for p in range(n_orbitals):
            pa = 2*p
            pb = 2*p + 1
            target_list = pa, pb

            pauli_index = 1, 2
            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])
            circuit.add_gate(gate)

            pauli_index = 2, 1
            gate = PauliRotation(target_list, pauli_index, -theta_list[p+ioff])
            circuit.add_gate(gate)

            if "ebcs" in ansatz:
                if p < n_orbitals - 1:
                    circuit.add_CNOT_gate(pa, pa+2)
                    circuit.add_CNOT_gate(pb, pb+2)
        upcc_Gsingles(circuit, n_orbitals, theta_list, ndim1, n_orbitals, i)
    return circuit


def cost_bcs(Quket, print_level, theta_list, k):
    """Function:
    Energy functional of kBCS

    Author(s): Takahiro Yoshikura
    """
    t1 = time.time()

    noa = Quket.noa
    nob = Quket.nob
    nva = Quket.nva
    nvb = Quket.nvb
    n_qubits = Quket.n_qubits
    det = Quket.det
    ndim1 = Quket.ndim1
    ndim = Quket.ndim

    state = QuantumState(n_qubits)
    state.set_computational_basis(det)
    circuit = set_circuit_bcs(ansatz, n_qubits, n_orbitals, ndim1, ndim,
                              theta_list, k)
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
        SaveTheta(ndim, theta_list, cf.tmp)
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


