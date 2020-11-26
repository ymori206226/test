import sys
import numpy as np
import time
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import QubitOperator
from qulacs import QuantumState
from qulacs import QuantumCircuit
from .ucclib import ucc_singles
from . import config
from . import utils


def set_circuit_rhf(n_qubit_system,n_electron):
    """ Function:
    Construct circuit for RHF |0000...1111> 
    """
    circuit = QuantumCircuit(n_qubit_system)
    for i in range(n_electron):
        circuit.add_X_gate(i)
    return circuit

def set_circuit_rohf(n_qubit_system,noa,nob):
    """ Function:
    Construct circuit for ROHF |0000...10101111> 
    """
    circuit = QuantumCircuit(n_qubit_system)
    for i in range(noa):
        circuit.add_X_gate(2*i)
    for i in range(nob):
        circuit.add_X_gate(2*i+1)
    return circuit
    
def set_circuit_uhf(n_qubit_system,noa,nob,nva,nvb,kappa_list):
    """ Function:
    Construct circuit for UHF by orbital rotation 
    """
    circuit = QuantumCircuit(n_qubit_system)
    ucc_singles(circuit,noa,nob,nva,nvb,kappa_list)
    return circuit


def cost_uhf(print_level,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,kappa_list):
    """ Function:
    Energy functional of UHF
    """
    t1 = time.time()
    state = QuantumState(n_qubit_system)
    if noa==nob:
        circuit_rhf = set_circuit_rhf(n_qubit_system,n_electron)
    else:
        circuit_rhf = set_circuit_rohf(n_qubit_system,noa,nob)
    circuit_rhf.update_quantum_state(state)
    circuit_uhf = set_circuit_uhf(n_qubit_system,noa,nob,nva,nvb,kappa_list)
    circuit_uhf.update_quantum_state(state)
    Euhf = qulacs_hamiltonian.get_expectation_value(state)
    S2 = qulacs_s2.get_expectation_value(state)
    t2 = time.time() 
    cpu1 = t2 - t1
    if print_level > 0:
        cput = t2 - config.t_old 
        config.t_old = t2
        with open(config.log,'a') as f:
            print(" E[UHF] = ", '{:.12f}'.format(Euhf),  "  <S**2> =", '%2.15f' % S2, "  CPU Time = ", '%2.5f' % cput, " (%2.5f / step)" % cpu1, file=f)
        utils.SaveTheta(noa*nva+nob*nvb,kappa_list,config.tmp)
    if print_level > 1:
        with open(config.log,'a') as f:
            print('(UHF state)',file=f)
        utils.print_state(state,n_qubit_system)
    return Euhf, S2

def mix_orbitals(noa,nob,nva,nvb,mix_level,random=False,angle=np.pi/4):
    """ Function:
    Prepare kappa_list that mixes homo-lumo to break spin-symmetry. 
    ===parameters===
        mix_level:  the number of orbital pairs to be mixed 
        random:     [Bool] randomly mix orbitals
        angle:      mixing angle
        kappa:      kappa amplitudes in return
    """
    kappa = np.zeros(noa*nva + nob*nvb) 
    ia = 0
    if random:
        for a in range(nva):
            for i in range(noa):
                kappa[i + a * noa] = np.random.rand() - 0.5
                kappa[i + a * nob + noa*nva] = np.random.rand()-0.5
    else:
        for p in range(mix_level):
            a = p
            iA = (noa - (p + 1))
            iB = (nob - (p + 1))
            kappa[iA + a * noa] = angle 
            kappa[iB + a * nob + noa*nva] = -angle 
    return kappa
    

def bs_orbitals(kappa,ialpha,aalpha,jbeta,bbeta,noa,nob,nva,nvb):
    """ Function:
    Prepare kappa amplitudes that generates broken-symmetry exicted state 
    #   kappa:      kappa amplitudes in return
    #   ialpha -->  aalpha
    #   jbeta -->  bbeta
    """
    kappa[ialpha + (aalpha-noa) * noa] += np.pi/6
    kappa[jbeta + (bbeta-nob) * nob + noa*nva] += -np.pi/6

    
