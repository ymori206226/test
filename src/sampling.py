"""
#######################
#        quket        #
#######################

sampling.py

Functions related to sampling simulations.
(2020/12/06) Currently disabled.

"""

import sys
import numpy as np
import time
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import QubitOperator
from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs.gate import P0,P1,add, merge, Identity, X, Y, Z
from .ucclib import ucc_singles
from . import config
from .fileio import prints



def sample_observable(state, obs, n_sample):
    """ Function
    Args:
        state (qulacs.QuantumState):
        obs (qulacs.Observable)
        n_sample (int):  number of samples for each observable
    Return:
        :float: sampled expectation value of the observable

    Author(s): Takashi Tsuchimochi
    """
    n_term = obs.get_term_count()
    n_qubit = obs.get_qubit_count()

    exp = 0
    buf_state = QuantumState(n_qubit)
    for i in range(n_term):
        pauli_term = obs.get_term(i)
        coef = pauli_term.get_coef()
        pauli_id = pauli_term.get_pauli_id_list()
        pauli_index = pauli_term.get_index_list()
        if len(pauli_id) == 0:  # means identity
            exp += coef
            continue
        buf_state.load(state)
        measurement_circuit = QuantumCircuit(n_qubit)
        mask = ''.join(['1' if n_qubit-1-k in pauli_index
                        else '0' for k in range(n_qubit)])
        for single_pauli, index in zip(pauli_id, pauli_index):
            if single_pauli == 1:
                measurement_circuit.add_H_gate(index)
            elif single_pauli == 2:
                measurement_circuit.add_Sdag_gate(index)
                measurement_circuit.add_H_gate(index)
        measurement_circuit.update_quantum_state(buf_state)
        samples = buf_state.sampling(n_sample)
        mask = int(mask, 2)
        exp += coef*sum(list(map(lambda x: (-1) **
                                     (bin(x & mask).count('1')), samples)))/n_sample

    return exp

def adaptive_sample_observable(state, obs, n_sample):
    """
    Args:
        state (qulacs.QuantumState):
        obs (qulacs.Observable)
        n_sample (int):  number of samples for each observable
    Return:
        :float: sampled expectation value of the observable
    """
    n_term = obs.get_term_count()
    n_qubit = obs.get_qubit_count()

    exp = 0
    buf_state = QuantumState(n_qubit)
    ### check the coefficients for each term...
    coef_list=[]
    sum_coef = 0
    for i in range(n_term):
        pauli_term = obs.get_term(i)
        coef_list.append(abs(pauli_term.get_coef()))
        sum_coef += abs(pauli_term.get_coef())
#    ### sort
    indices = [*range(n_term)]
    sorted_indices = sorted(indices, key=lambda i: -coef_list[i])
    sorted_coef_list = [coef_list[i] for i in sorted_indices]
    ### determine sampling wight
    n_sample_total = n_sample * n_term
    n_sample_list=[]
    n_count = 0
    for i in range(n_term):
        n_sample_list.append(int( n_sample_total * coef_list[i]/sum_coef))
        n_count += n_sample_list[i]
    n_rest = n_sample_total - n_count
    for i in range(n_rest):
        n_sample_list[sorted_indices[i]] +=1

    j = 0
    for i in range(n_term):
        if n_sample_list[i] == 0:
            continue
        j = j + n_sample_list[i]
        pauli_term = obs.get_term(i)
        coef = pauli_term.get_coef()
        pauli_id = pauli_term.get_pauli_id_list()
        pauli_index = pauli_term.get_index_list()
        if len(pauli_id) == 0:  # means identity
            exp += coef
            continue
        buf_state.load(state)
        measurement_circuit = QuantumCircuit(n_qubit)
        mask = ''.join(['1' if n_qubit-1-k in pauli_index
                        else '0' for k in range(n_qubit)])
        for single_pauli, index in zip(pauli_id, pauli_index):
            if single_pauli == 1:
                measurement_circuit.add_H_gate(index)
            elif single_pauli == 2:
                measurement_circuit.add_Sdag_gate(index)
                measurement_circuit.add_H_gate(index)
        measurement_circuit.update_quantum_state(buf_state)
        samples = buf_state.sampling(n_sample_list[i])
        mask = int(mask, 2)
        exp += coef*sum(list(map(lambda x: (-1) **
                                     (bin(x & mask).count('1')), samples)))/n_sample_list[i]
    return exp


def test_observable(state, obs, obsZ, n_sample):
    """ Function
    Args:
        state (qulacs.QuantumState): This includes entangled ancilla (n_qubit = n_qubit_system + 1)
        obs (qulacs.Observable): This does not include ancilla Z (n_qubit_system)
        obsZ (qulacs.Observable): Single Pauli Z for ancilla (1)
        poststate0 (qulacs.QuantumState): post-measurement state when ancilla = 0 (n_qubit_system)
        poststate1 (qulacs.QuantumState): post-measurement state when ancilla = 1 (n_qubit_system)
        n_sample (int):  number of samples for each observable
    Return:
        :float: sampled expectation value of the observable

    Author(s): Takashi Tsuchimochi
    """
    n_term = obs.get_term_count()
    n_qubit = obs.get_qubit_count()
    p0 = state.get_zero_probability(n_qubit) 
    p1 = 1-p0
    opt='0'+str(n_qubit)+'b'

    expH =0
    exp = []
    coef = []
    buf_state = QuantumState(n_qubit)
    for i in range(n_term):
        pauli_term = obs.get_term(i)
        coef.append(pauli_term.get_coef().real)
        pauli_id = pauli_term.get_pauli_id_list()
        pauli_index = pauli_term.get_index_list()
        if len(pauli_id) == 0:  # means identity
            exp += coef
            continue
        buf_state.load(state)
        measurement_circuit = QuantumCircuit(n_qubit)
        mask = ''.join(['1' if n_qubit-1-k in pauli_index
                        else '0' for k in range(n_qubit)])

        measure_observable = QubitOperator((),1)
        for single_pauli, index in zip(pauli_id, pauli_index):
            if single_pauli == 1:
                ###  X
                measurement_circuit.add_H_gate(index)
                measure_observable *= QubitOperator('X%d' % index)
            elif single_pauli == 2:
                ###  Y
                measurement_circuit.add_Sdag_gate(index)
                measurement_circuit.add_H_gate(index)
                measure_observable *= QubitOperator('Y%d' % index)
            elif single_pauli == 3:
                ###  Z
                measure_observable *= QubitOperator('Z%d' % index)
        qulacs_measure_observable = create_observable_from_openfermion_text(str(measure_observable))
        measurement_circuit.update_quantum_state(buf_state)
        #exp.append(obsZ.get_expectation_value(buf_state).real)
        samples = buf_state.sampling(n_sample)
        #print('samples? ',format(samples[0],opt))
        #print("I = :",'%5d' % i, "  h_I ", '%10.5f' % coef[i], "    <P_I> ", '%10.5f' % exp[i])
        #mask = int(mask, 2)
        #print(sum(list(map(lambda x: (-1) **(bin(x & mask).count('1')), samples))))
        #print(coef*sum(list(map(lambda x: (-1) **
        #                             (bin(x & mask).count('1')), samples))))
        #expH += coef[i] * exp[i]
        #samples = buf_state.sampling(n_sample)
        

        mask = int(mask, 2)
        prob = sum(list(map(lambda x: (-1) **
                                     (bin(x & mask).count('1')), samples)))/n_sample
        measure_list = list(map(int,np.ones(n_qubit)*2))
        for j in pauli_index:
            measure_list[j] = 1
        print(qulacs_measure_observable.get_expectation_value(state))
        expH += coef[i] * prob
        print("coef: ", '%10.5f' % coef[i],
            "   prob: ", '%10.5f' % prob)
    return expH

def sample_observable_noisy_circuit(circuit, initial_state, obs,
                                    n_circuit_sample=1000,
                                    n_sample_per_circuit=1):
    """ Function
    Args:
        circuit (:class:`qulacs.QuantumCircuit`)
        initial_state (:class:`qulacs.QuantumState`)
        obs (:class:`qulacs.Observable`)
        n_circuit_sample (:class:`int`):  number of circuit samples
        n_sample (:class:`int`):  number of samples per one circuit samples
    Return:
        :float: sampled expectation value of the observable

    Author(s): Unknown
    """
    exp = 0
    state = QuantumState(obs.get_qubit_count())
    for _ in range(n_circuit_sample):
        state.load(initial_state)
        circuit.update_quantum_state(state)
        exp += sample_observable(state,obs,n_sample_per_circuit)
    del state
    exp /= n_circuit_sample
    return exp

def test_transition_observable(state, obs, poststate0, poststate1, n_sample):
    from . import utils
    """
    Args:
        state (qulacs.QuantumState): This includes entangled ancilla (n_qubit = n_qubit_system + 1)
        obs (qulacs.Observable): This does not include ancilla Z (n_qubit_system)
        poststate0 (qulacs.QuantumState): post-measurement state when ancilla = 0 (n_qubit_system)
        poststate1 (qulacs.QuantumState): post-measurement state when ancilla = 1 (n_qubit_system)
        n_sample (int):  number of samples for each observable
    Return:
        :float: sampled expectation value of the observable
    """
    n_term = obs.get_term_count()
    n_qubit = obs.get_qubit_count()
    p0 = state.get_zero_probability(n_qubit-1) 
    p1 = 1-p0
    opt='0'+str(n_qubit)+'b'
    prints('p0:',p0, '  p1:' ,p1)
    prints("post(0)")
    utils.print_state(poststate0,n_qubit)
    prints("post(1)")
    utils.print_state(poststate1,n_qubit)

    expH =0
    exp = []
    coef = []
    buf_state = QuantumState(n_qubit)
    for i in range(n_term):
        pauli_term = obs.get_term(i)
        coef.append(pauli_term.get_coef().real)
        pauli_id = pauli_term.get_pauli_id_list()
        pauli_index = pauli_term.get_index_list()
        if len(pauli_id) == 0:  # means identity
            exp += coef
            continue
        buf_state.load(state)
        measurement_circuit = QuantumCircuit(n_qubit)
        mask = ''.join(['1' if n_qubit-1-k in pauli_index
                        else '0' for k in range(n_qubit)])

        measure_observable = QubitOperator((),1)
        #measure_observable = QubitOperator('Z%d' % n_qubit)
        for single_pauli, index in zip(pauli_id, pauli_index):
            if single_pauli == 1:
                ###  X
                measurement_circuit.add_H_gate(index)
                measure_observable *= QubitOperator('X%d' % index)
            elif single_pauli == 2:
                ###  Y
                measurement_circuit.add_Sdag_gate(index)
                measurement_circuit.add_H_gate(index)
                measure_observable *= QubitOperator('Y%d' % index)
            elif single_pauli == 3:
                ###  Z
                measure_observable *= QubitOperator('Z%d' % index)
        qulacs_measure_observable = create_observable_from_openfermion_text(str(measure_observable))
        ### p0 ###
        H0 = qulacs_measure_observable.get_expectation_value(poststate0)
        ### p1 ###
        H1 = qulacs_measure_observable.get_expectation_value(poststate1)
        prob = p0 * H0 - p1 * H1
        #print(prob, qulacs_measure_observable.get_expectation_value(state), obs.get_expectation_value(state))
        prob= qulacs_measure_observable.get_expectation_value(state)
        expH += coef[i] * prob

         
        #measurement_circuit.update_quantum_state(buf_state)
        #samples = buf_state.sampling(n_sample)
        #print('samples? ',format(samples[0],opt))
        #print("I = :",'%5d' % i, "  h_I ", '%10.5f' % coef[i], "    <P_I> ", '%10.5f' % exp[i])
        #mask = int(mask, 2)
        #print(sum(list(map(lambda x: (-1) **(bin(x & mask).count('1')), samples))))
        #print(coef*sum(list(map(lambda x: (-1) **
        #                             (bin(x & mask).count('1')), samples))))
        #expH += coef[i] * exp[i]
        #samples = buf_state.sampling(n_sample)
        

        #mask = int(mask, 2)
        #prob = sum(list(map(lambda x: (-1) **
        #                             (bin(x & mask).count('1')), samples)))/n_sample
        #measure_list = list(map(int,np.ones(n_qubit)*2))
        #for j in pauli_index:
        #    measure_list[j] = 1
        #print(qulacs_measure_observable.get_expectation_value(state))
        #expH += coef[i] * prob
        print("coef: ", '%10.5f' % coef[i],
            "   prob: ", '%10.5f' % prob)
    return expH

def sample_observable_noisy_circuit(circuit, initial_state, obs,
                                    n_circuit_sample=1000,
                                    n_sample_per_circuit=1):
    """
    Args:
        circuit (:class:`qulacs.QuantumCircuit`)
        initial_state (:class:`qulacs.QuantumState`)
        obs (:class:`qulacs.Observable`)
        n_circuit_sample (:class:`int`):  number of circuit samples
        n_sample (:class:`int`):  number of samples per one circuit samples
    Return:
        :float: sampled expectation value of the observable

    Author(s): Unknown
    """
    exp = 0
    state = QuantumState(obs.get_qubit_count())
    for _ in range(n_circuit_sample):
        state.load(initial_state)
        circuit.update_quantum_state(state)
        exp += sample_observable(state,obs,n_sample_per_circuit)
    del state
    exp /= n_circuit_sample
    return exp

def cost_phf_sample(print_level,n_qubit,n_electron,noa,nob,nva,nvb,rho,anc,qulacs_hamiltonian,qulacs_hamiltonianZ,qulacs_s2Z,qulacs_ancZ,coef0_H,coef0_S2,ref,theta_list,samplelist):
    """ Function:
    Sample Hamiltonian and S**2 expectation values with PHF and PUCCSD.
    Write out the statistics in csv files.

    Author(s): Takashi Tsuchimochi
    """
    from .phflib import set_circuit_rhfZ, set_circuit_uhfZ, controlled_Ug
    from .ucclib import set_circuit_uccsd
    from . import utils
    import csv
    import pprint
    t1 = time.time()
    n_qubit_system = n_qubit - 1
    state = QuantumState(n_qubit)
    circuit_rhf = set_circuit_rhfZ(n_qubit,n_electron)
    circuit_rhf.update_quantum_state(state)
    if(ref == "phf"):
        circuit_uhf = set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,theta_list)
        circuit_uhf.update_quantum_state(state)
        print("pHF")
    elif(ref == "puccsd"):
        circuit = set_circuit_uccsd(n_qubit,noa,nob,nva,nvb,theta_list)
        for i in range(rho):
            circuit.update_quantum_state(state)
        print("UCCSD")

    if(print_level > -1):
        print('State before projection')
        utils.print_state(state,n_qubit_system)
    '''
    ### Set post-measurement states ####
    poststate0 = state.copy()
    poststate1 = state.copy()
    circuit0 = QuantumCircuit(n_qubit)
    circuit1 = QuantumCircuit(n_qubit)
    ### Projection to anc = 0 or anc = 1 ###
    circuit0.add_gate(P0(0))
    circuit1.add_gate(P1(0))
    circuit0.update_quantum_state(poststate0)
    circuit1.update_quantum_state(poststate1)
    ### Renormalize each state ###        
    norm0 = poststate0.get_squared_norm()
    norm1 = poststate1.get_squared_norm()
    poststate0.normalize(norm0)
    poststate1.normalize(norm1)
    '''
    
#    '''
    ### grid loop ###
    Ng = 4
    beta = ([-0.861136311594053,-0.339981043584856,0.339981043584856,0.861136311594053])
    wg   = ([0.173927422568724,0.326072577431273,0.326072577431273,0.173927422568724])
    Ng = 2
    beta = ([0.577350269189626,-0.577350269189626])
    wg   = ([0.5, 0.5])
    ### a list to compute the probability to observe 0 in ancilla qubit
    p0_list = []
    for j in range(n_qubit-1):
        p0_list.append(2)
    p0_list.append(0)
    ### Array for <HUg>, <S2Ug>, <Ug>
    #samplelist = [10,100,1000,10000,100000,1000000,10000000]
    ncyc = 4
    prints("",filepath='./log2.txt')
    for i_sample in samplelist:
        i_sample_x = i_sample
        if(i_sample == 10000000):
            print("OK")
            ncyc = ncyc*10
            i_sample_x = 1000000
        sampleHUg1 = []
        sampleHUg2 = []
        sampleHUg3 = []
        sampleHUg4 = []
        sampleS2Ug1 = []
        sampleS2Ug2 = []
        sampleS2Ug3 = []
        sampleS2Ug4 = []
        sampleUg1 = []
        sampleUg2 = []
        sampleUg3 = []
        sampleUg4 = []
        #sampleEn = []
        #sampleS2 = []
        sampleHUg = np.zeros((ncyc,Ng))
        sampleS2Ug = np.zeros((ncyc,Ng))
        sampleUg = np.zeros((ncyc,Ng))
        sampleEn = np.zeros((ncyc,1))
        sampleS2 = np.zeros((ncyc,1))
        for icyc in range(ncyc):
            prints("n_sample : ",i_sample_x, "(", '%3d' % icyc, "/", ncyc,")",filepath='./log2.txt')
            HUg = []
            S2Ug = []
            Ug = []
            Ephf = 0
            S2 = 0
            Norm = 0
            for i in range(Ng):
                ### Copy quantum state of UHF (cannot be done in real device) ###
                state_g = QuantumState(n_qubit)
                state_g.load(state)
                ### Construct Ug test
                circuit_ug = QuantumCircuit(n_qubit)
                ### Hadamard on anc
                circuit_ug.add_H_gate(anc)
                controlled_Ug(circuit_ug,n_qubit,anc,np.arccos(beta[i]))
                circuit_ug.add_H_gate(anc)       
                circuit_ug.update_quantum_state(state_g)

                ### Set post-measurement states ####
                poststate0 = state_g.copy()
                poststate1 = state_g.copy()
                circuit0 = QuantumCircuit(n_qubit)
                circuit1 = QuantumCircuit(n_qubit)
                ### Projection to anc = 0 or anc = 1 ###
                circuit0.add_gate(P0(anc))
                circuit1.add_gate(P1(anc))
                circuit0.update_quantum_state(poststate0)
                circuit1.update_quantum_state(poststate1)
                ### Renormalize each state ###        
                norm0 = poststate0.get_squared_norm()
                norm1 = poststate1.get_squared_norm()
                poststate0.normalize(norm0)
                poststate1.normalize(norm1)
                ### Set ancilla qubit of poststate1 to zero (so that it won't be used) ###
                circuit_anc = QuantumCircuit(n_qubit)
                circuit_anc.add_X_gate(anc)
                circuit_anc.update_quantum_state(poststate1)
                print(test_transition_observable(state_g, qulacs_hamiltonianZ, poststate0, poststate1, 100000))
                #exit()



        
        
                ### Probabilities for getting 0 and 1 in ancilla qubit ###
                p0 = state_g.get_marginal_probability(p0_list)
                p1 = 1 - p0        
                
                ### Compute expectation value <HUg> ###
                HUg.append(sample_observable(state_g, qulacs_hamiltonianZ, i_sample_x).real)
#                HUg.append(adaptive_sample_observable(state_g, qulacs_hamiltonianZ, i_sample_x).real)
                ### <S2Ug> ###
                S2Ug.append(sample_observable(state_g, qulacs_s2Z, i_sample_x).real)
#                S2Ug.append(adaptive_sample_observable(state_g, qulacs_s2Z, i_sample_x).real)
                #S2Ug.append(qulacs_s2Z.get_expectation_value(state_g))
#                HUg.append(0)
                #S2Ug.append(0)
        
        #        Ug.append(p0 - p1)
                n_term = qulacs_hamiltonianZ.get_term_count()
                n_sample_total = i_sample_x * n_term
                # in the worst-case scenario, Ug is measured as many times as n_sample_total (required to evaluate HUg)
                Ug.append(sample_observable(state_g, qulacs_ancZ, i_sample_x*n_term).real)
                #p0_sample = 0
                #for j_sample in range(n_sample_total):
                #    if(p0 > np.random.rand()):
                #       p0_sample += 1 
                #Ug.append(2*p0_sample/n_sample_total - 1)
                ### Norm accumulation ###
                Norm += wg[i] * Ug[i]
                sampleHUg[icyc,i] = HUg[i]
                sampleS2Ug[icyc,i] = S2Ug[i]
                sampleUg[icyc,i] = Ug[i]
    #        print('p0 : ',p0,'  p1 : ',p1,  '  p0 - p1 : ',p0-p1)
    #    '''
            
            sampleHUg1.append(HUg[0])
            sampleHUg2.append(HUg[1])
            #sampleHUg3.append(HUg[2])
            #sampleHUg4.append(HUg[3])
            sampleS2Ug1.append(S2Ug[0])
            sampleS2Ug2.append(S2Ug[1])
            #sampleS2Ug3.append(S2Ug[2])
            #sampleS2Ug4.append(S2Ug[3])
            sampleUg1.append(Ug[0])
            sampleUg2.append(Ug[1])
            #sampleUg3.append(Ug[2])
            #sampleUg4.append(Ug[3])
        ### Energy calculation <HP>/<P> and <S**2P>/<P> ###
            Ephf = 0
            for i in range(Ng):
                Ephf += wg[i] * HUg[i] / Norm
                S2 += wg[i] * S2Ug[i] / Norm
#                print(" <S**2> = ", S2, '\n')    
            Ephf += coef0_H
            S2 += coef0_S2
            sampleEn[icyc,0] = Ephf
            sampleS2[icyc,0] = S2
            #print(" <E[PHF]> (Nsample = ",i_sample,") = ", Ephf)
#        print("(n_sample = ", i_sample, ") :  sample HUg1\n",sampleHUg1)
#        print("(n_sample = ", i_sample, ") :  sample HUg2\n",sampleHUg2)
#        print("(n_sample = ", i_sample, ") :  sample HUg3\n",sampleHUg3)
#        print("(n_sample = ", i_sample, ") :  sample HUg4\n",sampleHUg4)
#        print("(n_sample = ", i_sample, ") :  sample S2Ug1\n",sampleS2Ug1)
#        print("(n_sample = ", i_sample, ") :  sample S2Ug2\n",sampleS2Ug2)
#        print("(n_sample = ", i_sample, ") :  sample S2Ug3\n",sampleS2Ug3)
#        print("(n_sample = ", i_sample, ") :  sample S2Ug4\n",sampleS2Ug4)
#        print("(n_sample = ", i_sample, ") :  sample Ug1\n",sampleUg1)
#        print("(n_sample = ", i_sample, ") :  sample Ug2\n",sampleUg2)
#        print("(n_sample = ", i_sample, ") :  sample Ug3\n",sampleUg3)
#        print("(n_sample = ", i_sample, ") :  sample Ug4\n",sampleUg4)
#        print("(n_sample = ", i_sample, ") :  sample HUg1\n",sampleHUg1)
#        print("(n_sample = ", i_sample, ") :  sample HUg2\n",sampleHUg2)
#        print("(n_sample = ", i_sample, ") :  sample HUg3\n",sampleHUg3)
#        print("(n_sample = ", i_sample, ") :  sample HUg4\n",sampleHUg4)
#        print("(n_sample = ", i_sample, ") :  sample En\n",sampleEn)
#        print("(n_sample = ", i_sample, ") :  sample S2\n",sampleS2)
        with open('./Ug_%d.csv' % i_sample, 'w') as fUg:
            writer = csv.writer(fUg)
            writer.writerows(sampleUg)
        with open('./HUg_%d.csv' % i_sample, 'w') as fHUg:
            writer = csv.writer(fHUg)
            writer.writerows(sampleHUg)
        with open('./S2Ug_%d.csv' % i_sample, 'w') as fS2Ug:
            writer = csv.writer(fS2Ug)
            writer.writerows(sampleS2Ug)
        with open('./En_%d.csv' % i_sample, 'w') as fEn:
            writer = csv.writer(fEn)
            writer.writerows(sampleEn)
        with open('./S2_%d.csv' % i_sample, 'w') as fS2:
            writer = csv.writer(fS2)
            writer.writerows(sampleS2)
    return Ephf, S2
    
    
def cost_uhf_sample(print_level,n_qubit_system,n_electron,noa,nob,nva,nvb,qulacs_hamiltonian,qulacs_s2,kappa_list,samplelist):
    """ Function:
    Sample Hamiltonian and S**2 expectation values with UHF.
    Write out the statistics in csv files.

    Author(s): Takashi Tsuchimochi
    """
    import csv
    from .hflib import set_circuit_rhf, set_circuit_uhf
    ncyc = 13
    prints("",filepath='./log.txt',opentype='w')
    for i_sample in samplelist:
        sampleEn = np.zeros((ncyc,1))
        sampleS2 = np.zeros((ncyc,1))
        for icyc in range(ncyc):
            prints("n_sample : ",i_sample, "(", '%3d' % icyc, "/", ncyc,")", filepath='./log.txt')
            opt='0'+str(n_qubit_system)+'b'
            state = QuantumState(n_qubit_system)
            circuit_rhf = set_circuit_rhf(n_qubit_system,n_electron)
            circuit_rhf.update_quantum_state(state)
            circuit = set_circuit_uhf(n_qubit_system,noa,nob,nva,nvb,kappa_list)
            circuit.update_quantum_state(state)
            Euhf = sample_observable(state, qulacs_hamiltonian, i_sample).real
            #S2 = sample_observable(state, qulacs_s2, i_sample).real
            #Euhf = adaptive_sample_observable(state, qulacs_hamiltonian, i_sample).real
            #S2 = adaptive_sample_observable(state, qulacs_s2, i_sample).real
            sampleEn[icyc,0] = Euhf
            #sampleS2[icyc,0] = S2
            S2 = 0
        with open('./UEn_%d.csv' % i_sample, 'w') as fEn:
            writer = csv.writer(fEn)
            writer.writerows(sampleEn)
#        with open('./US2_%d.csv' % i_sample, 'w') as fS2:
#            writer = csv.writer(fS2)
#            writer.writerows(sampleS2)
    return Euhf, S2





def cost_phf_sample_oneshot(print_level,n_qubit,n_electron,noa,nob,nva,nvb,anc,qulacs_hamiltonianZ,qulacs_s2Z,qulacs_ancZ,coef0_H,coef0_S2,kappa_list):
    """ Function:
    Test function for sampling Hamiltonian and S** expectation values with PHF just for once. 

    Author(s): Takashi Tsuchimochi
    """
    import csv
    from .phflib import set_circuit_rhfZ, set_circuit_uhfZ, controlled_Ug
    import csv
    import pprint
    t1 = time.time()
    state = QuantumState(n_qubit)
    circuit_rhf = set_circuit_rhfZ(n_qubit,n_electron)
    circuit_rhf.update_quantum_state(state)
    circuit_uhf = set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,kappa_list)
    circuit_uhf.update_quantum_state(state)
    
    ### Set post-measurement states ####
    poststate0 = state.copy()
    poststate1 = state.copy()
    circuit0 = QuantumCircuit(n_qubit)
    circuit1 = QuantumCircuit(n_qubit)
    ### Projection to anc = 0 or anc = 1 ###
    circuit0.add_gate(P0(0))
    circuit1.add_gate(P1(0))
    circuit0.update_quantum_state(poststate0)
    circuit1.update_quantum_state(poststate1)
    ### Renormalize each state ###        
    norm0 = poststate0.get_squared_norm()
    norm1 = poststate1.get_squared_norm()
    poststate0.normalize(norm0)
    poststate1.normalize(norm1)

    
#    '''
    ### grid loop ###
    Ng = 4
    beta = ([-0.861136311594053,-0.339981043584856,0.339981043584856,0.861136311594053])
    wg   = ([0.173927422568724,0.326072577431273,0.326072577431273,0.173927422568724])
    ### a list to compute the probability to observe 0 in ancilla qubit
    p0_list = []
    for j in range(n_qubit-1):
        p0_list.append(2)
    p0_list.append(0)
    ### Array for <HUg>, <S2Ug>, <Ug>
    samplelist = [5,50,500,5000,50000,500000,5000000]
    Ng = 4
    ncyc = 10
    prints("",filepath='./log.txt',opentype='w')
    for i_sample in samplelist:
        sampleEn = []
        sampleS2 = []
        for icyc in range(ncyc):
            prints("n_sample : ",i_sample, "(", '%3d' % icyc, "/", ncyc,")", filepath='./log.txt')
            HUg = []
            S2Ug = []
            Ug = []
            Ephf = 0
            S2 = 0
            Norm = 0
            for i in range(Ng):
                ### Copy quantum state of UHF (cannot be done in real device) ###
                state_g = QuantumState(n_qubit)
                circuit_rhf.update_quantum_state(state_g)
                circuit_uhf.update_quantum_state(state_g)
                ### Construct Ug test
                circuit_ug = QuantumCircuit(n_qubit)
                ### Hadamard on anc
                circuit_ug.add_H_gate(anc)
                controlled_Ug(circuit_ug,n_qubit,anc,np.arccos(beta[i]))
                circuit_ug.add_H_gate(anc)       
                circuit_ug.update_quantum_state(state_g)
        
        
                ### Probabilities for getting 0 and 1 in ancilla qubit ###
                p0 = state_g.get_marginal_probability(p0_list)
                p1 = 1 - p0        
                
                ### Compute expectation value <HUg> ###
                HUg.append(sample_observable(state_g, qulacs_hamiltonianZ, i_sample).real)
                ### <S2Ug> ###
                S2Ug.append(sample_observable(state_g, qulacs_s2Z, i_sample).real)
                #S2Ug.append(qulacs_s2Z.get_expectation_value(state_g))
        
        #        Ug.append(p0 - p1)
                Ug.append(sample_observable(state_g, qulacs_ancZ, i_sample).real)
                ### Norm accumulation ###
                Norm += wg[i] * Ug[i]
                sampleHUg[icyc,i] = HUg[i]
                sampleS2Ug[icyc,i] = S2Ug[i]
                sampleUg[icyc,i] = Ug[i]
    #        print('p0 : ',p0,'  p1 : ',p1,  '  p0 - p1 : ',p0-p1)
    #    '''
            
            sampleHUg1.append(HUg[0])
            sampleHUg2.append(HUg[1])
            sampleHUg3.append(HUg[2])
            sampleHUg4.append(HUg[3])
            sampleS2Ug1.append(S2Ug[0])
            sampleS2Ug2.append(S2Ug[1])
            sampleS2Ug3.append(S2Ug[2])
            SAMpleS2Ug4.append(S2Ug[3])
            sampleUg1.append(Ug[0])
            sampleUg2.append(Ug[1])
            sampleUg3.append(Ug[2])
            sampleUg4.append(Ug[3])
        ### Energy calculation <HP>/<P> and <S**2P>/<P> ###
            Ephf = 0
            for i in range(Ng):
                Ephf += wg[i] * HUg[i] / Norm
                S2 += wg[i] * S2Ug[i] / Norm
#                print(" <E[PHF]> (Nsample = ",i_sample,") = ", Ephf)
#                print(" <S**2> = ", S2, '\n')    
            Ephf += coef0_H
            S2 += coef0_S2
            sampleEn[icyc,0] = Ephf
            sampleS2[icyc,0] = S2
#        print("(n_sample = ", i_sample, ") :  sample HUg1\n",sampleHUg1)
#        print("(n_sample = ", i_sample, ") :  sample HUg2\n",sampleHUg2)
#        print("(n_sample = ", i_sample, ") :  sample HUg3\n",sampleHUg3)
#        print("(n_sample = ", i_sample, ") :  sample HUg4\n",sampleHUg4)
#        print("(n_sample = ", i_sample, ") :  sample S2Ug1\n",sampleS2Ug1)
#        print("(n_sample = ", i_sample, ") :  sample S2Ug2\n",sampleS2Ug2)
#        print("(n_sample = ", i_sample, ") :  sample S2Ug3\n",sampleS2Ug3)
#        print("(n_sample = ", i_sample, ") :  sample S2Ug4\n",sampleS2Ug4)
#        print("(n_sample = ", i_sample, ") :  sample Ug1\n",sampleUg1)
#        print("(n_sample = ", i_sample, ") :  sample Ug2\n",sampleUg2)
#        print("(n_sample = ", i_sample, ") :  sample Ug3\n",sampleUg3)
#        print("(n_sample = ", i_sample, ") :  sample Ug4\n",sampleUg4)
#        print("(n_sample = ", i_sample, ") :  sample HUg1\n",sampleHUg1)
#        print("(n_sample = ", i_sample, ") :  sample HUg2\n",sampleHUg2)
#        print("(n_sample = ", i_sample, ") :  sample HUg3\n",sampleHUg3)
#        print("(n_sample = ", i_sample, ") :  sample HUg4\n",sampleHUg4)
#        print("(n_sample = ", i_sample, ") :  sample En\n",sampleEn)
#        print("(n_sample = ", i_sample, ") :  sample S2\n",sampleS2)
        with open('./HUg_%d.csv' % i_sample, 'w') as fHUg:
            writer = csv.writer(fHUg)
            writer.writerows(sampleHUg)
        with open('./S2Ug_%d.csv' % i_sample, 'w') as fS2Ug:
            writer = csv.writer(fS2Ug)
            writer.writerows(sampleS2Ug)
        with open('./Ug_%d.csv' % i_sample, 'w') as fUg:
            writer = csv.writer(fUg)
            writer.writerows(sampleUg)
        with open('./En_%d.csv' % i_sample, 'w') as fEn:
            writer = csv.writer(fEn)
            writer.writerows(sampleEn)
        with open('./S2_%d.csv' % i_sample, 'w') as fS2:
            writer = csv.writer(fS2)
            writer.writerows(sampleS2)

    return Ephf, S2
    
