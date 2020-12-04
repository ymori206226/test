import sys
import numpy as np
import time
from qulacs.observable import create_observable_from_openfermion_text
from openfermion.ops import QubitOperator
from qulacs import QuantumState
from qulacs import QuantumCircuit
from . import config as cf
from . import mpilib as mpi
from .ucclib import ucc_singles, set_circuit_uccsd, set_circuit_uccd, set_circuit_sauccd
from .utils     import SaveTheta, print_state, print_amplitudes


def set_circuit_rhfZ(n_qubit,n_electron):
    """ Function:
    Construct circuit for RHF |0000...1111> with one ancilla
    """
    circuit = QuantumCircuit(n_qubit)
    for i in range(n_electron):
        circuit.add_X_gate(i)
    return circuit

def set_circuit_rohfZ(n_qubit,noa,nob):
    """ Function:
    Construct circuit for ROHF |0000...10101111> with one ancilla
    """
# generate circuit for rhf
    circuit = QuantumCircuit(n_qubit)
    for i in range(noa):
        circuit.add_X_gate(2*i)
    for i in range(nob):
        circuit.add_X_gate(2*i+1)
    return circuit

def set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,theta_list):
    """ Function:
    Construct circuit for UHF with one ancilla
    """
    circuit = QuantumCircuit(n_qubit)
    ucc_singles(circuit,noa,nob,nva,nvb,theta_list)
    return circuit

def set_circuit_Ug(circuit,n_qubit_system,beta):
    """ Function:
    Construct circuit for Ug in spin-projection 
    """
    ### Ug
    for i in range(n_qubit_system):
        if i%2==0:
            #
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i+1,np.pi/2)
            circuit.add_CNOT_gate(i+1,i)
            circuit.add_RZ_gate(i,-beta/2)
            circuit.add_CNOT_gate(i+1,i)
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i+1,-np.pi/2)            
            ###
            circuit.add_H_gate(i+1)
            circuit.add_RX_gate(i,np.pi/2)
            circuit.add_CNOT_gate(i+1,i)
            circuit.add_RZ_gate(i,beta/2)
            circuit.add_CNOT_gate(i+1,i)
            circuit.add_H_gate(i+1)
            circuit.add_RX_gate(i,-np.pi/2)    


def controlled_Ug(circuit,n_qubit,anc,beta):
    """ Function:
    Construct circuit for controlled-Ug in spin-projection 
    """
    ### Controlled Ug
    for i in range(n_qubit - 1):
        if i%2==0:
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i+1,np.pi/2)
            circuit.add_CNOT_gate(i+1,i)
            circuit.add_RZ_gate(i,-beta/4)
            circuit.add_CNOT_gate(anc,i)
            circuit.add_RZ_gate(i,beta/4)
            circuit.add_CNOT_gate(anc,i)
            circuit.add_CNOT_gate(i+1,i)
            circuit.add_H_gate(i)
            circuit.add_RX_gate(i+1,-np.pi/2)            
            ###
            circuit.add_H_gate(i+1)
            circuit.add_RX_gate(i,np.pi/2)
            circuit.add_CNOT_gate(i+1,i)
            circuit.add_RZ_gate(i,beta/4)
            circuit.add_CNOT_gate(anc,i)
            circuit.add_RZ_gate(i,-beta/4)
            circuit.add_CNOT_gate(anc,i)
            circuit.add_CNOT_gate(i+1,i)
            circuit.add_H_gate(i+1)
            circuit.add_RX_gate(i,-np.pi/2)    



def cost_proj(print_level,n_qubit,n_electron,noa,nob,nva,nvb,rho,DS,anc,qulacs_hamiltonianZ,qulacs_s2Z,coef0_H,coef0_S2,ref,kappa_list,theta_list=0,threshold=0.01):
    """ Function:
    Energy functional for projected methods (phf, puccsd, puccd, opt_puccd)
    """
    t1 = time.time()
    ndim1 = noa*nva + nob*nvb
    ndim2aa = int(noa*(noa-1)*nva*(nva-1)/4)  
    ndim2ab = int(noa*nob*nva*nvb)  
    ndim2bb = int(nob*(nob-1)*nvb*(nvb-1)/4)  
    ndim2 = ndim2aa + ndim2ab + ndim2bb
    n_qubit_system = n_qubit - 1
    state = QuantumState(n_qubit)
    if(noa==nob):
        circuit_rhf = set_circuit_rhfZ(n_qubit,n_electron)
    else:
        circuit_rhf = set_circuit_rohfZ(n_qubit,noa,nob)
    circuit_rhf.update_quantum_state(state)
    if(ref == "phf"):
        circuit_uhf = set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,kappa_list)
        circuit_uhf.update_quantum_state(state)
        if(print_level > 0):
            SaveTheta(ndim1,kappa_list,cf.tmp)
    elif(ref == "puccsd"):
        # First prepare UHF determinant
        circuit_uhf = set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,kappa_list)
        circuit_uhf.update_quantum_state(state)
        # Then prepare UCCSD 
        theta_list_rho = theta_list/rho
        circuit = set_circuit_uccsd(n_qubit,noa,nob,nva,nvb,theta_list_rho)
        for i in range(rho):
            circuit.update_quantum_state(state)
        if(print_level > 0):
            SaveTheta(ndim1+ndim2,theta_list,cf.tmp)
    elif(ref == "puccd"):
        # First prepare UHF determinant
        circuit_uhf = set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,kappa_list)
        circuit_uhf.update_quantum_state(state)
        # Then prepare UCCD 
        theta_list_rho = theta_list/rho
        circuit = set_circuit_uccd(n_qubit,noa,nob,nva,nvb,theta_list_rho)
        for i in range(rho):
            circuit.update_quantum_state(state)
        if(print_level > 0):
            SaveTheta(ndim2,theta_list,cf.tmp)
    elif(ref == "opt_puccd"):
        if DS:
            # First prepare UHF determinant
            circuit_uhf = set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,theta_list)
            circuit_uhf.update_quantum_state(state)
            # Then prepare UCCD 
            theta_list_rho = theta_list[ndim1:ndim1+ndim2]/rho
            circuit = set_circuit_uccd(n_qubit,noa,nob,nva,nvb,theta_list_rho)
            for i in range(rho):
                circuit.update_quantum_state(state)
        else:
            # First prepare UCCD
            theta_list_rho = theta_list[ndim1:ndim1+ndim2]/rho
            circuit = set_circuit_uccd(n_qubit,noa,nob,nva,nvb,theta_list_rho)
            for i in range(rho):
                circuit.update_quantum_state(state)
            # then rotate
            circuit_uhf = set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,theta_list)
            circuit_uhf.update_quantum_state(state)

        if(print_level > 0):
            SaveTheta(ndim1+ndim2,theta_list,cf.tmp)
    elif(ref == "opt_psauccd"):
        theta_list_rho = theta_list[ndim1:ndim1+ndim2]/rho
        circuit = set_circuit_sauccd(n_qubit,noa,nva,theta_list_rho)
        for i in range(rho):
            circuit.update_quantum_state(state)
        circuit_uhf = set_circuit_uhfZ(n_qubit,noa,nob,nva,nvb,theta_list)
        circuit_uhf.update_quantum_state(state)
    if(print_level > 1):
        if mpi.mpi.main_rank:
            with open(cf.log,'a') as f:
                print('State before projection',file=f)
            print_state(state,n_qubit_system)
        if(ref == "puccsd" or ref == "opt_puccd"):
            print_amplitudes(theta_list,noa,nob,nva,nvb,threshold)


#    '''
    ### grid loop ###
    if cf.ng == 4 :
        beta = ([0.861136311594053,0.339981043584856,-0.339981043584856,-0.861136311594053])
        wg   = ([0.173927422568724,0.326072577431273,0.326072577431273,0.173927422568724])
    elif cf.ng == 3:
        beta = ([0.774596669241483, 0, -0.774596669241483]) 
        if cf.spin == 1:
            wg   = ([0.277777777777776, 0.444444444444444, 0.277777777777776]) 
        elif cf.spin ==3:
            if noa - nob == 2:
                wg   = ([0.739415278850614, 0.666666666666667, 0.09391805448271476])
            else:
                wg   = ([0.645497224367899,0, -0.645497224367899]) 
    elif cf.ng == 2:
        beta = ([0.577350269189626,-0.577350269189626])
        if cf.spin == 1:
            wg   = ([0.5, 0.5])
        elif cf.spin ==3:
            if noa - nob == 2:
                wg = ([1.1830127018922187,0.3169872981077805])
            else:
                wg   = ([0.866025404, -0.866025404])
    ### a list to compute the probability to observe 0 in ancilla qubit
    ### Array for <HUg>, <S2Ug>, <Ug>
    HUg = []
    S2Ug = []
    Ug = []
    S2 = 0
    Norm = 0
    for i in range(cf.ng):
        ### Copy quantum state of UHF (cannot be done in real device) ###
        state_g = QuantumState(n_qubit)
        state_g.load(state)
        ### Construct Ug test
        circuit_ug = QuantumCircuit(n_qubit)
        ### Hadamard on anc
        circuit_ug.add_H_gate(anc)
        #circuit_ug.add_X_gate(anc)
        controlled_Ug(circuit_ug,n_qubit,anc,np.arccos(beta[i]))
        #circuit_ug.add_X_gate(anc)
        circuit_ug.add_H_gate(anc)       
        circuit_ug.update_quantum_state(state_g)

        ### Compute expectation value <HUg> ###
        HUg.append(qulacs_hamiltonianZ.get_expectation_value(state_g))
        ### <S2Ug> ###
        S2Ug.append(qulacs_s2Z.get_expectation_value(state_g))
        ### <Ug> ###
        p0 = state_g.get_zero_probability(anc)
        p1 = 1 - p0
        Ug.append(p0 - p1)
        ### Norm accumulation ###
        Norm += wg[i] * Ug[i]
#        print('p0 : ',p0,'  p1 : ',p1,  '  p0 - p1 : ',p0-p1)
#    '''
#    print("Time: ",t2-t1)
    ### Energy calculation <HP>/<P> and <S**2P>/<P> ###
    Ep = 0
    for i in range(cf.ng):
        Ep += wg[i] * HUg[i] / Norm
        S2 += wg[i] * S2Ug[i] / Norm
    t2 = time.time() 
    cpu1 = t2 - t1
    Ep += coef0_H
    S2 += coef0_S2
    if print_level == -1 and mpi.main_rank:
        with open(cf.log,'a') as f:
            print(" Initial E[%s] = " % ref, '{:.12f}'.format(Ep),  "  <S**2> =", '%2.15f' % S2, "rho = %d" % rho, file=f)
    if print_level == 1 and mpi.main_rank:
        cput = t2 - cf.t_old 
        cf.t_old = t2
        with open(cf.log,'a') as f:
            print(" E[%s] = " % ref, '{:.12f}'.format(Ep),  "  <S**2> =", '%2.15f' % S2, "  CPU Time = ", '%2.5f' % cput, " (%2.5f / step)" % cpu1, file=f)
    if print_level > 1 and mpi.main_rank:
        with open(cf.log,'a') as f:
            print(" Final E[%s] = " % ref, '{:.12f}'.format(Ep),  "  <S**2> =", '%2.15f' % S2, "rho = %d" % rho, file=f)
        print_state(state,n_qubit-1)
        if(ref == "puccsd" or ref == "opt_puccd"):
            print_amplitudes(theta_list,noa,nob,nva,nvb)
        with open(cf.log,'a') as f:
            print("HUg", HUg,file=f)
            print("Ug",Ug,file=f)

    return Ep, S2


def S2Proj(Q,spin=1,Ms=0,ng=4):
    '''
       Perform spin-projection to QuantumState |Q> 

          |Q'>  =  Ps |Q>   

       where Ps is a spin-projection operator (non-unitary).
       
          Ps = \sum_i^ng   wg[i] Ug[i] 

       This function provides a shortcut to |Q'>, which is unreal.
       One actually needs to develop a quantum circuit for this (See PRR 2, 043142 (2020)).

    '''
    ### Set grid points ###
    if ng == 4 :
        beta = ([0.861136311594053,0.339981043584856,-0.339981043584856,-0.861136311594053])
        if spin == 1:
            wg   = ([0.173927422568724,0.326072577431273,0.326072577431273,0.173927422568724])
        elif spin == 3:
            if Ms == 2:
                wg   = ([0.485553962586922,0.655396608886142,0.322821123407678,0.03622830511924918])
            elif Ms == 0:
                wg   = ([ 0.449325657467673, 0.332575485478464,  -0.332575485478464 ,- 0.449325657467673])
    elif ng == 3:
        beta = ([0.774596669241483, 0, -0.774596669241483])
        if spin == 1:
            wg   = ([0.277777777777776, 0.444444444444444, 0.277777777777776])
        elif spin ==3:
            if Ms == 2:
                wg   = ([0.739415278850614, 0.666666666666667, 0.09391805448271476])
            elif Ms == 0:
                wg   = ([0.645497224367899,0, -0.645497224367899])
    elif ng == 2:
        beta = ([0.577350269189626,-0.577350269189626])
        if spin == 1:
            wg   = ([0.5, 0.5])
        elif spin ==3:
            if Ms == 2:
                wg = ([1.1830127018922187,0.3169872981077805])
            elif Ms == 0:
                wg   = ([0.866025404, -0.866025404])
    if beta == None:
        error(" Error in S2Proj. Check ng. ")
    if wg == None:
        error(" Error in S2Proj. Check spin and Ms. ")

    n_qubit = Q.get_count_qubits()
    state_P = QuantumState(n_qubit)
    state_P.multiply_coef(0)
    for i in range(ng):
        ### Copy quantum state of UHF (cannot be done in real device) ###
        state_g = QuantumState(n_qubit)
        state_g.load(Q)
        ### Construct Ug circuit
        circuit_ug = QuantumCircuit(n_qubit)
        set_circuit_Ug(circuit_ug,n_qubit,np.arccos(beta[i]))
        circuit_ug.update_quantum_state(state_g)
        state_g.multiply_coef(wg[i])
        state_P.add_state(state_g)
    return state_P

