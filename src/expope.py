"""
#######################
#        quket        #
#######################

expope.py

Functions to prepare several types of rotations, including singles and doubles rotations.

"""

from . import config as cf

def Gdouble_ope(p,q,r,s,circuit,theta):
    """ Function:
    Construct exp[theta ( p!q!rs - s!r!qp ) ] as a whole unitary and add to circuit 
    Here, max(p,q,r,s) = p is assumed.
    There are 5 cases:
        (1) p > q > r > s (includes p > q > s > r)
        (2) p > r > q > s (includes p > r > s > q)
        (3) p = r > q > s (includes p = r > s > q)
        (4) p > q = r > s (includes p > q = s > r)
        (5) p > r > q = s (includes p > s > q = r)
        
    Note that in Qulacs, the function "PauliRotation" rotates as
                 Exp [i theta/2 ...]
    so theta is divided by two. Accordingly, we need to multiply theta by two.
    

    Args:
        p (int): excitation index 
        q (int): excitation index 
        r (int): excitation index 
        s (int): excitation index 
        circuit (QuantumCircuit): circuit to be updated
        theta (float): real rotation parameter

    Returns:
        circuit (QuantumCircuit): circuit to be updated

    Author(s): Takashi Tsuchimochi
    """

    if p != max(p,q,r,s):
        print("Error!    p != max(p,q,r,s)")
        return
    if (p == q or r == s):
        print("Caution:    p == q   or  r == s")
        return

    if p == r:
        if q > s:
            # p^ q^ p s    ( p  > q,    p > s)
            Gdoubles_pqps(p,q,s,circuit,theta)
        elif q < s:
            # p^ q^ p s   =  p^ s^ p q   ( p  > s,    p > q)    
            Gdoubles_pqps(p,s,q,circuit,theta)       
        else:
            print("Error!    p^r^ pr  - h.c.  =  zero")
    elif p == s:  #(necessarily  r < s)
        Gdoubles_pqps(p,q,r,circuit,-theta)
    elif q == r:
        if q > s:
            # p^ q^ q s     (p > q,   q > s)
            Gdoubles_pqqs(p,q,s,circuit,theta)
        elif q < s:
            # p^ q^ q s  = - p^ q^ s q     (p > q,  s > q)
            Gdoubles_pqrq(p,q,s,circuit,-theta)
            
    elif q == s:
        if q < r:
            # p^ q^ r q     (p > q,   r > q)
            Gdoubles_pqrq(p,q,r,circuit,theta)
        elif q > r:
            # p^ q^ r q  =  - p^ q^ q r     (p > q,   q > r)
            Gdoubles_pqqs(p,q,r,circuit,-theta)

    else:
        if (r > s):
            Gdoubles_pqrs(p,q,r,s,circuit,theta)
        elif (r < s):
            Gdoubles_pqrs(p,q,s,r,circuit,-theta)


def Gdoubles_pqrs(p,q,r,s,circuit,theta,approx=cf.approx_exp):
    """ Function
        Given 
            Epqrs = p^ q^ r s
        with p > q,  r > s and max(p,q,r,s) = p, 
        compute Exp[i theta (Epqrs - Epqrs!)]
    """

    from qulacs.gate import PauliRotation    
    i4 = p  # (Fixed)
    if (q > r > s):
        i1 = s
        i2 = r
        i3 = q
    elif (r > q > s):
        i1 = s
        i2 = q
        i3 = r
    elif (r > s > q):
        i1 = q
        i2 = s
        i3 = r
    
    
    ### Type (a): 
    ### (1)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p X_q X_r X_s)]
    ### (2)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p Y_q X_r X_s)]
    ### (3)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p Y_q X_r Y_s)]
    ### (4)   Exp[ i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p Y_q Y_r X_s)]
    ### (5)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p X_q X_r Y_s)]
    ### (6)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p X_q Y_r X_s)]
    ### (7)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (Y_p X_q Y_r Y_s)]
    ### (8)   Exp[-i theta/8  Prod_{k=i1+1}^{i2-1} Z_k  Prod_{l=i3+1}^{i4-1} Z_l  (X_p Y_q Y_r Y_s)]
    target_list = []
    pauli_index = []
    
    for k in range(i1+1,i2):
        target_list.append(k)
        pauli_index.append(3)
    for l in range(i3+1,i4):
        target_list.append(l)
        pauli_index.append(3)

    target_list.append(p)
    target_list.append(q)
    target_list.append(r)
    target_list.append(s)
    ### (1)
    pauli_index.append(2)  # Y_p
    pauli_index.append(1)  # X_q
    pauli_index.append(1)  # X_r
    pauli_index.append(1)  # X_s
    gate = PauliRotation(target_list, pauli_index, theta/4) 
    circuit.add_gate(gate) 

    ### (2)
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.append(1)  # X_p
    pauli_index.append(2)  # Y_q
    pauli_index.append(1)  # X_r
    pauli_index.append(1)  # X_s
    gate = PauliRotation(target_list, pauli_index, theta/4) 
    circuit.add_gate(gate) 

    ### (3)
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.append(2)  # Y_p
    pauli_index.append(2)  # Y_q
    pauli_index.append(1)  # X_r
    pauli_index.append(2)  # Y_s
    gate = PauliRotation(target_list, pauli_index, theta/4) 
    circuit.add_gate(gate) 
        
    ### (4)
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.append(2)  # Y_p
    pauli_index.append(2)  # Y_q
    pauli_index.append(2)  # Y_r
    pauli_index.append(1)  # X_s
    gate = PauliRotation(target_list, pauli_index, theta/4) 
    circuit.add_gate(gate) 
    
    ### (5)
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.append(1)  # X_p
    pauli_index.append(1)  # X_q
    pauli_index.append(1)  # X_r
    pauli_index.append(2)  # Y_s
    gate = PauliRotation(target_list, pauli_index, -theta/4) 
    circuit.add_gate(gate)   
        
    ### (6)
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.append(1)  # X_p
    pauli_index.append(1)  # Y_q
    pauli_index.append(2)  # Y_r
    pauli_index.append(1)  # X_s
    gate = PauliRotation(target_list, pauli_index, -theta/4) 
    circuit.add_gate(gate) 
    
    ### (7)
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.append(2)  # Y_p
    pauli_index.append(1)  # X_q
    pauli_index.append(2)  # Y_r
    pauli_index.append(2)  # Y_s
    gate = PauliRotation(target_list, pauli_index, -theta/4) 
    circuit.add_gate(gate) 
    
    ### (8)
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.append(1)  # X_p
    pauli_index.append(2)  # Y_q
    pauli_index.append(2)  # Y_r
    pauli_index.append(2)  # Y_s
    gate = PauliRotation(target_list, pauli_index, -theta/4) 
    circuit.add_gate(gate) 

def Gdoubles_pqps(p,q,s,circuit,theta):
    """ Function
        Given 
            Epqps = p^ q^ p s
        with p > q > s, compute Exp[i theta (Epqps - Epqps!)]
    """
    from qulacs.gate import PauliRotation
    ### Type (b)
    ### (1)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  (Z_p X_q Y_s )]
    ### (2)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  (Y_q X_s )]
    ### (3)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  (Z_p Y_q X_s )]
    ### (4)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  (X_q Y_s )]        
    target_list = []
    pauli_index = []
    
    for k in range(s+1,q):
        target_list.append(k)
        pauli_index.append(3)

    target_list.append(p)
    target_list.append(q)
    target_list.append(s)
    
    ### (1)
    pauli_index.append(3)  # Z_p  
    pauli_index.append(1)  # X_q
    pauli_index.append(2)  # Y_s      
    gate = PauliRotation(target_list, pauli_index, theta/2) 
    circuit.add_gate(gate)    
              
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    
    ### (2)
    pauli_index.append(0)  # I_p
    pauli_index.append(2)  # Y_q
    pauli_index.append(1)  # X_s
    gate = PauliRotation(target_list, pauli_index, theta/2) 
    circuit.add_gate(gate)         
        
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()        
        
    ### (3)
    pauli_index.append(3)  # Z_p  
    pauli_index.append(2)  # Y_q
    pauli_index.append(1)  # X_s        
    gate = PauliRotation(target_list, pauli_index, -theta/2) 
    circuit.add_gate(gate)         

    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()        
        
    ### (4)        
    pauli_index.append(0)  # I_p   
    pauli_index.append(1)  # X_q
    pauli_index.append(2)  # Y_s     
    gate = PauliRotation(target_list, pauli_index, -theta/2) 
    circuit.add_gate(gate)
        
def Gdoubles_pqqs(p,q,s,circuit,theta):
    """ Function
        Given 
            Epqps = p^ q^ q s
        with p > q > s, compute Exp[i theta (Epqqs - Epqqs!)]
    """
    from qulacs.gate import PauliRotation    
    ### Type (c)
    ### (1)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (X_p Z_q Y_s )]
    ### (2)   Exp[ i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (Y_p X_s )]
    ### (3)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (Y_p Z_q X_s )]
    ### (4)   Exp[-i theta/4  Prod_{k=s+1}^{q-1} Z_k  Prod_{l=q+1}^{p-1} Z_l  (X_p Y_s )]        
    target_list = []
    pauli_index = []
    
    for k in range(s+1,q):
        target_list.append(k)
        pauli_index.append(3)
    for k in range(q+1,p):
        target_list.append(k)
        pauli_index.append(3)
        
    target_list.append(p)
    target_list.append(q)
    target_list.append(s)
    
    ### (1)
    pauli_index.append(1)  # X_p  
    pauli_index.append(3)  # Z_q
    pauli_index.append(2)  # Y_s      
    gate = PauliRotation(target_list, pauli_index, theta/2) 
    circuit.add_gate(gate)    
              
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    
    ### (2)
    pauli_index.append(2)  # Y_p
    pauli_index.append(0)  # I_q
    pauli_index.append(1)  # X_s
    gate = PauliRotation(target_list, pauli_index, theta/2) 
    circuit.add_gate(gate)         
        
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()        
        
    ### (3)
    pauli_index.append(2)  # Y_p  
    pauli_index.append(3)  # Z_q
    pauli_index.append(1)  # X_s        
    gate = PauliRotation(target_list, pauli_index, -theta/2) 
    circuit.add_gate(gate)         

    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()        
        
    ### (4)        
    pauli_index.append(1)  # X_p   
    pauli_index.append(0)  # I_q
    pauli_index.append(2)  # Y_s     
    gate = PauliRotation(target_list, pauli_index, -theta/2) 
    circuit.add_gate(gate)


def Gdoubles_pqrq(p,q,r,circuit,theta):
    """ Function
        Given 
            Epqps = p^ q^ r q
        with p > r > q, compute Exp[i theta (Epqrq - Epqrq!)]
    """
    from qulacs.gate import PauliRotation    
    ### Type (d)
    ### (1)   Exp[ i theta/4  Prod_{k=r+1}^{p-1} Z_k  (X_p Z_q Y_r)]
    ### (2)   Exp[ i theta/4  Prod_{k=r+1}^{p-1} Z_k  (Y_p X_r)]
    ### (3)   Exp[-i theta/4  Prod_{k=r+1}^{p-1} Z_k  (Y_p Z_q X_r )]
    ### (4)   Exp[-i theta/4  Prod_{k=r+1}^{p-1} Z_k  (X_p Y_r )]        
    target_list = []
    pauli_index = []
    
    for k in range(r+1,p):
        target_list.append(k)
        pauli_index.append(3)
        
    target_list.append(p)
    target_list.append(q)
    target_list.append(r)
    
    ### (1)
    pauli_index.append(1)  # X_p  
    pauli_index.append(3)  # Z_q
    pauli_index.append(2)  # Y_r      
    gate = PauliRotation(target_list, pauli_index, theta/2) 
    circuit.add_gate(gate)    
              
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()
    
    ### (2)
    pauli_index.append(2)  # Y_p
    pauli_index.append(0)  # I_q
    pauli_index.append(1)  # X_r
    gate = PauliRotation(target_list, pauli_index, theta/2) 
    circuit.add_gate(gate)         
        
    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()        
        
    ### (3)
    pauli_index.append(2)  # Y_p  
    pauli_index.append(3)  # Z_q
    pauli_index.append(1)  # X_r       
    gate = PauliRotation(target_list, pauli_index, -theta/2) 
    circuit.add_gate(gate)         

    pauli_index.pop()
    pauli_index.pop()
    pauli_index.pop()        
        
    ### (4)        
    pauli_index.append(1)  # X_p   
    pauli_index.append(0)  # I_q
    pauli_index.append(2)  # Y_r    
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

