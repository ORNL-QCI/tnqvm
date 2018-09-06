import random
import numpy as np
import pyxacc as xacc

def write_qasm(rounds, nQ):
    """"
    Parameters
    ----------
    nQ: int, number of qubits
    entanglers: tuple of tuples whose length determines the circuit depth
                sub-tuples enumerates CNOT's acting at a given depth level
                
    Returns
    -------
    param_counter: number of variational parameters in circuit
    """
    file_name = "Supremacy_1D_{0}_qubits_{1}_rounds.qasm".format(nQ, rounds)
    file = open(file_name, "w")
    file.write("__qpu__ f(AcceleratorBuffer b) {\n")
    local_gates = ('X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'H')
    
    if nQ%2==0: # even
        evens = tuple((i, i+1) for i in range(nQ) if i%2==0)
        odds = tuple((i, i+1) for i in range(nQ) if i%2==1)
    
    if nQ%2==1: # odd
        evens = tuple((i, i+1) for i in range(nQ-2) if i%2==0)
        odds = tuple((i, i+1) for i in range(nQ) if i%2==1)
        
    for i in range(nQ): 
        file.write('H ' + str(i) + '\n')  # initial Hadamards

    # random gates
    for i in range(rounds):
        for j in range(nQ): 
            gate = random.choice(local_gates)
            if 'R' in gate:
                angle = str(np.random.uniform(0,np.pi));
                file.write('{} '.format(str(gate) + '('+str(angle)+')') + str(j) + '\n')
            else:
                file.write('{} '.format(str(gate)) + str(j) + ' \n')
                       
        for e in evens:
            file.write('CNOT {0} {1}\n'.format(*e))
            
        for e in odds:
            file.write('CNOT {0} {1}\n'.format(*e))
            
    file.write("}")
    file.close()
    return open(file_name).read()

@profile
def execute(nq, rounds):
    src = write_qasm(rounds, nq)
    qpu = xacc.getAccelerator('tnqvm')
    xacc.setOption('tnqvm-verbose','')
    f = xacc.compileKernel(qpu, src)
    buf = qpu.createBuffer('q',nq+1)
    qpu.execute(buf, f)

if __name__ == '__main__':
    xacc.Initialize(['--compiler','quil'])
    execute(10,2)