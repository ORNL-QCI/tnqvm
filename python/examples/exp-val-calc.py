import sys
from pathlib import Path
sys.path.insert(1, str(Path.home()) + '/.xacc')
import xacc
xacc.qasm('''
.compiler xasm
.circuit ansatz
.qbit q
U(q[1], 1.5708,0,3.14159);
U(q[0], 1.5708,1.5708,4.71239); 
CNOT(q[0], q[1]);
U(q[2], 1.5708,-3.14159,3.14159); 
U(q[3], 1.5708,0,3.14159); 
CNOT(q[2], q[3]);
Rz(q[3], 0.101476); 
CNOT(q[2], q[3]);
CNOT(q[1], q[2]);
CNOT(q[0], q[1]);
U(q[3], 1.5708,0,3.14159); 
U(q[2], 1.5708,0,3.14159); 
U(q[0], 1.5708,1.5708,4.71239); 
U(q[1], 1.5708,0,3.14159); 
''')
ansatz = xacc.getCompiled('ansatz')
#qpu = xacc.getAccelerator('qsim')
qpu = xacc.getAccelerator('tnqvm:exatn', {"exp-val-by-conjugate": True, "max-qubit": 2})
print("QPU:", qpu.name())
buffer = xacc.qalloc(4)
geom = '''
Na  0.000000   0.0      0.0
H   0.0        0.0  1.914388
'''
fo = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
ao = [5, 9, 15, 19]
H = xacc.getObservable('pyscf', {'basis': 'sto-3g', 'geometry': geom,
                                       'frozen-spin-orbitals': fo, 'active-spin-orbitals': ao})
# print(H.toString())
opt = xacc.getOptimizer('nlopt')
vqe = xacc.getAlgorithm('vqe', {
                        'ansatz': ansatz,
                        'accelerator': qpu,
                        'observable': H,
                        'optimizer': opt
                        })
# # xacc.set_verbose(True)
energy = vqe.execute(buffer, [])
print('Energy = ', energy)
