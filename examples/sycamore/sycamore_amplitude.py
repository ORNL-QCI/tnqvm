import xacc
import os
from pathlib import Path
import math

dir_path = os.path.dirname(os.path.realpath(__file__))

RESOURCE_DIR = dir_path + "/resources"
CIRCUIT_DEPTH = 1
XASM_SRC_FILE = RESOURCE_DIR + "/sycamore_53_" + str(CIRCUIT_DEPTH) + "_0.xasm"
xasmSrcStr = Path(XASM_SRC_FILE).read_text()

xasmCompiler = xacc.getCompiler("xasm")
ir = xasmCompiler.compile(xasmSrcStr)
program = ir.getComposites()[0]
BIT_STRING = []
N_QUBITS = 53
INITIAL_STATE_BIT_STRING = []
for i in range(N_QUBITS):
    BIT_STRING.append(0)
    INITIAL_STATE_BIT_STRING.append(0)

qpu = xacc.getAccelerator("tnqvm", {
    "tnqvm-visitor": "exatn",
    "bitstring": BIT_STRING,
    "exatn-buffer-size-gb": 2
})

qubitReg = xacc.qalloc(N_QUBITS)
qpu.execute(qubitReg, program)
#print(qubitReg)
realAmpl = qubitReg["amplitude-real"]
imagAmpl = qubitReg["amplitude-imag"]

print("================= RESULT =================")
print(INITIAL_STATE_BIT_STRING, " -->", BIT_STRING, ":", realAmpl, imagAmpl)
print("Bit-string probability:", math.sqrt(realAmpl*realAmpl + imagAmpl*imagAmpl))