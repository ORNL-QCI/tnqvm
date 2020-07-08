# This is a utility script which parses qflex input circuit 
# into XASM kernel. 
# Note: the qubit index mapping (lattice to linear) is
# specific to the Sycamore device.
import os, re

qubitIdxMap = {}
qubitIdxMap[5] = 0
qubitIdxMap[6] = 1
qubitIdxMap[14] = 2
qubitIdxMap[15] = 3
qubitIdxMap[16] = 4
qubitIdxMap[17] = 5
qubitIdxMap[24] = 6
qubitIdxMap[25] = 7
qubitIdxMap[26] = 8
qubitIdxMap[27] = 9
qubitIdxMap[28] = 10
qubitIdxMap[32] = 11
qubitIdxMap[33] = 12
qubitIdxMap[34] = 13
qubitIdxMap[35] = 14
qubitIdxMap[36] = 15
qubitIdxMap[37] = 16
qubitIdxMap[38] = 17
qubitIdxMap[39] = 18
qubitIdxMap[41] = 19
qubitIdxMap[42] = 20
qubitIdxMap[43] = 21
qubitIdxMap[44] = 22
qubitIdxMap[45] = 23
qubitIdxMap[46] = 24
qubitIdxMap[47] = 25
qubitIdxMap[48] = 26
qubitIdxMap[49] = 27
qubitIdxMap[50] = 28
qubitIdxMap[51] = 29
qubitIdxMap[52] = 30
qubitIdxMap[53] = 31
qubitIdxMap[54] = 32
qubitIdxMap[55] = 33
qubitIdxMap[56] = 34
qubitIdxMap[57] = 35
qubitIdxMap[58] = 36
qubitIdxMap[61] = 37
qubitIdxMap[62] = 38
qubitIdxMap[63] = 39
qubitIdxMap[64] = 40
qubitIdxMap[65] = 41
qubitIdxMap[66] = 42
qubitIdxMap[67] = 43
qubitIdxMap[72] = 44
qubitIdxMap[73] = 45
qubitIdxMap[74] = 46
qubitIdxMap[75] = 47
qubitIdxMap[76] = 48
qubitIdxMap[83] = 49
qubitIdxMap[84] = 50
qubitIdxMap[85] = 51
qubitIdxMap[94] = 52

# Change current dir to this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def parseFile(fileName):
    xasmSrcLines = []
    xasmSrcLines.append('__qpu__ void sycamoreCirc(qbit q) {')

    with open('resources/' + fileName, 'r') as sourceFile:
        line = sourceFile.readline()
        while line:
            line = line.replace('rz', 'Rz')
            line = line.replace(', ', ',')
            components = re.split('\s+', line)
            if len(components) == 4:
                # 1-q gate
                gateName = components[1]
                if gateName.startswith('Rz') is True:
                    # qflex Rz gate param is the exponent (not radiant)
                    angle = str(float(gateName[3:-1]) * (3.14159265359))
                    gateName = 'Rz'
                if gateName == 'x_1_2':
                    angle = '1.57079632679'
                    gateName = 'Rx'
                if gateName == 'y_1_2':
                    angle = '1.57079632679'
                    gateName = 'Ry'
                if gateName == 'hz_1_2':
                    # qFlex's hz_1_2 == Cirq's PhasedXPowGate(phase_exponent=0.25, exponent=0.5)
                    # i.e. hz_1_2 == Z^-0.25─X^0.5─Z^0.25 == Rz(pi/4) - Rx(-pi/2) - Rz(-pi/4)
                    qubit = 'q[' + str(qubitIdxMap[int(components[2])]) + ']'
                    # Comment block around this transformation
                    xasmSrcLines.append('// Begin hz_1_2')
                    xasmSrcLines.append('Rz' + '(' + qubit + ', ' + '-0.78539816339' + ');')
                    xasmSrcLines.append('Rx' + '(' + qubit + ', ' + '1.57079632679' + ');')
                    xasmSrcLines.append('Rz' + '(' + qubit + ', ' + '0.78539816339' + ');')
                    xasmSrcLines.append('// End hz_1_2')
                    # We don't need to handle this gate anymore.
                    gateName = ''
                qubit = 'q[' + str(qubitIdxMap[int(components[2])]) + ']'
                if gateName ==  'Rx' or gateName ==  'Ry' or gateName ==  'Rz':
                    xasmSrcLines.append(gateName + '(' + qubit + ', ' + angle + ');')
                elif len(gateName) > 0:
                    xasmSrcLines.append(gateName + '(' + qubit +');')
            if len(components) == 5:
                # 2-q gate
                gateName = components[1]
                if gateName.startswith('fsim') is True:
                    angles = gateName[5:-1].split(',')
                    # qFlex uses fractions of pi instead of radians
                    theta = float(angles[0]) * 3.14159265359
                    phi = float(angles[1]) * 3.14159265359
                    gateName = 'fSim'
                    qubit1 = 'q[' + str(qubitIdxMap[int(components[2])]) + ']'
                    qubit2 = 'q[' + str(qubitIdxMap[int(components[3])]) + ']'
                    xasmSrcLines.append(gateName + '(' + qubit1 + ', ' + qubit2 + ', ' + str(theta) + ', ' + str(phi) + ');')
            line = sourceFile.readline()
        sourceFile.close()

    xasmSrcLines.append('}')
    return '\n'.join(xasmSrcLines)

# Parse all qFlex files to XASM and save
for filename in os.listdir('resources'):
    if filename.endswith('.txt'): 
        xasmSrc = parseFile(filename)
        pre, ext = os.path.splitext(filename)
        xasmFilename = pre + '.xasm'
        with open('resources/' + xasmFilename, 'w') as xasmFile:
            xasmFile.write(xasmSrc)
