/***********************************************************************************
 * Copyright (c) 2016, UT-Battelle
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the xacc nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Initial API and implementation - Alex McCaskey
 *
 **********************************************************************************/
#include "XACC.hpp"

// Quantum Kernel executing teleportation of
// qubit state to another.
// test
    const char* src = R"src(__qpu__ teleport(qbit qreg){
CNOT(qreg[19], qreg[57]);
Z(qreg[54]);
Y(qreg[45]);
Y(qreg[67]);
CNOT(qreg[79], qreg[23]);
H(qreg[70]);
CNOT(qreg[44], qreg[70]);
CNOT(qreg[60], qreg[92]);
CNOT(qreg[54], qreg[81]);
H(qreg[99]);
X(qreg[23]);
X(qreg[24]);
CNOT(qreg[68], qreg[59]);
CNOT(qreg[63], qreg[13]);
CNOT(qreg[37], qreg[1]);
CNOT(qreg[37], qreg[48]);
H(qreg[34]);
X(qreg[5]);
X(qreg[74]);
CNOT(qreg[31], qreg[88]);
H(qreg[34]);
CNOT(qreg[70], qreg[60]);
CNOT(qreg[92], qreg[95]);
X(qreg[29]);
Z(qreg[57]);
CNOT(qreg[79], qreg[7]);
CNOT(qreg[11], qreg[22]);
X(qreg[95]);
X(qreg[61]);
Y(qreg[23]);
CNOT(qreg[12], qreg[3]);
Z(qreg[74]);
CNOT(qreg[89], qreg[65]);
Z(qreg[91]);
CNOT(qreg[63], qreg[33]);
CNOT(qreg[67], qreg[97]);
X(qreg[43]);
CNOT(qreg[74], qreg[90]);
X(qreg[69]);
H(qreg[47]);
Y(qreg[48]);
H(qreg[54]);
Z(qreg[94]);
CNOT(qreg[40], qreg[77]);
CNOT(qreg[53], qreg[87]);
CNOT(qreg[16], qreg[11]);
Y(qreg[40]);
X(qreg[9]);
CNOT(qreg[24], qreg[63]);
Y(qreg[43]);
Y(qreg[3]);
Z(qreg[91]);
Y(qreg[70]);
CNOT(qreg[57], qreg[7]);
CNOT(qreg[82], qreg[86]);
CNOT(qreg[76], qreg[62]);
CNOT(qreg[29], qreg[68]);
Z(qreg[62]);
H(qreg[42]);
Y(qreg[53]);
CNOT(qreg[25], qreg[47]);
H(qreg[19]);
CNOT(qreg[29], qreg[54]);
Y(qreg[23]);
X(qreg[20]);
Z(qreg[12]);
Z(qreg[57]);
Z(qreg[72]);
CNOT(qreg[19], qreg[29]);
CNOT(qreg[34], qreg[8]);
CNOT(qreg[42], qreg[23]);
CNOT(qreg[10], qreg[74]);
CNOT(qreg[21], qreg[46]);
X(qreg[77]);
H(qreg[42]);
H(qreg[39]);
CNOT(qreg[86], qreg[25]);
X(qreg[64]);
Y(qreg[42]);
})src";

int main (int argc, char** argv) {

	// Initialize the XACC Framework
	xacc::Initialize(argc, argv);

	auto qpu = xacc::getAccelerator("tnqvm");

	// Allocate a register of 3 qubits
	auto qubitReg = qpu->createBuffer("qreg", 100);

	// Create a Program
	xacc::Program program(qpu, src);

	// Request the quantum kernel representing
	// the above source code
	auto teleport = program.getKernel("teleport");

	// Execute!
	teleport(qubitReg);

	qubitReg->print(std::cout);

	// Finalize the XACC Framework
	xacc::Finalize();

	return 0;
}



