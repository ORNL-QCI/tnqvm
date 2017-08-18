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
// A 100-qubit circuit with 357 gates
// can be simulated by TNQVM (using ITensor backend) on single 2.8GHz CPU in about 168 minutes
X(qreg[88]);
CNOT(qreg[75], qreg[32]);
CNOT(qreg[6], qreg[73]);
CNOT(qreg[36], qreg[77]);
Y(qreg[82]);
CNOT(qreg[62], qreg[55]);
Z(qreg[15]);
H(qreg[13]);
CNOT(qreg[43], qreg[57]);
CNOT(qreg[15], qreg[67]);
CNOT(qreg[53], qreg[67]);
H(qreg[22]);
CNOT(qreg[29], qreg[33]);
X(qreg[76]);
CNOT(qreg[76], qreg[50]);
CNOT(qreg[32], qreg[35]);
H(qreg[60]);
H(qreg[80]);
CNOT(qreg[39], qreg[30]);
CNOT(qreg[67], qreg[95]);
CNOT(qreg[22], qreg[69]);
X(qreg[33]);
Z(qreg[3]);
H(qreg[67]);
Y(qreg[42]);
CNOT(qreg[44], qreg[67]);
CNOT(qreg[29], qreg[44]);
CNOT(qreg[90], qreg[58]);
CNOT(qreg[34], qreg[65]);
CNOT(qreg[80], qreg[91]);
Z(qreg[77]);
Z(qreg[9]);
CNOT(qreg[54], qreg[39]);
CNOT(qreg[52], qreg[13]);
CNOT(qreg[29], qreg[73]);
CNOT(qreg[97], qreg[31]);
CNOT(qreg[42], qreg[7]);
CNOT(qreg[41], qreg[54]);
CNOT(qreg[19], qreg[10]);
Y(qreg[90]);
CNOT(qreg[86], qreg[98]);
Z(qreg[94]);
CNOT(qreg[50], qreg[49]);
X(qreg[42]);
CNOT(qreg[47], qreg[0]);
Z(qreg[6]);
Z(qreg[87]);
CNOT(qreg[14], qreg[4]);
CNOT(qreg[89], qreg[41]);
CNOT(qreg[43], qreg[13]);
X(qreg[5]);
Z(qreg[43]);
CNOT(qreg[12], qreg[34]);
X(qreg[67]);
Y(qreg[50]);
Z(qreg[69]);
Z(qreg[66]);
Z(qreg[27]);
Z(qreg[65]);
CNOT(qreg[97], qreg[40]);
CNOT(qreg[42], qreg[18]);
Z(qreg[77]);
Z(qreg[25]);
Y(qreg[31]);
CNOT(qreg[46], qreg[25]);
X(qreg[28]);
X(qreg[57]);
H(qreg[45]);
CNOT(qreg[64], qreg[81]);
CNOT(qreg[86], qreg[32]);
Z(qreg[25]);
CNOT(qreg[53], qreg[83]);
H(qreg[98]);
Z(qreg[61]);
H(qreg[49]);
Y(qreg[0]);
Z(qreg[1]);
CNOT(qreg[65], qreg[94]);
X(qreg[43]);
CNOT(qreg[38], qreg[54]);
H(qreg[78]);
Z(qreg[52]);
CNOT(qreg[30], qreg[4]);
CNOT(qreg[28], qreg[87]);
CNOT(qreg[2], qreg[32]);
CNOT(qreg[48], qreg[54]);
CNOT(qreg[66], qreg[81]);
Y(qreg[68]);
X(qreg[70]);
CNOT(qreg[52], qreg[79]);
CNOT(qreg[20], qreg[51]);
CNOT(qreg[35], qreg[57]);
CNOT(qreg[31], qreg[10]);
CNOT(qreg[57], qreg[41]);
CNOT(qreg[58], qreg[64]);
X(qreg[33]);
CNOT(qreg[90], qreg[13]);
CNOT(qreg[64], qreg[48]);
X(qreg[14]);
CNOT(qreg[59], qreg[17]);
H(qreg[68]);
Z(qreg[79]);
Z(qreg[12]);
Y(qreg[23]);
CNOT(qreg[73], qreg[26]);
X(qreg[3]);
Y(qreg[25]);
CNOT(qreg[24], qreg[53]);
X(qreg[28]);
X(qreg[90]);
Y(qreg[88]);
CNOT(qreg[87], qreg[40]);
CNOT(qreg[35], qreg[59]);
CNOT(qreg[78], qreg[62]);
CNOT(qreg[34], qreg[64]);
CNOT(qreg[42], qreg[73]);
Y(qreg[28]);
H(qreg[27]);
CNOT(qreg[67], qreg[84]);
CNOT(qreg[50], qreg[2]);
Y(qreg[2]);
Y(qreg[13]);
CNOT(qreg[57], qreg[67]);
Z(qreg[95]);
Z(qreg[49]);
CNOT(qreg[91], qreg[38]);
CNOT(qreg[10], qreg[88]);
Y(qreg[61]);
CNOT(qreg[34], qreg[0]);
Y(qreg[92]);
CNOT(qreg[41], qreg[29]);
CNOT(qreg[38], qreg[55]);
H(qreg[98]);
Z(qreg[56]);
CNOT(qreg[32], qreg[19]);
CNOT(qreg[86], qreg[5]);
CNOT(qreg[94], qreg[19]);
H(qreg[31]);
H(qreg[50]);
X(qreg[7]);
X(qreg[61]);
CNOT(qreg[30], qreg[96]);
CNOT(qreg[63], qreg[99]);
Z(qreg[65]);
H(qreg[23]);
CNOT(qreg[95], qreg[16]);
Y(qreg[43]);
CNOT(qreg[31], qreg[49]);
X(qreg[72]);
CNOT(qreg[78], qreg[11]);
CNOT(qreg[94], qreg[86]);
Y(qreg[55]);
CNOT(qreg[16], qreg[71]);
CNOT(qreg[25], qreg[12]);
H(qreg[31]);
CNOT(qreg[49], qreg[11]);
CNOT(qreg[87], qreg[39]);
Z(qreg[8]);
Y(qreg[21]);
X(qreg[79]);
CNOT(qreg[43], qreg[38]);
CNOT(qreg[21], qreg[59]);
CNOT(qreg[25], qreg[45]);
X(qreg[12]);
Y(qreg[13]);
CNOT(qreg[43], qreg[74]);
Z(qreg[62]);
CNOT(qreg[1], qreg[73]);
Y(qreg[36]);
CNOT(qreg[11], qreg[67]);
Y(qreg[79]);
H(qreg[45]);
H(qreg[6]);
CNOT(qreg[1], qreg[5]);
CNOT(qreg[41], qreg[99]);
X(qreg[2]);
CNOT(qreg[75], qreg[77]);
H(qreg[97]);
Y(qreg[32]);
CNOT(qreg[20], qreg[52]);
H(qreg[51]);
CNOT(qreg[52], qreg[14]);
CNOT(qreg[64], qreg[6]);
CNOT(qreg[86], qreg[22]);
CNOT(qreg[42], qreg[94]);
CNOT(qreg[44], qreg[82]);
Z(qreg[38]);
CNOT(qreg[23], qreg[84]);
CNOT(qreg[88], qreg[97]);
CNOT(qreg[7], qreg[23]);
CNOT(qreg[9], qreg[82]);
CNOT(qreg[85], qreg[86]);
Z(qreg[95]);
CNOT(qreg[22], qreg[14]);
CNOT(qreg[46], qreg[44]);
CNOT(qreg[56], qreg[74]);
CNOT(qreg[9], qreg[94]);
H(qreg[95]);
CNOT(qreg[90], qreg[95]);
H(qreg[23]);
CNOT(qreg[55], qreg[9]);
CNOT(qreg[33], qreg[7]);
Y(qreg[73]);
CNOT(qreg[52], qreg[22]);
CNOT(qreg[44], qreg[17]);
Z(qreg[66]);
CNOT(qreg[12], qreg[93]);
X(qreg[58]);
CNOT(qreg[17], qreg[26]);
CNOT(qreg[32], qreg[51]);
Y(qreg[72]);
CNOT(qreg[51], qreg[84]);
CNOT(qreg[90], qreg[59]);
Z(qreg[8]);
X(qreg[16]);
X(qreg[54]);
CNOT(qreg[26], qreg[33]);
X(qreg[22]);
Z(qreg[94]);
CNOT(qreg[39], qreg[7]);
CNOT(qreg[0], qreg[49]);
Z(qreg[48]);
CNOT(qreg[56], qreg[71]);
CNOT(qreg[58], qreg[42]);
CNOT(qreg[24], qreg[99]);
CNOT(qreg[74], qreg[1]);
Z(qreg[67]);
H(qreg[79]);
CNOT(qreg[36], qreg[69]);
X(qreg[92]);
X(qreg[91]);
CNOT(qreg[16], qreg[41]);
H(qreg[2]);
CNOT(qreg[49], qreg[65]);
CNOT(qreg[39], qreg[85]);
CNOT(qreg[35], qreg[13]);
X(qreg[79]);
Z(qreg[49]);
CNOT(qreg[11], qreg[9]);
CNOT(qreg[25], qreg[68]);
CNOT(qreg[96], qreg[43]);
CNOT(qreg[44], qreg[99]);
CNOT(qreg[34], qreg[74]);
Z(qreg[40]);
Y(qreg[26]);
CNOT(qreg[61], qreg[10]);
X(qreg[65]);
Z(qreg[32]);
CNOT(qreg[5], qreg[22]);
Z(qreg[69]);
Y(qreg[80]);
CNOT(qreg[12], qreg[63]);
Z(qreg[86]);
X(qreg[90]);
Y(qreg[74]);
CNOT(qreg[29], qreg[88]);
X(qreg[46]);
CNOT(qreg[83], qreg[20]);
CNOT(qreg[9], qreg[58]);
Z(qreg[68]);
CNOT(qreg[77], qreg[82]);
CNOT(qreg[11], qreg[59]);
CNOT(qreg[95], qreg[18]);
H(qreg[89]);
X(qreg[75]);
CNOT(qreg[4], qreg[66]);
H(qreg[93]);
CNOT(qreg[26], qreg[87]);
CNOT(qreg[44], qreg[28]);
CNOT(qreg[35], qreg[59]);
CNOT(qreg[34], qreg[44]);
Y(qreg[47]);
CNOT(qreg[77], qreg[22]);
H(qreg[45]);
Y(qreg[3]);
Z(qreg[95]);
CNOT(qreg[81], qreg[38]);
CNOT(qreg[93], qreg[48]);
H(qreg[82]);
Y(qreg[28]);
CNOT(qreg[69], qreg[12]);
CNOT(qreg[92], qreg[14]);
X(qreg[83]);
Y(qreg[49]);
Z(qreg[43]);
CNOT(qreg[7], qreg[62]);
CNOT(qreg[89], qreg[44]);
H(qreg[80]);
Z(qreg[6]);
Z(qreg[78]);
H(qreg[60]);
Z(qreg[60]);
X(qreg[27]);
X(qreg[75]);
CNOT(qreg[1], qreg[41]);
CNOT(qreg[49], qreg[93]);
X(qreg[55]);
CNOT(qreg[95], qreg[13]);
CNOT(qreg[59], qreg[76]);
CNOT(qreg[79], qreg[91]);
Z(qreg[48]);
CNOT(qreg[58], qreg[94]);
CNOT(qreg[17], qreg[39]);
Z(qreg[63]);
CNOT(qreg[92], qreg[38]);
Z(qreg[60]);
H(qreg[26]);
CNOT(qreg[77], qreg[35]);
H(qreg[92]);
CNOT(qreg[79], qreg[9]);
CNOT(qreg[36], qreg[32]);
Z(qreg[73]);
H(qreg[43]);
X(qreg[11]);
H(qreg[51]);
Z(qreg[87]);
Y(qreg[43]);
X(qreg[57]);
H(qreg[58]);
H(qreg[78]);
Y(qreg[16]);
Y(qreg[95]);
CNOT(qreg[95], qreg[10]);
CNOT(qreg[56], qreg[59]);
CNOT(qreg[62], qreg[99]);
CNOT(qreg[21], qreg[4]);
X(qreg[24]);
X(qreg[16]);
CNOT(qreg[70], qreg[84]);
CNOT(qreg[33], qreg[95]);
Z(qreg[96]);
CNOT(qreg[34], qreg[70]);
CNOT(qreg[12], qreg[97]);
CNOT(qreg[49], qreg[65]);
CNOT(qreg[17], qreg[84]);
CNOT(qreg[19], qreg[96]);
X(qreg[18]);
Y(qreg[91]);
CNOT(qreg[63], qreg[62]);
CNOT(qreg[51], qreg[29]);
CNOT(qreg[96], qreg[63]);
CNOT(qreg[38], qreg[49]);
H(qreg[71]);
CNOT(qreg[15], qreg[41]);
CNOT(qreg[88], qreg[2]);
Y(qreg[44]);
CNOT(qreg[40], qreg[1]);
X(qreg[92]);
Z(qreg[40]);
CNOT(qreg[89], qreg[96]);
H(qreg[47]);
CNOT(qreg[79], qreg[52]);
H(qreg[38]);
CNOT(qreg[95], qreg[86]);
Z(qreg[49]);
Y(qreg[58]);
Y(qreg[9]);
})src";



int main (int argc, char** argv) {

	// Initialize the XACC Framework
	xacc::Initialize(argc, argv);

	auto qpu = xacc::getAccelerator("tnqvm");

	// Allocate a register of 100 qubits
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



