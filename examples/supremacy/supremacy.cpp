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
#include "tnqvm/TNQVMBuffer.hpp"

// Quantum Kernel executing teleportation of
// qubit state to another.
// test
    const char* src = R"src(__qpu__ supremacy(qbit q) {
cbit c[13] ;
H(q[0]);
H(q[1]);
H(q[2]);
H(q[3]);
H(q[4]);
H(q[5]);
H(q[6]);
H(q[7]);
H(q[8]);
H(q[9]);
H(q[10]);
H(q[11]);
H(q[12]);
X(q[0]);
Rz(q[1], 1.5707963267948966);
Rz(q[2], 0.7853981633974483);
X(q[3]);
X(q[4]);
H(q[5]);
H(q[6]);
Rz(q[7], 1.5707963267948966);
H(q[8]);
Y(q[9]);
Rz(q[10], 1.5707963267948966);
Y(q[11]);
Y(q[12]);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Z(q[0]);
Rz(q[1], 0.7853981633974483);
H(q[2]);
Z(q[3]);
Rz(q[4], 0.7853981633974483);
Z(q[5]);
H(q[6]);
Z(q[7]);
Z(q[8]);
H(q[9]);
Z(q[10]);
Y(q[11]);
Y(q[12]);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Z(q[0]);
Z(q[1]);
X(q[2]);
Z(q[3]);
Z(q[4]);
Rz(q[5], 0.7853981633974483);
Z(q[6]);
Rz(q[7], 0.7853981633974483);
Y(q[8]);
Z(q[9]);
Z(q[10]);
Rz(q[11], 1.5707963267948966);
Rz(q[12], 0.7853981633974483);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Rz(q[0], 1.5707963267948966);
Y(q[1]);
Y(q[2]);
Rz(q[3], 1.5707963267948966);
Z(q[4]);
Rz(q[5], 0.7853981633974483);
Rz(q[6], 0.7853981633974483);
H(q[7]);
Rz(q[8], 1.5707963267948966);
X(q[9]);
Y(q[10]);
Z(q[11]);
Rz(q[12], 0.7853981633974483);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Rz(q[0], 0.7853981633974483);
H(q[1]);
Y(q[2]);
Rz(q[3], 1.5707963267948966);
H(q[4]);
X(q[5]);
Rz(q[6], 0.7853981633974483);
X(q[7]);
Y(q[8]);
Rz(q[9], 1.5707963267948966);
Z(q[10]);
Z(q[11]);
Y(q[12]);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Rz(q[0], 1.5707963267948966);
H(q[1]);
X(q[2]);
H(q[3]);
Rz(q[4], 0.7853981633974483);
Y(q[5]);
Rz(q[6], 0.7853981633974483);
X(q[7]);
X(q[8]);
Rz(q[9], 0.7853981633974483);
X(q[10]);
H(q[11]);
Y(q[12]);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Z(q[0]);
Rz(q[1], 1.5707963267948966);
Rz(q[2], 0.7853981633974483);
Rz(q[3], 1.5707963267948966);
H(q[4]);
Rz(q[5], 0.7853981633974483);
Y(q[6]);
Rz(q[7], 1.5707963267948966);
Y(q[8]);
H(q[9]);
Rz(q[10], 0.7853981633974483);
Rz(q[11], 1.5707963267948966);
Rz(q[12], 1.5707963267948966);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Z(q[0]);
Y(q[1]);
H(q[2]);
Rz(q[3], 1.5707963267948966);
Rz(q[4], 0.7853981633974483);
H(q[5]);
Y(q[6]);
X(q[7]);
Z(q[8]);
Rz(q[9], 0.7853981633974483);
Z(q[10]);
Rz(q[11], 0.7853981633974483);
H(q[12]);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Z(q[0]);
Y(q[1]);
X(q[2]);
Y(q[3]);
Y(q[4]);
Rz(q[5], 1.5707963267948966);
X(q[6]);
Y(q[7]);
Y(q[8]);
X(q[9]);
X(q[10]);
Rz(q[11], 0.7853981633974483);
Z(q[12]);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
Rz(q[0], 0.7853981633974483);
X(q[1]);
Z(q[2]);
Z(q[3]);
Y(q[4]);
Z(q[5]);
X(q[6]);
Y(q[7]);
H(q[8]);
Rz(q[9], 1.5707963267948966);
Y(q[10]);
X(q[11]);
H(q[12]);
CNOT(q[0], q[1]);
CNOT(q[2], q[3]);
CNOT(q[4], q[5]);
CNOT(q[6], q[7]);
CNOT(q[8], q[9]);
CNOT(q[10], q[11]);
CNOT(q[1], q[2]);
CNOT(q[3], q[4]);
CNOT(q[5], q[6]);
CNOT(q[7], q[8]);
CNOT(q[9], q[10]);
CNOT(q[11], q[12]);
c[0] = MeasZ(q[0]);
c[1] = MeasZ(q[1]);
c[2] = MeasZ(q[2]);
c[3] = MeasZ(q[3]);
c[4] = MeasZ(q[4]);
c[5] = MeasZ(q[5]);
c[6] = MeasZ(q[6]);
c[7] = MeasZ(q[7]);
c[8] = MeasZ(q[8]);
c[9] = MeasZ(q[9]);
c[10] = MeasZ(q[10]);
c[11] = MeasZ(q[11]);
c[12] = MeasZ(q[12]);
}
)src";

int main (int argc, char** argv) {

	// Initialize the XACC Framework
	xacc::Initialize(argc, argv);

	auto qpu = xacc::getAccelerator("tnqvm");

	// Allocate a register of 2 qubits
	auto qubitReg = qpu->createBuffer("q", 13);
	std::dynamic_pointer_cast<tnqvm::TNQVMBuffer>(qubitReg)->set_verbose(1);

	// Create a Program
	xacc::Program program(qpu, src);

	auto k = program.getKernel("supremacy");

	k(qubitReg);

	// Execute!
//	std::ofstream file("energy_vs_theta.csv");
//	file<<"theta, Z_0, Z_1, Z_0 Z_1\n";
	double pi = 3.14159265359;

	auto ms = qubitReg->getMeasurementStrings();

	for (auto m : ms) {
		std::cout << "M: " << m << "\n";
	}

	// Finalize the XACC Framework
	xacc::Finalize();

	return 0;
}


