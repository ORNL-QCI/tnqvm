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
#include "xacc.hpp"

// Quantum Kernel executing teleportation of
// qubit state to another.
// test
// !!! IMPORTANT NOTE !!! This XASM source code is *OBSOLETE*, hence may not be compiled by the latest XACC compiler.
    const char* src = R"src(
__qpu__ prepare_ansatz(qbit qreg, double theta){
	Rx(qreg[0], 3.1415926);
	Ry(qreg[1], 1.57079);
	Rx(qreg[0], 7.8539752);
	CNOT(qreg[1], qreg[0]);
	Rz(qreg[0], theta);
	CNOT(qreg[1], qreg[0]);
	Ry(qreg[1], 7.8539752);
	Rx(qreg[0], 1.57079);
}

// measure the 1st term of Hamiltonian on the ansatz
__qpu__ term0(qbit qreg, double theta){
	prepare_ansatz(qreg, theta);
	cbit creg[1];
	creg[0] = MeasZ(qreg[0]);
}

__qpu__ term1(qbit qreg, double theta){
	prepare_ansatz(qreg, theta);
	cbit creg[1];
	creg[0] = MeasZ(qreg[1]);
}

__qpu__ term2(qbit qreg, double theta){
	prepare_ansatz(qreg, theta);
	cbit creg[2];
	creg[0] = MeasZ(qreg[0]);
	creg[1] = MeasZ(qreg[1]);
}
)src";

int main (int argc, char** argv) {

	std::ofstream file("energy_vs_theta.csv");
	file<<"theta, Z_0, Z_1, Z_0 Z_1\n";
	double pi = 3.14159265359;

	// Initialize the XACC Framework
	xacc::Initialize(argc, argv);

	auto qpu = xacc::getAccelerator("tnqvm");

	// Allocate a register of 2 qubits
	auto qubitReg = xacc::qalloc(2);

	// Create a Program
	auto xasmCompiler = xacc::getCompiler("xasm");
  	auto program = xasmCompiler->compile(src, qpu);
	
	int n_terms=3;
	auto hamiltonianTermKernels = program->getComposites();

	// Execute!
	for(double theta = -pi; theta<=pi; theta += .1){
		file<<theta;
		
		for (auto& kernel: hamiltonianTermKernels){
			auto k_evaled = kernel->operator()({ theta });
			qpu->execute(qubitReg, k_evaled);
			file << ", ";
			file << qubitReg->getExpectationValueZ();
		}

		file << "\n";
		file.flush();
	}
	file.close();

	// Finalize the XACC Framework
	xacc::Finalize();

	return 0;
}


