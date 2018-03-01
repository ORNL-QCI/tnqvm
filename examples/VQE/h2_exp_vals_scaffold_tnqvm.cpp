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
const char* src = R"src(
__qpu__ prepare_ansatz(AcceleratorBuffer qreg, double theta){
        RX(3.1415926) 0
        RY(1.57079) 1
        RX(7.8539752) 0
        CNOT 1 0
        RZ(theta) 0
        CNOT 1 0
        RY(7.8539752) 1
        RX(1.57079) 0
}

// measure the 1st term of Hamiltonian on the ansatz
__qpu__ term0(AcceleratorBuffer qreg, double theta){
        prepare_ansatz(qreg, theta)
        MEASURE 0 [0]
}

__qpu__ term1(AcceleratorBuffer qreg, double theta){
        prepare_ansatz(qreg, theta)
        MEASURE 1 [0]
}

__qpu__ term2(AcceleratorBuffer qreg, double theta){
        prepare_ansatz(qreg, theta)
        MEASURE 0 [0]
        MEASURE 1 [1]
}
)src";

int main (int argc, char** argv) {

	// Initialize the XACC Framework
	xacc::Initialize(argc, argv);

	xacc::setCompiler("quil");

	auto qpu = xacc::getAccelerator("tnqvm");

	// Allocate a register of 2 qubits
	auto buffer = qpu->createBuffer("qreg", 2);
	// Create a Program
	xacc::Program program(qpu, src);

	int n_terms=3;

	// Execute!
	std::ofstream file("energy_vs_theta.csv");
	file<<"theta, Z_0, Z_1, Z_0 Z_1\n";
	double pi = 3.14159265359;

	for(double theta = -pi; theta<=pi; theta += .1){
		file<<theta;
		for(int i=0; i<n_terms; ++i){
			std::string kernel_name = "term"+std::to_string(i);
			auto measure_term = program.getKernel<double>(kernel_name);
			buffer->resetBuffer();
			measure_term(buffer, theta);
			auto aver = buffer->getExpectationValueZ();
			file<<", "<<aver;
		}
		file<<std::endl;
	}
	file.close();
	
	buffer->print(std::cout);

	// Finalize the XACC Framework
	xacc::Finalize();

	return 0;
}


