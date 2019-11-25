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

//#include "kernels_src.hpp"
// !!! IMPORTANT NOTE !!! This XASM source code is *OBSOLETE*, hence may not be compiled by the latest XACC compiler.
const char* circuit_src = R"src((qbit qreg, double theta0, double theta1){
cbit creg[4];
X(qreg[0]);
X(qreg[1]);
H(qreg[2]);
Rx(qreg[0],1.5708);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
Rz(qreg[2],1.57);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
H(qreg[2]);
Rx(qreg[0],10.9956);
Rx(qreg[2],1.5708);
H(qreg[0]);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
Rz(qreg[2],10.9964);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
Rx(qreg[2],10.9956);
H(qreg[0]);
H(qreg[3]);
Rx(qreg[1],1.5708);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],1.57);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
H(qreg[3]);
Rx(qreg[1],10.9956);
Rx(qreg[3],1.5708);
H(qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],10.9964);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
Rx(qreg[3],10.9956);
H(qreg[1]);
H(qreg[3]);
Rx(qreg[2],1.5708);
Rx(qreg[1],1.5708);
Rx(qreg[0],1.5708);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],12.1914);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
H(qreg[3]);
Rx(qreg[2],10.9956);
Rx(qreg[1],10.9956);
Rx(qreg[0],10.9956);
Rx(qreg[3],1.5708);
Rx(qreg[2],1.5708);
H(qreg[1]);
Rx(qreg[0],1.5708);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],0.375);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
Rx(qreg[3],10.9956);
Rx(qreg[2],10.9956);
H(qreg[1]);
Rx(qreg[0],10.9956);
H(qreg[3]);
Rx(qreg[2],1.5708);
H(qreg[1]);
H(qreg[0]);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],0.375);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
H(qreg[3]);
Rx(qreg[2],10.9956);
H(qreg[1]);
H(qreg[0]);
Rx(qreg[3],1.5708);
Rx(qreg[2],1.5708);
Rx(qreg[1],1.5708);
H(qreg[0]);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],0.375);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
Rx(qreg[3],10.9956);
Rx(qreg[2],10.9956);
Rx(qreg[1],10.9956);
H(qreg[0]);
H(qreg[3]);
H(qreg[2]);
H(qreg[1]);
Rx(qreg[0],1.5708);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],12.1914);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
H(qreg[3]);
H(qreg[2]);
H(qreg[1]);
Rx(qreg[0],10.9956);
Rx(qreg[3],1.5708);
H(qreg[2]);
Rx(qreg[1],1.5708);
Rx(qreg[0],1.5708);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],12.1914);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
Rx(qreg[3],10.9956);
H(qreg[2]);
Rx(qreg[1],10.9956);
Rx(qreg[0],10.9956);
H(qreg[3]);
H(qreg[2]);
Rx(qreg[1],1.5708);
H(qreg[0]);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],12.1914);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
H(qreg[3]);
H(qreg[2]);
Rx(qreg[1],10.9956);
H(qreg[0]);
Rx(qreg[3],1.5708);
H(qreg[2]);
H(qreg[1]);
H(qreg[0]);
CNOT(qreg[0],qreg[1]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[2],qreg[3]);
Rz(qreg[3],0.375);
CNOT(qreg[2],qreg[3]);
CNOT(qreg[1],qreg[2]);
CNOT(qreg[0],qreg[1]);
Rx(qreg[3],10.9956);
H(qreg[2]);
H(qreg[1]);
H(qreg[0]);
H(qreg[3]);
Rx(qreg[2],1.5708);
Rx(qreg[1],1.5708);
H(qreg[0]);
creg[3] = MeasZ(qreg[3]);
creg[2] = MeasZ(qreg[2]);
creg[1] = MeasZ(qreg[1]);
creg[0] = MeasZ(qreg[0]);
}
)src";

std::string n_copies_kernel_src(int n_copies){
	std::string src;
	for(int i=0; i<n_copies; ++i){
		src += "__qpu__ kernel_"+std::to_string(i)+circuit_src;
	}
	return src;
}

int main (int argc, char** argv) {

	xacc::Initialize(argc, argv);

	// Allocate a register of 2 qubits
	// Create a Program
	auto qpu = xacc::getAccelerator("tnqvm");
	auto dummy_buffer = xacc::qalloc(4); // to work around Error "Could not find AcceleratorBuffer with id qreg"
	const int n_copies = 1000;
	auto src = n_copies_kernel_src(n_copies);
	auto xasmCompiler = xacc::getCompiler("xasm");
  	auto program = xasmCompiler->compile(src, qpu);
	auto kernels = program->getComposites();

	// Execute!
	double pi = 3.14159265359;
	double theta0 = .2;
	double theta1 = .5;

	double sum = 0.;
#pragma omp parallel for reduction (+:sum), num_threads(8)
	for(int i=0; i < kernels.size(); ++i){
		auto localqpu = xacc::getAccelerator("tnqvm");
		auto buffer = xacc::qalloc(4);
		localqpu->execute(buffer, kernels[i]);
		auto aver = buffer->getExpectationValueZ();
		sum += aver;
	}
	if (std::abs(sum/kernels.size()-(-.141236))<1e-5){
		std::cout<<"Test PASS!    ";
	}else{
		std::cout<<"Test FAIL!  <Z> of each copy should be -.141236, but ";
	}
	std::cout<<"<Z> of each copy = "<<sum/kernels.size()<<std::endl;
	
	xacc::Finalize();

	return 0;
}


