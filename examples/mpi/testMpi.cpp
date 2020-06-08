#include "xacc.hpp"

int main (int argc, char** argv) {

	// Initialize the XACC Framework
	xacc::Initialize(argc, argv);
	auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});

    // Allocate a register of 4 qubits
	auto qubitReg = xacc::qalloc(4);

	// Create a Program
	auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test(qbit q, double theta) {
      H(q[0]);
      CX(q[0], q[1]);
      CX(q[1], q[2]);
      CX(q[2], q[3]);
    })", qpu);

	// Request the quantum kernel representing
	// the above source code
	auto program = ir->getComposite("test");
	const auto rotationAngle = M_PI / 3.0;
	auto evaled = program->operator()({ rotationAngle });
	// Execute!
	qpu->execute(qubitReg, evaled);
    
    qubitReg->print();

	// Finalize the XACC Framework
	xacc::Finalize();

	return 0;
}
