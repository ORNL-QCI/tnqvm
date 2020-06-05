#include "xacc.hpp"

int main (int argc, char** argv) {

	// Initialize the XACC Framework
	xacc::Initialize(argc, argv);
	auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});

    // Allocate a register of 2 qubits
	auto qubitReg = xacc::qalloc(2);

	// Create a Program
	auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test(qbit q, double theta) {
      H(q[0]);
      CX(q[0], q[1]);
	  Rx(q[0], theta);
	  Ry(q[1], theta);
	  H(q[1]);
	  CX(q[1], q[0]);
      Measure(q[0]);
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
