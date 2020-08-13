// Runs a validation test by contracting the tensor network and its conjuate
// to make sure that the result is 1.0.
#include "xacc.hpp"
#include <iostream>
#include <fstream>
#include <numeric>

int main(int argc, char **argv)
{
    xacc::Initialize();
    xacc::set_verbose(true);
    xacc::setLoggingLevel(2);

    // Options: 4, 5, 6, 8, 10, 12, 14, 16, 18, 20
    const int CIRCUIT_DEPTH = 4;

   // Construct the full path to the XASM source file
    const std::string XASM_SRC_FILE = std::string(RESOURCE_DIR) + "/sycamore_53_" + std::to_string(CIRCUIT_DEPTH) + "_0.xasm";
    // Read XASM source
    std::ifstream inFile;
    inFile.open(XASM_SRC_FILE);
    std::stringstream strStream;
    strStream << inFile.rdbuf();
    const std::string kernelName = "sycamoreCirc";
    std::string xasmSrcStr = strStream.str();
    // Construct a unique kernel name:
    const std::string newKernelName = kernelName + "_" + std::to_string(CIRCUIT_DEPTH);
    xasmSrcStr.replace(xasmSrcStr.find(kernelName), kernelName.length(), newKernelName);

    // ExaTN visitor
    auto qpu = xacc::getAccelerator("tnqvm", {
        std::make_pair("tnqvm-visitor", "exatn"),
        std::make_pair("contract-with-conjugate", true),
        std::make_pair("exatn-buffer-size-gb", 2)
    });

    // Allocate a register of 53 qubits
    auto qubitReg = xacc::qalloc(53);

    // Compile the XASM program
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(xasmSrcStr, qpu);
    auto program = ir->getComposites()[0];
    qpu->execute(qubitReg, program);
    qubitReg->print();

    xacc::Finalize();
    return 0;
}
