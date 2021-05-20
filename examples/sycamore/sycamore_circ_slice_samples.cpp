// Computes a single amplitude from a Sycamore circuit via full tensor contraction
#include "xacc.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
#include <cassert>
// Initial state:
const std::vector<int> INITIAL_STATE_BIT_STRING(53, 0);

std::string bitStringVecToString(const std::vector<int>& in_vec)
{
    std::stringstream s;
    for (const auto& bit: in_vec) s << bit;
    return s.str();
}

int main(int argc, char **argv)
{
    xacc::Initialize();
    xacc::set_verbose(true);
    xacc::logToFile(true);
    xacc::setLoggingLevel(1);

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

    const int NB_OPEN_QUBITS = 21;
    // The bitstring to calculate amplitude
    // Example: bitstring = 000000000...00 (-1 -1)
    // there are NB_OPEN_QUBITS open legs at the end (-1) values
    std::vector<int> BIT_STRING(53, 0);
    std::fill_n(BIT_STRING.begin() + (BIT_STRING.size() - NB_OPEN_QUBITS),
                NB_OPEN_QUBITS, -1);
    // ExaTN visitor:
    // Note:
    // (1) "exatn" == "exatn:double" uses double (64-bit) type.
    // (1) "exatn:float" uses float (32-bit) type.
    const std::string OPTIMIZER_NAME = "cotengra"; // "cotengra"
    auto qpu = xacc::getAccelerator("tnqvm", {
        std::make_pair("tnqvm-visitor", "exatn"),
        std::make_pair("bitstring", BIT_STRING),
        std::make_pair("exatn-buffer-size-gb", 2),
        std::make_pair("exatn-contract-seq-optimizer", OPTIMIZER_NAME)
    });

    // Allocate a register of 53 qubits
    auto qubitReg = xacc::qalloc(53);

    // Compile the XASM program
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(xasmSrcStr, qpu);
    auto program = ir->getComposites()[0];
    qpu->execute(qubitReg, program);
    // qubitReg->print();
    const auto realAmpl = (*qubitReg)["amplitude-real-vec"].as<std::vector<double>>();
    const auto imagAmpl = (*qubitReg)["amplitude-imag-vec"].as<std::vector<double>>();
    // Open 21 legs
    assert(realAmpl.size() == (1ULL << NB_OPEN_QUBITS));
    assert(imagAmpl.size() == (1ULL << NB_OPEN_QUBITS));
    
    std::cout << "Slice vector of size: " << realAmpl.size() << "\n";
    xacc::Finalize();
    return 0;
}
