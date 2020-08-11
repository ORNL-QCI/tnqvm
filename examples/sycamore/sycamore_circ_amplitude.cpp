// Computes a single amplitude from a Sycamore circuit via full tensor contraction
#include "xacc.hpp"
#include <iostream>
#include <fstream>
#include <numeric>

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
    //xacc::set_verbose(true);
    //xacc::logToFile(true);
    //xacc::setLoggingLevel(2);

    // Options: 4, 5, 6, 8, 10, 12, 14, 16, 18, 20
    const int CIRCUIT_DEPTH = 8;

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

    // The bitstring to calculate amplitude
    // Example: bitstring = 000000000...00
    const std::vector<int> BIT_STRING(53, 0);

    // ExaTN visitor
    auto qpu = xacc::getAccelerator("tnqvm", {
        std::make_pair("tnqvm-visitor", "exatn"),
        std::make_pair("bitstring", BIT_STRING),
        std::make_pair("exatn-buffer-size-gb", 2)
    });

    // Allocate a register of 53 qubits
    auto qubitReg = xacc::qalloc(53);

    // Compile the XASM program
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(xasmSrcStr, qpu);
    auto program = ir->getComposites()[0];
    qpu->execute(qubitReg, program);
    // qubitReg->print();
    const double realAmpl = (*qubitReg)["amplitude-real"].as<double>();
    const double imagAmpl = (*qubitReg)["amplitude-imag"].as<double>();

    // qflex style output:
    std::cout << "================= RESULT =================\n";
    std::cout << bitStringVecToString(INITIAL_STATE_BIT_STRING);
    std::cout << " --> ";
    std::cout << bitStringVecToString(BIT_STRING);
    std::cout << ": " << realAmpl << " " << imagAmpl << "\n";
    std::cout << "Bit-string probability: " << sqrt(realAmpl*realAmpl + imagAmpl*imagAmpl) << "\n";

    xacc::Finalize();
    return 0;
}
