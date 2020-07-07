// Simulate Sycamore circuit with MPS
#include "xacc.hpp"
#include <iostream>
#include <fstream>
#include <numeric>

int main(int argc, char **argv) 
{
    xacc::Initialize();
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

    // The bitstring to calculate amplitude
    // Example: bitstring = 000000000...00
    const std::vector<int> BIT_STRING(53, 0);

    // ExaTN visitor
    auto qpu = xacc::getAccelerator("tnqvm", { 
        std::make_pair("tnqvm-visitor", "exatn"),
        std::make_pair("bitstring", BIT_STRING)
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
