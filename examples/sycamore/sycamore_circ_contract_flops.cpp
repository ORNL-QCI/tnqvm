// Calculate the *expected* flops required to contract the tensor network
#include "xacc.hpp"
#include<iostream>
#include<fstream>

int main(int argc, char **argv) 
{
    xacc::Initialize();
    // Circuit depth: allowed values = { 4, 5, 6, 8, 10, 12, 14, 16, 18, 20 }
    const int CIRCUIT_DEPTH = 4;    
    // Construct the full path to the XASM source file
    const std::string XASM_SRC_FILE = std::string(RESOURCE_DIR) + "/sycamore_53_" + std::to_string(CIRCUIT_DEPTH) + "_0.xasm";
    // Read XASM source
    std::ifstream inFile;
    inFile.open(XASM_SRC_FILE); 
    std::stringstream strStream;
    strStream << inFile.rdbuf(); 
    const std::string xasmSrcStr = strStream.str(); 

    // ExaTN visitor
    auto qpu = xacc::getAccelerator("tnqvm", { std::make_pair("tnqvm-visitor", "exatn") });
    // Allocate a register of 53 qubits
    auto qubitReg = xacc::qalloc(53);

    // Compile the XASM program
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(xasmSrcStr, qpu);
    // Make sure we can compile the XASM source:
    std::cout << "Compiled: \n" << ir->getComposites()[0]->toString() << "\n\n";
    xacc::Finalize();
    return 0;
} 
