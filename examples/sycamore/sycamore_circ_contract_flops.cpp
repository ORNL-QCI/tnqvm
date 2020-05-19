// Calculate the *expected* flops required to contract the tensor network
#include "xacc.hpp"
#include <iostream>
#include <fstream>
#include <numeric>

int main(int argc, char **argv) 
{
    xacc::Initialize();

    // ExaTN optimizers:
    const std::vector<std::string> OPTIMIZER_NAMES { "metis", "greed" };

    // Circuit depths
    const std::vector<int> CIRCUIT_DEPTHS { 4, 5, 6, 8, 10, 12, 14, 16, 18, 20 };  
    
    // Collect Flops data for all combinations (circuit depth + optimizer)
    for (const auto& optimizer : OPTIMIZER_NAMES)
    {
        std::cout << "************************************************\n";
        std::cout << "                   "  << optimizer << "\n";
        std::cout << "************************************************\n";
        for (const auto& depth : CIRCUIT_DEPTHS)
        {
            // Construct the full path to the XASM source file
            const std::string XASM_SRC_FILE = std::string(RESOURCE_DIR) + "/sycamore_53_" + std::to_string(depth) + "_0.xasm";
            // Read XASM source
            std::ifstream inFile;
            inFile.open(XASM_SRC_FILE); 
            std::stringstream strStream;
            strStream << inFile.rdbuf(); 
            const std::string kernelName = "sycamoreCirc";
            std::string xasmSrcStr = strStream.str(); 
            // Construct a unique kernel name: 
            const std::string newKernelName = kernelName + "_" + optimizer + "_" + std::to_string(depth);
            xasmSrcStr.replace(xasmSrcStr.find(kernelName), kernelName.length(), newKernelName);

            // ExaTN visitor
            auto qpu = xacc::getAccelerator("tnqvm", { 
                std::make_pair("tnqvm-visitor", "exatn"),
                // We only want to calculate the theoretical flops
                // required to do the tensor contraction,
                // i.e., don't actually contract the tensor network. 
                std::make_pair("calc-contract-cost-flops", true),
                // Using the corresponding ExaTN optimizer
                std::make_pair("exatn-contract-seq-optimizer", optimizer) 
            });

            // Allocate a register of 53 qubits
            auto qubitReg = xacc::qalloc(53);

            // Compile the XASM program
            auto xasmCompiler = xacc::getCompiler("xasm");
            auto ir = xasmCompiler->compile(xasmSrcStr, qpu);
            auto program = ir->getComposites()[0];
            // Execute: this will calculate the Flops requirement for tensor network contraction.
            qpu->execute(qubitReg, program);

            const std::vector<double> bitStringFlops = (*qubitReg)["bitstring-contract-flops"].as<std::vector<double>>();
            // For bitstring flops, we sum all the flops.
            const double bitStringFlopsTotal = std::accumulate(bitStringFlops.begin(), bitStringFlops.end(), 0.0);
            const std::vector<double> bitStringMemBytes = (*qubitReg)["bitstring-max-node-bytes"].as<std::vector<double>>();
            // For bitstring memory, we take the max value.
            const double bitStringMemMax = *std::max_element(bitStringMemBytes.begin(), bitStringMemBytes.end());

            // Print out the data:
            std::cout << ">> Depth = " << depth << "\n";
            const double flops = (*qubitReg)["contract-flops"].as<double>();
            const double mem = (*qubitReg)["max-node-bytes"].as<double>();
            const int elapsedMs = (*qubitReg)["optimizer-elapsed-time-ms"].as<int>();
            std::cout << " ==== Amplitude Calculation === \n";
            std::cout << "     - Flops = " << std::scientific << flops << "\n";
            std::cout << "     - Memory = " << std::scientific << mem << " [bytes] \n";
            std::cout << "     - Elapsed time = " << elapsedMs << " [ms]\n";
            std::cout << " ==== Bitstring Projection === \n";
            std::cout << "     - Total Flops = " << std::scientific << bitStringFlopsTotal << "\n";
            std::cout << "     - Max Memory = " << std::scientific << bitStringMemMax << " [bytes] \n";
        }
    }

    xacc::Finalize();
    return 0;
} 
