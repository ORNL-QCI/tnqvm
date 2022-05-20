#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_observable.hpp"

#include <fstream>
#include <string>

const std::string QASM_SRC_FILE = "qaoa_14q.qasm";
constexpr int N_QUBITS = 14;

int main(int argc, char **argv) {

    // Initialize:
    xacc::Initialize(argc, argv);
    xacc::set_verbose(true);
    //xacc::logToFile(true);
    //xacc::setLoggingLevel(1);

    // Get Accelerator backend:
    auto accelerator_qpp = xacc::getAccelerator("qpp");
    auto accelerator_tnqvm_gen = xacc::getAccelerator("tnqvm",
                                 {{"tnqvm-visitor", "exatn-gen:float"},
                                  {"exatn-buffer-size-gb", 2},
                                  {"reconstruct-layers", 4},
                                  {"reconstruct-tolerance", 1e-5},
                                  {"max-bond-dim", 32}});
    auto accelerator = accelerator_qpp;

    // Read Source file:
    std::ifstream inFile;
    inFile.open(QASM_SRC_FILE);
    std::stringstream strStream;
    strStream << inFile.rdbuf();
    const std::string qasmSrcStr = strStream.str();

    // Get Compiler:
    auto compiler = xacc::getCompiler("staq");

    // Compile:
    auto IR = compiler->compile(qasmSrcStr);
    auto program = IR->getComposites()[0];

    // Execute:
    auto buffer = xacc::qalloc(N_QUBITS);
    accelerator->execute(buffer, program);
    buffer->print();
    std::cout << "Result: " << buffer->getExpectationValueZ() << std::endl;

    // Finalize:
    xacc::Finalize();
}
