#include "xacc.hpp"
#include "xacc_service.hpp"
#include "xacc_observable.hpp"

#include <fstream>
#include <string>

const std::string QASM_SRC_FILE = "ASP_8q_5001.qasm";
constexpr int N_QUBITS = 8;

int main(int argc, char **argv) {

    // Initialize:
    xacc::Initialize(argc, argv);
    xacc::set_verbose(true);
    xacc::logToFile(true);
    xacc::setLoggingLevel(1);

    // Get Accelerator backend:
    auto accelerator_qpp = xacc::getAccelerator("qpp");
    auto accelerator_tnqvm_gen = xacc::getAccelerator("tnqvm",
                                 {{"tnqvm-visitor", "exatn-gen:double"},
                                  {"exatn-buffer-size-gb", 4},
                                  {"reconstruct-layers", 6},
                                  {"reconstruct-tolerance", 1e-6},
                                  {"max-bond-dim", 4},
                                  {"reconstruct-builder", "TTN"},
                                  {"exatn-contract-seq-optimizer", "cutnn"}});
    auto accelerator = accelerator_tnqvm_gen;

    // Read Source file:
    std::ifstream inFile;
    inFile.open(QASM_SRC_FILE);
    std::stringstream strStream;
    strStream << inFile.rdbuf();
    const std::string qasmSrcStr = strStream.str();

    // Get Compiler:
    auto compiler = xacc::getCompiler("staq");

    // Compile Circuit:
    auto IR = compiler->compile(qasmSrcStr);
    auto program = IR->getComposites()[0];

    // Define Observable:
    const std::string obs_str = [&](){
     std::string result = std::to_string(1.0/static_cast<double>(N_QUBITS)) + " Z0";
     for (int i = 1; i < N_QUBITS; ++i) {
      result += (" + " + std::to_string(1.0/static_cast<double>(N_QUBITS)) + " Z" + std::to_string(i));
     }
     return result;
    }();

    // Generate Composites for Observable:
    auto avg_mag = xacc::quantum::getObservable("pauli", obs_str);
    auto composites = avg_mag->observe(program);

    // Allocate Qubits and execute:
    auto buffer = xacc::qalloc(N_QUBITS);
    accelerator->execute(buffer, composites);
    buffer->print();
    const double avg_mag_result = avg_mag->postProcess(buffer);
    std::cout << "Result: " << avg_mag_result << std::endl;

    // Finalize:
    xacc::Finalize();
}
