// Simulate Sycamore circuit with MPS
#include "xacc.hpp"
#include <iostream>
#include <fstream>
#include <numeric>

std::string bitStringVecToString(const std::vector<int> &in_vec) {
  std::stringstream s;
  for (const auto &bit : in_vec)
    s << bit;
  return s.str();
}

int main(int argc, char **argv) {
  xacc::Initialize();
  xacc::set_verbose(true);
  // Options: 1, 2, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20
  const int CIRCUIT_DEPTH = 1;

  // Construct the full path to the XASM source file
  const std::string XASM_SRC_FILE = std::string(RESOURCE_DIR) +
                                    "/sycamore_53_" +
                                    std::to_string(CIRCUIT_DEPTH) + "_0.xasm";
  // Read XASM source
  std::ifstream inFile;
  inFile.open(XASM_SRC_FILE);
  std::stringstream strStream;
  strStream << inFile.rdbuf();
  const std::string kernelName = "sycamoreCirc";
  std::string xasmSrcStr = strStream.str();
  // Construct a unique kernel name:
  const std::string newKernelName =
      kernelName + "_" + std::to_string(CIRCUIT_DEPTH);
  xasmSrcStr.replace(xasmSrcStr.find(kernelName), kernelName.length(),
                     newKernelName);

  // The bitstring to calculate amplitude
  // Example: bitstring = 000000000...00
  const std::vector<int> BIT_STRING(53, 0);

  // ExaTN MPS visitor
  auto qpu = xacc::getAccelerator("tnqvm",
                                  {std::make_pair("tnqvm-visitor", "exatn-mps"),
                                   std::make_pair("bitstring", BIT_STRING),
                                   // Cap the bond dimension
                                   std::make_pair("max-bond-dim", 256),
                                   std::make_pair("exatn-buffer-size-gb", 8)});

  // Allocate a register of 53 qubits
  auto qubitReg = xacc::qalloc(53);

  // Compile the XASM program
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(xasmSrcStr, qpu);
  auto program = ir->getComposites()[0];
  qpu->execute(qubitReg, program);

  // qubitReg->print();
  // Rank-0
  if (qubitReg->hasExtraInfoKey("amplitude-real")) {
    const double realAmpl = (*qubitReg)["amplitude-real"].as<double>();
    const double imagAmpl = (*qubitReg)["amplitude-imag"].as<double>();

    // qflex style output:
    const std::vector<int> INITIAL_STATE_BIT_STRING(53, 0);
    std::cout << "================= RESULT =================\n";
    std::cout << bitStringVecToString(INITIAL_STATE_BIT_STRING);
    std::cout << " --> ";
    std::cout << bitStringVecToString(BIT_STRING);
    std::cout << ": " << realAmpl << " " << imagAmpl << "\n";
    std::cout << "Bit-string probability: "
              << sqrt(realAmpl * realAmpl + imagAmpl * imagAmpl) << "\n";
  }
  
  xacc::Finalize();
  return 0;
}
