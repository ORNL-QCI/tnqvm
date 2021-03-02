#include "xacc.hpp"
#include "xacc_service.hpp"

int main(int argc, char **argv) {
  xacc::Initialize();
  xacc::set_verbose(true);
  // xacc::logToFile(true);
  // xacc::setLoggingLevel(2);
  // Number of qubits
  constexpr int NB_QUBITS = 4;
  constexpr int NB_ELECTRONS = NB_QUBITS / 2;
  // Using UCCSD as the base ansatz circuit
  auto tmp = xacc::getService<xacc::Instruction>("uccsd");
  auto uccsd = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);
  uccsd->expand({{"ne", NB_ELECTRONS}, {"nq", NB_QUBITS}});
  auto provider = xacc::getIRProvider("quantum");
  for (int i = 0; i < NB_QUBITS; ++i) {
    uccsd->addInstruction(provider->createInstruction("Measure", i));
  }
  std::cout << uccsd->getVariables().size() << "\n";
  const std::vector<double> params(uccsd->getVariables().size(), 1.0);
  auto evaled_uccsd = uccsd->operator()(params);
  std::cout << evaled_uccsd->toString() << "\n";
  // Exatn-general visitor parameters:
  // Number of layers to perform reconstruction
  constexpr int NB_LAYERS = 10;
  constexpr double RECONSTRUCTION_TOL = 1e-4;
  // Max bond dim
  constexpr int MAX_BOND_DIM = 64;
  auto accelerator = xacc::getAccelerator(
      "tnqvm", {{"tnqvm-visitor", "exatn-gen"},
                {"reconstruct-layers", NB_LAYERS},
                {"reconstruct-tolerance", RECONSTRUCTION_TOL},
                {"max-bond-dim", MAX_BOND_DIM}});
  
	auto buffer = xacc::qalloc(NB_QUBITS);
	accelerator->execute(buffer, evaled_uccsd);
	buffer->print();
	xacc::Finalize();
  return 0;
}
