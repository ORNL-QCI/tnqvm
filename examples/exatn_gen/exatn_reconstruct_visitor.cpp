#include "xacc.hpp"
#include "xacc_service.hpp"
#include "NoiseModel.hpp"
#include "Optimizer.hpp"
#include "xacc_observable.hpp"
#include "Algorithm.hpp"

int benchmarkExaTnGen1()
{
 auto accelerator = xacc::getAccelerator("tnqvm",
                    {{"tnqvm-visitor", "exatn-gen"},
                     {"reconstruct-layers", 10}});
 xacc::qasm(R"(.compiler xasm
               .circuit deuteron_ansatz_h3_2
               .parameters t0, t1
               .qbit q
               X(q[0]);
               exp_i_theta(q, t0, {{"pauli", "X0 Y1 - Y0 X1"}});
               exp_i_theta(q, t1, {{"pauli", "X0 Z1 Y2 - X2 Z1 Y0"}});
              )");
 auto ansatz = xacc::getCompiled("deuteron_ansatz_h3_2");
 auto H_N_3 = xacc::quantum::getObservable("pauli",
               std::string("5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1 + "
                           "9.625 - 9.625 Z2 - 3.91 X1 X2 - 3.91 Y1 Y2"));
 auto optimizer = xacc::getOptimizer("nlopt");
 auto buffer = xacc::qalloc(3);
 auto vqe = xacc::getAlgorithm("vqe");
 vqe->initialize({std::make_pair("ansatz", ansatz),
                  std::make_pair("observable", H_N_3),
                  std::make_pair("accelerator", accelerator),
                  std::make_pair("optimizer", optimizer)});
 auto energies = vqe->execute(buffer, {0.0684968, 0.17797});
 buffer->print();
 std::cout << "Energy = " << energies[0] << " VS correct = " << -2.04482 << "\n";
 //EXPECT_NEAR(energies[0], -2.04482, 0.1);
 return 0;
}


int benchmarkExaTnGen2()
{
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

 auto accelerator_exatn_exact = xacc::getAccelerator("tnqvm",
      {{"tnqvm-visitor", "exatn:double"}
      ,{"bitstring", std::vector<int>(NB_QUBITS,-1)}
      ,{"exatn-buffer-size-gb", 2}
    //,{"exatn-contract-seq-optimizer", "cotengra"}
      });
 auto buffer1 = xacc::qalloc(NB_QUBITS);
 accelerator_exatn_exact->execute(buffer1, evaled_uccsd);
 buffer1->print();

 constexpr int NB_LAYERS = 5;
 constexpr double RECONSTRUCTION_TOL = 1e-4;
 constexpr int MAX_BOND_DIM = 64;
 auto accelerator_exatn_approx = xacc::getAccelerator("tnqvm",
      {{"tnqvm-visitor", "exatn-gen"},
       {"reconstruct-layers", NB_LAYERS},
       {"reconstruct-tolerance", RECONSTRUCTION_TOL},
       {"max-bond-dim", MAX_BOND_DIM}});
 auto buffer2 = xacc::qalloc(NB_QUBITS);
 accelerator_exatn_approx->execute(buffer2, evaled_uccsd);
 buffer2->print();
 return 0;
}


int main(int argc, char **argv) {
 xacc::Initialize(argc, argv);
 xacc::set_verbose(true);
 //xacc::logToFile(true);
 //xacc::setLoggingLevel(2);
 int error_code = 0;
 //if(error_code == 0) error_code = benchmarkExaTnGen1();
 if(error_code == 0) error_code = benchmarkExaTnGen2();
 xacc::Finalize();
 return error_code;
}
