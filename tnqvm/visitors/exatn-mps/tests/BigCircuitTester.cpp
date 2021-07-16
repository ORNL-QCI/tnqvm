#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

TEST(BigCircuitTester, checkBitStringSampling) 
{    
    auto tmp = xacc::getService<xacc::Instruction>("rcs");
    auto randomCirc = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);
    // Sycamore set-up: 53 qubits
    const int NB_QUBITS = 53;
    EXPECT_TRUE(randomCirc->expand({
        std::make_pair("nq", NB_QUBITS), 
        // TODO: we should increase this to 8-12 layers
        // once the ExaTN backend is updated to cap intermediate node size.
        std::make_pair("nlayers", 1), 
        std::make_pair("parametric-gates", false)
    }));
    std::cout << "Number of gates = " << randomCirc->nInstructions() << "\n";
    std::cout << "Circuit:\n" << randomCirc->toString() << "\n";

    auto accelerator = xacc::getAccelerator("tnqvm", {
        std::make_pair("tnqvm-visitor", "exatn"), 
        // Just produce one shot
        std::make_pair("shots", 1),
        // Use the default "metis" contraction seq. optimizer
        std::make_pair("exatn-contract-seq-optimizer", "metis"),
        // Just perform a dry-run rather than contracting the network.
        // This will save test time.
        std::make_pair("calc-contract-cost-flops", true)
    });

    auto qreg = xacc::qalloc(NB_QUBITS);
    const auto start = std::chrono::system_clock::now();
    accelerator->execute(qreg, randomCirc);
    const auto end = std::chrono::system_clock::now();
    std::cout << "Elapsed time in milliseconds : " 
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    << " ms\n";
}

TEST(BigCircuitTester, checkDeutschJozsa) {
  const auto generateDJ16 = []() {
    return (R"(__qpu__ void QBCIRCUIT(qreg q) {
  OPENQASM 2.0;
  include "qelib1.inc";
  creg c[16];
  x q[16];
  h q[0];
  h q[1];
  h q[2];
  h q[3];
  h q[4];
  h q[5];
  h q[6];
  h q[7];
  h q[8];
  h q[9];
  h q[10];
  h q[11];
  h q[12];
  h q[13];
  h q[14];
  h q[15];
  h q[16];
  x q[0];
  x q[2];
  x q[4];
  x q[6];
  x q[8];
  x q[10];
  x q[12];
  x q[14];
  cx q[0],q[16];
  cx q[1],q[16];
  cx q[2],q[16];
  cx q[3],q[16];
  cx q[4],q[16];
  cx q[5],q[16];
  cx q[6],q[16];
  cx q[7],q[16];
  cx q[8],q[16];
  cx q[9],q[16];
  cx q[10],q[16];
  cx q[11],q[16];
  cx q[12],q[16];
  cx q[13],q[16];
  cx q[14],q[16];
  cx q[15],q[16];
  x q[0];
  x q[2];
  x q[4];
  x q[6];
  x q[8];
  x q[10];
  x q[12];
  x q[14];
  h q[0];
  h q[1];
  h q[2];
  h q[3];
  h q[4];
  h q[5];
  h q[6];
  h q[7];
  h q[8];
  h q[9];
  h q[10];
  h q[11];
  h q[12];
  h q[13];
  h q[14];
  h q[15];
  measure q[0] -> c[0];
  measure q[1] -> c[1];
  measure q[2] -> c[2];
  measure q[3] -> c[3];
  measure q[4] -> c[4];
  measure q[5] -> c[5];
  measure q[6] -> c[6];
  measure q[7] -> c[7];
  measure q[8] -> c[8];
  measure q[9] -> c[9];
  measure q[10] -> c[10];
  measure q[11] -> c[11];
  measure q[12] -> c[12];
  measure q[13] -> c[13];
  measure q[14] -> c[14];
  measure q[15] -> c[15];
 }
            )");
  };
  auto qubitReg = xacc::qalloc(17);
  qubitReg->setName("q");
  xacc::storeBuffer(qubitReg);
  auto qasmCompiler = xacc::getCompiler("staq");
  auto ir = qasmCompiler->compile(generateDJ16(), nullptr);
  auto qpu = xacc::getAccelerator(
      "tnqvm:exatn-mps", {{"max-bond-dim", 5000}, {"svd-cutoff", 1.0e-12}});
  auto program = ir->getComposites()[0];
  qpu->execute(qubitReg, program);
  qubitReg->print();
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 