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
        // Use the "greed" contraction seq. optimizer
        // TODO: switch to the "metis" optimizer once it is fixed.
        std::make_pair("exatn-contract-seq-optimizer", "greed")
    });

    auto qreg = xacc::qalloc(NB_QUBITS);
    const auto start = std::chrono::system_clock::now();
    accelerator->execute(qreg, randomCirc);
    const auto end = std::chrono::system_clock::now();
    std::cout << "Elapsed time in milliseconds : " 
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    << " ms\n";
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 