#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

TEST(BigCircuitTester, checkMps) 
{    
    auto tmp = xacc::getService<xacc::Instruction>("rcs");
    auto randomCirc = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);
    const int NB_QUBITS = 25;
    EXPECT_TRUE(randomCirc->expand({
        std::make_pair("nq", NB_QUBITS), 
        std::make_pair("nlayers", 12), 
        std::make_pair("parametric-gates", false)
    }));
    std::cout << "Number of gates = " << randomCirc->nInstructions() << "\n";
    std::cout << "Circuit:\n" << randomCirc->toString() << "\n";

    auto accelerator = xacc::getAccelerator("tnqvm", {
        std::make_pair("tnqvm-visitor", "exatn"), 
        std::make_pair("shots", 1)
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