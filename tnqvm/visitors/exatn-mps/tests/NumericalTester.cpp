#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

TEST(NumericalTester, checkRandomCircuitGen) 
{    
    auto tmp = xacc::getService<xacc::Instruction>("rcs");
    auto randomCirc = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);

    EXPECT_TRUE(randomCirc->expand({std::make_pair("nq", 30), std::make_pair("nlayers", 3)}));
    // std::cout << "HELLO\n" << randomCirc->toString() << "\n";
}

TEST(NumericalTester, checkNormNoCutoff) 
{    
    auto tmp = xacc::getService<xacc::Instruction>("rcs");
    auto randomCirc = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);
    const int NB_QUBITS = 10;
    EXPECT_TRUE(randomCirc->expand({std::make_pair("nq", NB_QUBITS), std::make_pair("nlayers", 15)}));
    std::cout << "Number of gates = " << randomCirc->nInstructions() << "\n";
    auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 1)});
    auto qreg = xacc::qalloc(NB_QUBITS);
    accelerator->execute(qreg, randomCirc);
    const double norm = (*qreg)["norm"].as<double>();
    EXPECT_NEAR(norm, 1.0, 1e-6);
}

TEST(NumericalTester, checkNormCutoff) 
{    
    auto tmp = xacc::getService<xacc::Instruction>("rcs");
    auto randomCirc = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);
    const int NB_QUBITS = 10;
    EXPECT_TRUE(randomCirc->expand({std::make_pair("nq", NB_QUBITS), std::make_pair("nlayers", 15)}));
    std::cout << "Number of gates = " << randomCirc->nInstructions() << "\n";
    auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 1), std::make_pair("svd-cutoff", 1e-6)});
    auto qreg = xacc::qalloc(NB_QUBITS);
    std::cout << "Circuit:\n" << randomCirc->toString() << "\n";
    accelerator->execute(qreg, randomCirc);
    qreg->print();
    // const double norm = (*qreg)["norm"].as<double>();
    // EXPECT_NEAR(norm, 1.0, 1e-6);
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 