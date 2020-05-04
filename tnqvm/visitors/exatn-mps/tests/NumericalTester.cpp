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

TEST(NumericalTester, checkNorm) 
{    
    auto tmp = xacc::getService<xacc::Instruction>("rcs");
    auto randomCirc = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);
    const int NB_QUBITS = 10;
    EXPECT_TRUE(randomCirc->expand({std::make_pair("nq", NB_QUBITS), std::make_pair("nlayers", 15)}));
    std::cout << "Number of gates = " << randomCirc->nInstructions() << "\n";
    std::cout << "Circuit:\n" << randomCirc->toString() << "\n";

    // No cut-off:
    {
        std::cout << "Testing no SVD cut-off limit!\n";
        auto accelerator = xacc::getAccelerator("tnqvm", {
            std::make_pair("tnqvm-visitor", "exatn-mps"), 
            std::make_pair("shots", 10000)
            // Uncomment to enable detail logging
            // std::make_pair("exatn-logging-level", 1)
        });

        auto qreg = xacc::qalloc(NB_QUBITS);
        accelerator->execute(qreg, randomCirc);
        const double norm = (*qreg)["norm"].as<double>();
        EXPECT_NEAR(norm, 1.0, 1e-6);
        //qreg->print();
    }
    // With SVD cut-off limit:
    {
        std::cout << "Testing SVD cut-off limit!\n";
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000), std::make_pair("svd-cutoff", 1e-6)});
        auto qreg = xacc::qalloc(NB_QUBITS);
        accelerator->execute(qreg, randomCirc);
        const double norm = (*qreg)["norm"].as<double>();
        EXPECT_NEAR(norm, 1.0, 1e-6);
        //qreg->print();
    }
    // ExaTN: direct tensor contraction (no MPS)
    {
        std::cout << "Testing full tensor contraction!\n";
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(NB_QUBITS);
        accelerator->execute(qreg, randomCirc);
        // qreg->print();
    }
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 