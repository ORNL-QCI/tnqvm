#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

// Validate the distribution
TEST(NumericalTester, checkDistribution) 
{    
    auto tmp = xacc::getService<xacc::Instruction>("rcs");
    auto randomCirc = std::dynamic_pointer_cast<xacc::CompositeInstruction>(tmp);
    const int NB_QUBITS = 5;
    // Use a large number of shots to get true probability distribution.
    const int NB_SHOTS = 100000;
    EXPECT_TRUE(randomCirc->expand({std::make_pair("nq", NB_QUBITS), std::make_pair("nlayers", 15)}));
    std::cout << "Number of gates = " << randomCirc->nInstructions() << "\n";
    std::cout << "Circuit:\n" << randomCirc->toString() << "\n";

    // MPS simulation
    auto acceleratorMps = xacc::getAccelerator("tnqvm", {
        std::make_pair("tnqvm-visitor", "exatn-mps"), 
        std::make_pair("shots", NB_SHOTS),
        std::make_pair("exatn-logging-level", 1) 
    });
    auto qregMps = xacc::qalloc(NB_QUBITS);
    acceleratorMps->execute(qregMps, randomCirc);
    const double norm = (*qregMps)["norm"].as<double>();
    EXPECT_NEAR(norm, 1.0, 1e-6);
    qregMps->print();
    
    // ExaTN: direct tensor contraction (no MPS)
    auto acceleratorDirect = xacc::getAccelerator("tnqvm", {
        std::make_pair("tnqvm-visitor", "exatn"), 
        std::make_pair("shots", NB_SHOTS)
    });

    auto qregDirect = xacc::qalloc(NB_QUBITS);
    acceleratorDirect->execute(qregDirect, randomCirc);
    qregDirect->print();

    for (const auto& sampleBitString: qregMps->getMeasurements())
    {
        const double probMps = qregMps->computeMeasurementProbability(sampleBitString);
        const double probDirect = qregDirect->computeMeasurementProbability(sampleBitString);
        std::cout << sampleBitString << ": " << probMps << " vs. " << probDirect << "\n";
        EXPECT_NEAR(probMps, probDirect, 0.01);
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