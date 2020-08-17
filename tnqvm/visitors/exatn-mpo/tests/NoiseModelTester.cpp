#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

TEST(NoiseModelTester, checkReadout)
{
    const std::string BACKEND_JSON_FILE = std::string(BACKEND_CONFIG_DIR) + "/ibmqx2.json";
    std::ifstream inFile;
    inFile.open(BACKEND_JSON_FILE);
    std::stringstream strStream;
    strStream << inFile.rdbuf();
    const std::string json = strStream.str();  
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void testPrep0(qbit q) {
        Measure(q[0]);
    })");

    auto program = ir->getComposite("testPrep0");
    auto accelerator = xacc::getAccelerator("tnqvm", { { "tnqvm-visitor", "exatn-pmps" }, { "backend-json", json }, { "shots", 8192 } });
    auto qreg = xacc::qalloc(1);
    accelerator->execute(qreg, program);
    qreg->print();
    // Expected error: prob_meas1_prep0 = 0.06499999999999995
    EXPECT_NEAR(qreg->computeMeasurementProbability("1"), 0.06499999999999995, 0.01);
} 

int main(int argc, char **argv) 
{
    xacc::Initialize();
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    xacc::Finalize();
    return ret;
} 