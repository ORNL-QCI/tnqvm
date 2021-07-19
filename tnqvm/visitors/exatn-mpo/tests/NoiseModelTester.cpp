#include <memory>
#include <fstream>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"
#include <fstream>

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
    // Expected error: prob_meas1_prep0 = 0.01539999999999997
    EXPECT_NEAR(qreg->computeMeasurementProbability("1"), 0.01539999999999997, 0.01);
}

TEST(NoiseModelTester, checkGateTime)
{
    const std::string BACKEND_JSON_FILE = std::string(BACKEND_CONFIG_DIR) + "/ibmqx2.json";
    std::ifstream inFile;
    inFile.open(BACKEND_JSON_FILE);
    std::stringstream strStream;
    strStream << inFile.rdbuf();
    const std::string json = strStream.str();
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto program1 = xasmCompiler->compile(R"(__qpu__ void testXGate(qbit q) {
        X(q[0]);
        Measure(q[0]);
    })")->getComposite("testXGate");

    // H-Z-H is equivalent to X:
    // Note: H gate is a *half-length* gate (u2); Z gate is always *noiseless*.
    // Hence, although there are 3 gates in this circuit, the result is equivalent to
    // that of a single X gate.
    auto program2 = xasmCompiler->compile(R"(__qpu__ void testXGateTransform(qbit q) {
        H(q[0]);
        Z(q[0]);
        H(q[0]);
        Measure(q[0]);
    })")->getComposite("testXGateTransform");

    auto accelerator = xacc::getAccelerator("tnqvm", { { "tnqvm-visitor", "exatn-pmps" }, { "backend-json", json }, { "shots", 8192 } });
    auto qreg1 = xacc::qalloc(1);
    accelerator->execute(qreg1, program1);
    qreg1->print();

    auto qreg2 = xacc::qalloc(1);
    accelerator->execute(qreg2, program2);
    qreg2->print();
    EXPECT_NEAR(qreg1->computeMeasurementProbability("1"), qreg2->computeMeasurementProbability("1"), 0.01);
}

TEST(NoiseModelTester, checkGateError)
{
    const std::string BACKEND_JSON_FILE = std::string(BACKEND_CONFIG_DIR) + "/ibmqx2.json";
    std::ifstream inFile;
    inFile.open(BACKEND_JSON_FILE);
    std::stringstream strStream;
    strStream << inFile.rdbuf();
    const std::string json = strStream.str();
    // we can compute the number of gates that results in a completely mixed state.
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto program1 = xasmCompiler->compile(R"(__qpu__ void testRb(qbit q) {
        for (int i = 0; i < 200; i++) {
            X(q[0]);
        }
        Measure(q[0]);
    })")->getComposite("testRb");

    auto accelerator = xacc::getAccelerator("tnqvm", { { "tnqvm-visitor", "exatn-pmps" }, { "backend-json", json }, { "shots", 8192 } });
    auto qreg1 = xacc::qalloc(1);
    accelerator->execute(qreg1, program1);
    qreg1->print();
    // The error should be very high at this point.
    // Note: due to relaxation, we won't be able to get a perfect 50-50 completely mixed state.
    EXPECT_GT(qreg1->computeMeasurementProbability("1"), 0.1);
}

int main(int argc, char **argv)
{
    xacc::Initialize();
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    xacc::Finalize();
    return ret;
}
