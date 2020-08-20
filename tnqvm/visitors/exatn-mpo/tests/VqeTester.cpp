#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

// Test Vqe run with noise
TEST(VqeTester, checkSimple)
{
    const std::string BACKEND_JSON_FILE = std::string(BACKEND_CONFIG_DIR) + "/ibmqx2.json";
    std::ifstream inFile;
    inFile.open(BACKEND_JSON_FILE);
    std::stringstream strStream;
    strStream << inFile.rdbuf();
    const std::string json = strStream.str();  
    auto accelerator = xacc::getAccelerator("tnqvm", { { "tnqvm-visitor", "exatn-pmps" }, { "backend-json", json }, { "shots", 8192 } });
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz(qbit q, double t) {
      X(q[1]);
      Ry(q[0], t);
      CX(q[0], q[1]);
      H(q[1]);
      H(q[0]);
      Measure(q[0]);
      Measure(q[1]);
    })", accelerator);

    auto program = ir->getComposite("ansatz");
    // Expected results from deuteron_2qbit_xasm_X0X1
    const std::vector<double> expectedResults {
        0.0,
        -0.324699,
        -0.614213,
        -0.837166,
        -0.9694,
        -0.996584,
        -0.915773,
        -0.735724,
        -0.475947,
        -0.164595,
        0.164595,
        0.475947,
        0.735724,
        0.915773,
        0.996584,
        0.9694,
        0.837166,
        0.614213,
        0.324699,
        0.0
    };

    const auto angles = xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 20);
    for (size_t i = 0; i < angles.size(); ++i)
    {
        auto buffer = xacc::qalloc(2);
        auto evaled = program->operator()({ angles[i] });
        accelerator->execute(buffer, evaled);
        // Use a large tolerance due to noise
        EXPECT_NEAR(buffer->getExpectationValueZ(), expectedResults[i], 0.2);
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