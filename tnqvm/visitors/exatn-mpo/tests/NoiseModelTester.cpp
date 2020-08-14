#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

TEST(NoiseModelTester, checkSimple)
{
    const std::string BACKEND_JSON_FILE = std::string(BACKEND_CONFIG_DIR) + "/ibmqx2.json";
    std::ifstream inFile;
    inFile.open(BACKEND_JSON_FILE);
    std::stringstream strStream;
    strStream << inFile.rdbuf();
    const std::string json = strStream.str();  
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void testBell(qbit q) {
        X(q[0]);
        CX(q[0],q[1]);
        Measure(q[0]);
        Measure(q[1]);
    })");

    auto program = ir->getComposite("testBell");
    auto accelerator = xacc::getAccelerator("tnqvm", { { "tnqvm-visitor", "exatn-pmps" }, { "backend-json", json } });
    auto qreg = xacc::qalloc(2);
    accelerator->execute(qreg, program);
    qreg->print();
} 

int main(int argc, char **argv) 
{
    xacc::Initialize();
    ::testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    xacc::Finalize();
    return ret;
} 