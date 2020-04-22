#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

TEST(MpsMeasurementTester, checkSimple) 
{    
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q) {
        H(q[0]);
        H(q[1]);
        H(q[2]);
        CNOT(q[0], q[3]);
        CNOT(q[2], q[1]);
        CNOT(q[1], q[3]);
    })");

    auto program = ir->getComposite("test1");
    auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 1)});
    auto qreg = xacc::qalloc(5);
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