#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

TEST(ExaTnPmpsTester, checkSimple)
{
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q) {
    H(q[0]);
    CNOT(q[1], q[0]);
    Measure(q[0]);
    Measure(q[1]);
  })");

  auto program = ir->getComposite("test1");
  auto accelerator = xacc::getAccelerator("tnqvm", { std::make_pair("tnqvm-visitor", "exatn-pmps") });
  auto qreg = xacc::qalloc(2);
  accelerator->execute(qreg, program);
} 

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 