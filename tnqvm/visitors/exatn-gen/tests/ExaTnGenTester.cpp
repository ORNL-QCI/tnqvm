#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "NoiseModel.hpp"
#include "Optimizer.hpp"
#include "xacc_observable.hpp"
#include "Algorithm.hpp"

TEST(ExaTnGenTester, checkSimple) {
  xacc::set_verbose(true);
  auto accelerator =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen"}});

  auto xasmCompiler = xacc::getCompiler("xasm");
  auto program = xasmCompiler
                     ->compile(R"(__qpu__ void test1(qbit q) {
        X(q[0]);
        X(q[0]);
        Measure(q[0]);
      })",
                               nullptr)
                     ->getComposites()[0];

  auto buffer = xacc::qalloc(1);
  accelerator->execute(buffer, program);
  buffer->print();
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}