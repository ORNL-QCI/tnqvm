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
        H(q[0]);
        CNOT(q[0], q[1]);
        Rx(q[0], pi/2);
      })",
                               nullptr)
                     ->getComposites()[0];

  auto buffer = xacc::qalloc(2);
  accelerator->execute(buffer, program);
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}