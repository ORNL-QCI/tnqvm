#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "NoiseModel.hpp"
#include "Optimizer.hpp"
#include "xacc_observable.hpp"
#include "Algorithm.hpp"

TEST(ExaTnGenTester, checkExpVal) {
  xacc::set_verbose(true);
  {
    // Check exp-val by tensor expansion
    auto accelerator =
        xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen"}});
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto program = xasmCompiler
                       ->compile(R"(__qpu__ void testRY(qbit q, double t) {
        Ry(q[0], t);
        Measure(q[0]);
      })",
                                 accelerator)
                       ->getComposite("testRY");
    const auto angles = xacc::linspace(-M_PI, M_PI, 20);
    for (const auto &angle : angles) {
      auto evaled = program->operator()({angle});
      const double expResult =
          -1.0 + 2.0 * std::cos(angle / 2.0) * std::cos(angle / 2.0);
      std::cout << "Angle = " << angle << ": " << expResult << "\n";
      auto buffer = xacc::qalloc(1);
      accelerator->execute(buffer, evaled);
      // buffer->print();
      std::cout << "Angle = " << angle << ": " << buffer->getExpectationValueZ() << " vs. " << expResult  << "\n";
      EXPECT_NEAR(buffer->getExpectationValueZ(), expResult, 1e-3);
    }
  }
  {
    auto accelerator =
        xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-gen"}});
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz(qbit q, double t) {
      X(q[0]);
      Ry(q[1], t);
      CX(q[1], q[0]);
      H(q[0]);
      H(q[1]);
      Measure(q[0]);
      Measure(q[1]);
    })",
                                    accelerator);

    auto program = ir->getComposite("ansatz");
    // Expected results from deuteron_2qbit_xasm_X0X1
    const std::vector<double> expectedResults{
        0.0,       -0.324699, -0.614213, -0.837166, -0.9694,
        -0.996584, -0.915773, -0.735724, -0.475947, -0.164595,
        0.164595,  0.475947,  0.735724,  0.915773,  0.996584,
        0.9694,    0.837166,  0.614213,  0.324699,  0.0};

    const auto angles =
        xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 20);
    for (size_t i = 0; i < angles.size(); ++i) {
      auto buffer = xacc::qalloc(2);
      auto evaled = program->operator()({angles[i]});
      accelerator->execute(buffer, evaled);
      std::cout << "Angle = " << angles[i]
                << ": Expected = " << expectedResults[i]
                << "; ExaTN = " << buffer->getExpectationValueZ() << "\n";
      EXPECT_NEAR(buffer->getExpectationValueZ(), expectedResults[i], 1e-6);
    }
  }
}

int main(int argc, char **argv) {
  xacc::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}