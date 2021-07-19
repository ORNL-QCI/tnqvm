#include <memory>
#include <fstream>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"
#include <fstream>

namespace {
std::string getBackendJson() {
  const std::string BACKEND_JSON_FILE =
      std::string(BACKEND_CONFIG_DIR) + "/ibmqx2.json";
  std::ifstream inFile;
  inFile.open(BACKEND_JSON_FILE);
  std::stringstream strStream;
  strStream << inFile.rdbuf();
  return strStream.str();
}
} // namespace

TEST(ExaTnPmpsTester, checkSimple) {
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q) {
    // Run X gate 100 times, i.e. mimic randomized benchmarking.
    for (int i = 0; i < 100; i++) {
      X(q[0]);
    }
    Measure(q[0]);
  })");

  auto program = ir->getComposite("test1");
  auto accelerator =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-pmps"},
                                     {"backend-json", getBackendJson()}});
  auto qreg = xacc::qalloc(1);
  accelerator->execute(qreg, program);
  // Because of amplitude damping,
  // we won't get a perfect |0> state
  qreg->print();
  // There should be a non-zero count of "1" measurements.
  EXPECT_GT(qreg->computeMeasurementProbability("1"), 0.01);
}

TEST(ExaTnPmpsTester, checkMultipleQubits) {
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void test2(qbit q) {
    X(q[0]);
    X(q[1]);
    X(q[2]);
    Measure(q[0]);
    Measure(q[1]);
    Measure(q[2]);
  })");

  auto program = ir->getComposite("test2");
  auto accelerator =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-pmps"},
                                     {"backend-json", getBackendJson()}});
  auto qreg = xacc::qalloc(3);
  accelerator->execute(qreg, program);
  qreg->print();
  // |111> is the most common readout but not 100% due to noise.
  EXPECT_GT(qreg->computeMeasurementProbability("111"), 0.8);
  EXPECT_LT(qreg->computeMeasurementProbability("111"), 0.99);
}

TEST(ExaTnPmpsTester, checkBell) {
  auto xasmCompiler = xacc::getCompiler("xasm");
  auto ir = xasmCompiler->compile(R"(__qpu__ void testBell(qbit q) {
    X(q[0]);
    CX(q[0],q[1]);
    Measure(q[0]);
    Measure(q[1]);
  })");

  auto program = ir->getComposite("testBell");
  auto accelerator =
      xacc::getAccelerator("tnqvm", {{"tnqvm-visitor", "exatn-pmps"},
                                     {"backend-json", getBackendJson()}});
  auto qreg = xacc::qalloc(2);
  accelerator->execute(qreg, program);
  qreg->print();
  EXPECT_GT(qreg->computeMeasurementProbability("11"), 0.8);
  EXPECT_LT(qreg->computeMeasurementProbability("11"), 0.99);
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}