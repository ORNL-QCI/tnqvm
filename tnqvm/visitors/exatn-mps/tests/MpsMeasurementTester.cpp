#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"


TEST(MpsMeasurementTester, checkSimple) 
{    
    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q) {
            H(q[0]);
            for (int i = 0; i < 34; i++) {
                CNOT(q[i], q[i + 1]);
            }

            Measure(q[5]);
            Measure(q[3]);
            Measure(q[7]);
            Measure(q[34]);
        })");

        auto program = ir->getComposite("test1");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 1)});
        // Use a large enough qubit register to check memory constraint 
        auto qreg = xacc::qalloc(35);
        accelerator->execute(qreg, program);
        qreg->print();
        // We expect to create a entangle state (50 qubits): |000000 ... 00> + |111 .... 111>
        // We sample 4 qubits => either get 0000 or 1111 bit string.
        const auto prob0 = qreg->computeMeasurementProbability("0000");
        const auto prob1 = qreg->computeMeasurementProbability("1111");
        EXPECT_NEAR(prob0 + prob1, 1.0, 1e-12);
    }
}

TEST(MpsMeasurementTester, checkRandomSeed) 
{
  std::vector<std::map<std::string, int>> results;
  constexpr int NB_TESTS = 4;
  for (int i = 0; i < NB_TESTS; i++) {
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q) {
            H(q[0]);
            for (int i = 0; i < 3; i++) {
                CNOT(q[i], q[i + 1]);
            }

            Measure(q[0]);
            Measure(q[1]);
            Measure(q[2]);
            Measure(q[3]);
        })");

    auto program = ir->getComposite("test1");
    auto accelerator = xacc::getAccelerator(
        "tnqvm",
        {{"tnqvm-visitor", "exatn-mps"}, {"shots", 8192}, {"seed", 123}});
    auto qreg = xacc::qalloc(4);
    accelerator->execute(qreg, program);
    qreg->print();
    results.emplace_back(qreg->getMeasurementCounts());
  }
  EXPECT_TRUE(std::adjacent_find(results.begin(), results.end(),
                                 std::not_equal_to<>()) == results.end());
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 