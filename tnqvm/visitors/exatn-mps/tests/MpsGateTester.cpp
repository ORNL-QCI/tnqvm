#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"


TEST(MpsGateTester, checkSimple) 
{    
    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q) {
            H(q[0]);
            Measure(q[0]);
            Measure(q[1]);
            Measure(q[2]);
            Measure(q[3]);
        })");

        auto program = ir->getComposite("test1");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        // qreg->print();
        // 50-50 distribution (Q0)
        EXPECT_NEAR(qreg->computeMeasurementProbability("0000"), 0.5, 0.01);
        EXPECT_NEAR(qreg->computeMeasurementProbability("1000"), 0.5, 0.01);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void test2(qbit q) {
            H(q[1]);
            Measure(q[0]);
            Measure(q[1]);
            Measure(q[2]);
            Measure(q[3]);
        })");

        auto program = ir->getComposite("test2");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        // qreg->print();
        // 50-50 distribution (Q1)
        EXPECT_NEAR(qreg->computeMeasurementProbability("0000"), 0.5, 0.01);
        EXPECT_NEAR(qreg->computeMeasurementProbability("0100"), 0.5, 0.01);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void test3(qbit q) {
            H(q[2]);
            Measure(q[0]);
            Measure(q[1]);
            Measure(q[2]);
            Measure(q[3]);
        })");

        auto program = ir->getComposite("test3");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        // qreg->print();
        // 50-50 distribution (Q2)
        EXPECT_NEAR(qreg->computeMeasurementProbability("0000"), 0.5, 0.01);
        EXPECT_NEAR(qreg->computeMeasurementProbability("0010"), 0.5, 0.01);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void test4(qbit q) {
            H(q[3]);
            Measure(q[0]);
            Measure(q[1]);
            Measure(q[2]);
            Measure(q[3]);
        })");

        auto program = ir->getComposite("test4");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        // qreg->print();
        // 50-50 distribution (Q3)
        EXPECT_NEAR(qreg->computeMeasurementProbability("0000"), 0.5, 0.01);
        EXPECT_NEAR(qreg->computeMeasurementProbability("0001"), 0.5, 0.01);
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
