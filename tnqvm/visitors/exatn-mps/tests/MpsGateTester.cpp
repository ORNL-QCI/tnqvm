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
        EXPECT_NEAR(qreg->computeMeasurementProbability("0000"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("1000"), 0.5, 0.05);
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
        EXPECT_NEAR(qreg->computeMeasurementProbability("0000"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("0100"), 0.5, 0.05);
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
        EXPECT_NEAR(qreg->computeMeasurementProbability("0000"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("0010"), 0.5, 0.05);
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
        EXPECT_NEAR(qreg->computeMeasurementProbability("0000"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("0001"), 0.5, 0.05);
    }
}

TEST(MpsGateTester, checkTwoQubitGates) 
{    
    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void foo1(qbit q) {
            H(q[0]);
            CNOT(q[0], q[1]);
            Measure(q[0]);
            Measure(q[1]);
        })");

        auto program = ir->getComposite("foo1");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("00"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("11"), 0.5, 0.05);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void foo2(qbit q) {
            H(q[1]);
            CNOT(q[1], q[0]);
            Measure(q[0]);
            Measure(q[1]);
        })");

        auto program = ir->getComposite("foo2");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("00"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("11"), 0.5, 0.05);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void foo3(qbit q) {
            H(q[1]);
            CNOT(q[1], q[2]);
            Measure(q[1]);
            Measure(q[2]);
        })");

        auto program = ir->getComposite("foo3");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("00"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("11"), 0.5, 0.05);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void foo4(qbit q) {
            H(q[2]);
            CNOT(q[2], q[1]);
            Measure(q[2]);
            Measure(q[1]);
        })");

        auto program = ir->getComposite("foo4");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        EXPECT_NEAR(qreg->computeMeasurementProbability("00"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("11"), 0.5, 0.05);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void foo5(qbit q) {
            H(q[2]);
            CNOT(q[2], q[3]);
            Measure(q[2]);
            Measure(q[3]);
        })");

        auto program = ir->getComposite("foo5");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("00"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("11"), 0.5, 0.05);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void foo6(qbit q) {
            H(q[3]);
            CNOT(q[3], q[2]);
            Measure(q[2]);
            Measure(q[3]);
        })");

        auto program = ir->getComposite("foo6");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        EXPECT_NEAR(qreg->computeMeasurementProbability("00"), 0.5, 0.05);
        EXPECT_NEAR(qreg->computeMeasurementProbability("11"), 0.5, 0.05);
    }

    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void foo7(qbit q) {
            X(q[0]);
            CNOT(q[0], q[1]);
            CNOT(q[1], q[2]);
            CNOT(q[2], q[3]);
            Measure(q[0]);
            Measure(q[1]);
            Measure(q[2]);
            Measure(q[3]);
        })");

        auto program = ir->getComposite("foo7");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("1111"), 1.0, 0.01);
    }
}

TEST(MpsGateTester, checkDistanceQubitGate)
{
    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void bar1(qbit q) {
            X(q[0]);
            CNOT(q[0], q[3]);
            Measure(q[0]);
            Measure(q[1]);
            Measure(q[2]);
            Measure(q[3]);
        })");

        auto program = ir->getComposite("bar1");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(4);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("1001"), 1.0, 0.01);
    }
} 

TEST(MpsGateTester, checkTwoQubits)
{
    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void f1(qbit q) {
            X(q[0]);
            CNOT(q[0], q[1]);
            Measure(q[0]);
            Measure(q[1]);
        })");

        auto program = ir->getComposite("f1");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(2);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("11"), 1.0, 0.01);
    }
    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void f2(qbit q) {
            X(q[1]);
            CNOT(q[1], q[0]);
            Measure(q[0]);
            Measure(q[1]);
        })");

        auto program = ir->getComposite("f2");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(2);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("11"), 1.0, 0.01);
    }
} 

TEST(MpsGateTester, checkSingleQubit)
{
    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void func1(qbit q) {
            X(q[0]);
            Measure(q[0]);
        })");

        auto program = ir->getComposite("func1");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(1);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("1"), 1.0, 0.01);
    }
    {
        auto xasmCompiler = xacc::getCompiler("xasm");
        auto ir = xasmCompiler->compile(R"(__qpu__ void func2(qbit q) {
            H(q[0]);
            Z(q[0]);
            H(q[0]);
            Measure(q[0]);
        })");

        auto program = ir->getComposite("func2");
        auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qreg = xacc::qalloc(1);
        accelerator->execute(qreg, program);
        qreg->print();
        EXPECT_NEAR(qreg->computeMeasurementProbability("1"), 1.0, 0.01);
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
