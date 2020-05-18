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

TEST(MpsGateTester, checkGateiSwap) 
{    
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void iswap1(qbit q) {
        X(q[0]);
        iSwap(q[0], q[1]);
        Measure(q[0]);
        Measure(q[1]);
        Measure(q[2]);
    })");

    auto program = ir->getComposite("iswap1");
    auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 100)});
    auto qreg = xacc::qalloc(3);
    accelerator->execute(qreg, program);
    qreg->print();
    // iSwap is the same as Swap (just added the i phase)
    EXPECT_NEAR(qreg->computeMeasurementProbability("010"), 1.0, 0.01);
}

TEST(MpsGateTester, checkGatefSim) 
{    
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 1000)});
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void testFsim(qbit q, double x, double y) {
        X(q[0]);
        fSim(q[0], q[1], x, y);
        Measure(q[0]);
        Measure(q[1]);
    })", qpu);

    auto program = ir->getComposites()[0]; 
    const auto angles = xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 10);

    for (const auto& a : angles) 
    {
        auto buffer = xacc::qalloc(2);
        auto evaled = program->operator()({ a, 0.0 });
        qpu->execute(buffer, evaled);
        buffer->print();
        const auto expectedProb = std::sin(a) * std::sin(a);
        // fSim mixes 01 and 10 states w.r.t. the theta angle.
        EXPECT_NEAR(buffer->computeMeasurementProbability("01"), expectedProb, 0.1);
        EXPECT_NEAR(buffer->computeMeasurementProbability("10"), 1.0 - expectedProb, 0.1);
    }
}

TEST(MpsGateTester, testDeuteron)
{
    auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void ansatz(qbit q, double t) {
      X(q[0]);
      Ry(q[1], t);
      CX(q[1], q[0]);
      H(q[0]);
      H(q[1]);
      Measure(q[0]);
      Measure(q[1]);
    })", accelerator);

    auto program = ir->getComposite("ansatz");
    // Expected results from deuteron_2qbit_xasm_X0X1
    const std::vector<double> expectedResults {
        0.0,
        -0.324699,
        -0.614213,
        -0.837166,
        -0.9694,
        -0.996584,
        -0.915773,
        -0.735724,
        -0.475947,
        -0.164595,
        0.164595,
        0.475947,
        0.735724,
        0.915773,
        0.996584,
        0.9694,
        0.837166,
        0.614213,
        0.324699,
        0.0
    };

    const auto angles = xacc::linspace(-xacc::constants::pi, xacc::constants::pi, 20);
    // Pick a random angle to test (save the test time)
    const int indexToTest = rand() % angles.size(); 
    
    auto buffer = xacc::qalloc(2);
    auto evaled = program->operator()({ angles[indexToTest] });
    accelerator->execute(buffer, evaled);
    std::cout << "Angle = " <<  angles[indexToTest] << "; exp-val-z = " << buffer->getExpectationValueZ() << "\n";
    EXPECT_NEAR(buffer->getExpectationValueZ(), expectedResults[indexToTest], 0.05);
}

TEST(MpsGateTester, testGrover) 
{
    // Test Grover's algorithm
    // Amplify the amplitude of number 6 (110) state 
    const auto generateGroverSrc = [](const std::string& in_name) {
        return ("__qpu__ void " + in_name).append(R"((qbit q) { 
            H(q[0]);
            H(q[1]);
            H(q[2]);
            X(q[0]);
            H(q[2]);
            H(q[2]);
            CNOT(q[1], q[2]);
            Tdg(q[2]);
            CNOT(q[0], q[2]);
            T(q[2]);
            CNOT(q[1], q[2]);
            Tdg(q[2]);
            CNOT(q[0], q[2]);
            T(q[2]);
            H(q[2]);
            T(q[1]);
            CNOT(q[0], q[1]);
            T(q[0]);
            Tdg(q[1]);
            CNOT(q[0], q[1]);
            X(q[0]);
            H(q[2]);
            H(q[0]);
            H(q[1]);
            H(q[2]);
            X(q[0]);
            X(q[1]);
            X(q[2]);
            H(q[2]);
            H(q[2]);
            CNOT(q[1], q[2]);
            Tdg(q[2]);
            CNOT(q[0], q[2]);
            T(q[2]);
            CNOT(q[1], q[2]);
            Tdg(q[2]);
            CNOT(q[0], q[2]);
            T(q[2]);
            H(q[2]);
            T(q[1]);
            CNOT(q[0], q[1]);
            T(q[0]);
            Tdg(q[1]);
            CNOT(q[0], q[1]);
            H(q[2]);
            X(q[0]);
            X(q[1]);
            X(q[2]);      
            H(q[0]);
            H(q[1]);
            H(q[2]);
            Measure(q[2]);
            Measure(q[1]);
            Measure(q[0]);
        })");
    };    
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(generateGroverSrc("testGrover1"), nullptr);
    // 3-qubit
    {
        auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
        auto qubitReg = xacc::qalloc(3);
        auto program = ir->getComposites()[0];   
        qpu->execute(qubitReg, program);
        qubitReg->print();
        
        // Expected result: |110> is amplified to about 70% probability 
        const auto resultProb = qubitReg->computeMeasurementProbability("110"); 
        EXPECT_GT(resultProb, 0.5);  // lower bound: 50%
    }

    // Embed into a 4-qubit register:
    // This will test middle tensors inside the tensor train
    // {
    //     auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps"), std::make_pair("shots", 10000)});
    //     auto qubitReg = xacc::qalloc(4);
    //     auto program = ir->getComposites()[0];   
    //     qpu->execute(qubitReg, program);
    //     qubitReg->print();
        
    //     // Expected result: |110> is amplified to about 70% probability 
    //     const auto resultProb = qubitReg->computeMeasurementProbability("110"); 
    //     EXPECT_GT(resultProb, 0.5);  // lower bound: 50%
    // } 
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 
