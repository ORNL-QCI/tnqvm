#include <memory>
#include <gtest/gtest.h>
#include "xacc.hpp"
#include "xacc_service.hpp"

TEST(MpsOverMpiTester, checkSimple) 
{    
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
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
    auto qubitReg = xacc::qalloc(3);
    auto program = ir->getComposites()[0];   
    qpu->execute(qubitReg, program);
    qubitReg->print();
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
} 