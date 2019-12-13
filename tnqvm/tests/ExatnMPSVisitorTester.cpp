/***********************************************************************************
 * Copyright (c) 2017, UT-Battelle
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the xacc nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Contributors:
 *   Initial API and implementation - Alex McCaskey
 *
 **********************************************************************************/
#include <memory>
#include <gtest/gtest.h>
#include "TNQVM.hpp"
#include "xacc.hpp"
#include "base/Gates.hpp"
#include "utils/GateMatrixAlgebra.hpp"

using namespace tnqvm;
using namespace xacc::quantum;

namespace {
  inline double getExpectedValue(AcceleratorBuffer& in_buffer) {
    return  in_buffer["exp-val-z"].as<double>();
  };
}

// This test is just to confirm that the ExaTN backend can be instaniated
// and we can submit quantum circuit as tensors to the ExaTN backend. 
TEST(ExatnMPSVisitorTester, checkExatnMPSVisitor) {
  TNQVM acc;

  acc.initialize({std::make_pair("tnqvm-visitor", "exatn-mps")});  
  EXPECT_EQ(acc.getVisitorName(), "exatn-mps"); 

  auto qreg1 = xacc::qalloc(3);       // 3-qubit accelerator buffer
  auto provider = xacc::getIRProvider("quantum");
 
  auto f = provider->createComposite("foo", {}); // gate function

  auto x1 = provider->createInstruction(GetGateName(CommonGates::X), 0);
  auto h1 = provider->createInstruction(GetGateName(CommonGates::H), 1);
  auto cn1 = provider->createInstruction(GetGateName(CommonGates::CNOT), { 1, 2 });
  auto cn2 = provider->createInstruction(GetGateName(CommonGates::CNOT), { 0, 1 });
  auto h2 = provider->createInstruction(GetGateName(CommonGates::H), 0);
  
  f->addInstruction(x1);
  f->addInstruction(h1);
  f->addInstruction(cn1);
  f->addInstruction(cn2);
  f->addInstruction(h2);
  
  EXPECT_NO_THROW(acc.execute(qreg1, f));
}

TEST(ExatnMPSVisitorTester, testSimpleGates) {
  {
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
    auto qubitReg = xacc::qalloc(1);
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test1(qbit q) {
      Measure(q[0]);
    })", qpu);
 
    auto program = ir->getComposites()[0];   
    qpu->execute(qubitReg, program);
    // Initial qubit in |0> state -> expected value is 1.0
    EXPECT_NEAR(1.0, getExpectedValue(*qubitReg), 1e-12);
  }

  {
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
    auto qubitReg = xacc::qalloc(1);
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test2(qbit q) {
      H(q[0]);
      Measure(q[0]);
    })", qpu);
 
    auto program = ir->getComposites()[0];   
    qpu->execute(qubitReg, program);
    // |+> state -> expected value is 0.0
    EXPECT_NEAR(0.0, getExpectedValue(*qubitReg), 1e-12);
  }

  {
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
    auto qubitReg = xacc::qalloc(1);
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test3(qbit q) {
      X(q[0]);
      Measure(q[0]);
    })", qpu);
 
    auto program = ir->getComposites()[0];   
    qpu->execute(qubitReg, program);
    // |1> state -> expected value is -1.0
    EXPECT_NEAR(-1.0, getExpectedValue(*qubitReg), 1e-12);
  }

  {
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
    auto qubitReg = xacc::qalloc(3);
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test4(qbit q) {
      X(q[0]);
      CNOT(q[0], q[1]);
      Measure(q[1]);
    })", qpu);

    
    auto program = ir->getComposites()[0];   
    qpu->execute(qubitReg, program);
    // Expected state: |11>
    // Measure q1 should return -1
    EXPECT_NEAR(-1.0, getExpectedValue(*qubitReg), 1e-12);
  }

  {
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
    auto qubitReg = xacc::qalloc(3);
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test5(qbit q) {
      X(q[0]);
      X(q[2]);
      CNOT(q[1], q[2]);
      Measure(q[2]);
    })", qpu);
    
    // q1 is in |0> state => CNOT is not active.
    auto program = ir->getComposites()[0];   
    qpu->execute(qubitReg, program);
    EXPECT_NEAR(-1.0, getExpectedValue(*qubitReg), 1e-12);
  }

  {
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
    auto qubitReg = xacc::qalloc(3);
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void test6(qbit q) {
      H(q[2]);
      CNOT(q[2], q[1]);
      CNOT(q[1], q[0]);
      Measure(q[2]);
    })", qpu);
 
    auto program = ir->getComposites()[0];   
    qpu->execute(qubitReg, program);
    // GHZ state -> expected value is 0.0
    EXPECT_NEAR(0.0, getExpectedValue(*qubitReg), 1e-12);
  }
}

TEST(ExatnMPSVisitorTester, testMeasurement) {  
  const int nbTrials = 100;
  {
    auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
    auto qubitReg = xacc::qalloc(3);
    auto xasmCompiler = xacc::getCompiler("xasm");
    auto ir = xasmCompiler->compile(R"(__qpu__ void testMeasure1(qbit q) {
      H(q[2]);
      CNOT(q[2], q[1]);
      CNOT(q[1], q[0]);
      Measure(q[2]);
      Measure(q[1]);
      Measure(q[0]);
    })", qpu);
    
    auto program = ir->getComposites()[0];   
    for (int i = 0; i < nbTrials; ++i)
    {
      qpu->execute(qubitReg, program);
    }

    // This is GHZ state: |000> + |111>
    // hence we should only get 2 results
    EXPECT_EQ(qubitReg->getMeasurementCounts().size(), 2);
    // Allow for some random variations
    const int minCount = 0.4 * nbTrials;
    const int maxCount = 0.6 * nbTrials;
    for (const auto& resultToCount : qubitReg->getMeasurementCounts())
    {
      EXPECT_TRUE(resultToCount.first == "000" || resultToCount.first == "111");
      EXPECT_TRUE(resultToCount.second > minCount && resultToCount.second < maxCount);
    }
  }
}

TEST(ExatnMPSVisitorTester, checkDeuteuron) {
  auto accelerator = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn-mps")});
  // Make sure this is ExaTN
  EXPECT_EQ(std::static_pointer_cast<tnqvm::TNQVM>(accelerator)->getVisitorName(), "exatn-mps"); 
  auto xasmCompiler = xacc::getCompiler("xasm");
  
  const auto calculateExpectedResult = [](double in_theta) -> double {
    // Create a 2-qubit state vector for validation 
    auto expectedStateVector = AllocateStateVector(2);
    ApplySingleQubitGate(expectedStateVector, 0, GetGateMatrix<CommonGates::X>());
    ApplySingleQubitGate(expectedStateVector, 1, GetGateMatrix<CommonGates::Ry>(in_theta));
    ApplyCNOTGate(expectedStateVector, 1, 0);
    ApplySingleQubitGate(expectedStateVector, 0, GetGateMatrix<CommonGates::H>());
    ApplySingleQubitGate(expectedStateVector, 1, GetGateMatrix<CommonGates::H>());
    const bool result = ApplyMeasureOp(expectedStateVector, 0);
    if (result)
    {
      // Q0 is 1 => 01 and 11  
      EXPECT_NEAR(std::norm(expectedStateVector[0]), 0.0, 1e-12);
      EXPECT_NEAR(std::norm(expectedStateVector[2]), 0.0, 1e-12);
      return std::norm(expectedStateVector[1]) - std::norm(expectedStateVector[3]);     
    }
    else
    {
      // Q0 is 0 => 00 and 10
      EXPECT_NEAR(std::norm(expectedStateVector[1]), 0.0, 1e-12);
      EXPECT_NEAR(std::norm(expectedStateVector[3]), 0.0, 1e-12);
      return std::norm(expectedStateVector[0]) - std::norm(expectedStateVector[2]); 
    }
  };  
  
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

  const auto angles = linspace(-xacc::constants::pi, xacc::constants::pi, 20);
  for (const auto &a : angles) {
    auto buffer = xacc::qalloc(2);
    auto evaled = program->operator()({a});
    accelerator->execute(buffer, evaled);
    EXPECT_NEAR(std::abs(getExpectedValue(*buffer)), std::abs(calculateExpectedResult(a)), 1e-12);
  }
}

int main(int argc, char **argv) 
{
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
  xacc::Finalize();
} 