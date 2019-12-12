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

using namespace tnqvm;
using namespace xacc::quantum;

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
  const auto getExpectedValue = [](int in_qubitIndex, AcceleratorBuffer& in_buffer) -> double {
    return  mpark::get<double>(in_buffer.getInformation("exp-val-z"));
  };
  
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
    EXPECT_NEAR(1.0, getExpectedValue(0, *qubitReg), 1e-12);
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
    EXPECT_NEAR(0.0, getExpectedValue(0, *qubitReg), 1e-12);
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
    // |1> state -> expected value is 1.0
    EXPECT_NEAR(-1.0, getExpectedValue(0, *qubitReg), 1e-12);
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